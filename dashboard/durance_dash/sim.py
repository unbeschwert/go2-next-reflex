"""Local simulator (fake board + real gesture bridge + Go2 animator on an isolated ROS domain), the 3D-view
pose stream, and desktop RViz for the simulator or the board's test mode."""
from __future__ import annotations

import asyncio
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi import HTTPException

from . import simcache
from .config import project_file, sim_weights
from .procs import Proc, ros_env

PEOPLE = ["none", "heart", "left", "right", "empty"]
LOCAL = Path(__file__).parent / "local_scripts"


def urdf_model(path: Path) -> dict:
    """Joint tree of the URDF for the browser's lightweight 3D model."""
    root = ET.parse(path).getroot()
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        ax = j.find("axis")
        lim = j.find("limit")
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()],
            "rpy": [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()],
            "axis": [float(v) for v in (ax.get("xyz", "1 0 0") if ax is not None else "1 0 0").split()],
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
        })
    links = [lk.get("name") for lk in root.findall("link")]
    children = {j["child"] for j in joints}
    roots = [lk for lk in links if lk not in children]
    return {"root": "base" if "base" in links else (roots[0] if roots else None), "joints": joints, "links": links}


class Sim:
    def __init__(self, cfg, broker, tasks):
        self.cfg, self.broker, self.tasks = cfg, broker, tasks
        self.dir = cfg.state_dir / "sim"
        self.log_dir = cfg.state_dir / "logs"
        self.p: dict[str, Proc] = {}
        self.person = "none"
        self.view_source = "sim"
        self.rviz_source: str | None = None
        self.building: str | None = None
        self._model: dict | None = None
        self._last_pose = 0.0

    # ---------------------------------------------------------------- environment
    def env(self, domain: int, display: bool = False) -> dict:
        p = self.cfg.paths
        try:
            return ros_env(os.path.expanduser(p.laptop_env), os.path.expanduser(p.board_link_env), domain,
                           os.environ.get("DISPLAY", ":0") if display else None)
        except RuntimeError as e:
            raise HTTPException(503, f"ROS 2 environment not available: {e}")

    def cache_file(self) -> Path:
        w = sim_weights(self.cfg)
        z = Path(self.cfg.paths.share).expanduser() / "own_gestures_s20-30.zip"
        if not w.exists() or not z.exists():
            raise HTTPException(503, f"simulator needs {w} and {z}")
        return simcache.cache_path(self.cfg.state_dir, w, z, self.cfg.sim.sim_session)

    # ---------------------------------------------------------------- lifecycle
    def running(self) -> bool:
        return all(self.p.get(n) and self.p[n].alive for n in ("board", "bridge", "animator"))

    async def start(self, user: str) -> dict:
        if self.running():
            return self.state()
        cache = self.cache_file()
        if not cache.exists():
            if self.building is None:
                w = sim_weights(self.cfg)
                z = Path(self.cfg.paths.share).expanduser() / "own_gestures_s20-30.zip"

                async def build(log):
                    log("building the simulator's replay cache (once, about a minute)")
                    await asyncio.to_thread(simcache.build, project_file(self.cfg.paths, "spleat_sim.py").parent, w, z, self.cfg.sim.sim_session, cache, 6, log)
                    self.building = None
                    await self.start(user)
                    return {"cache": str(cache)}
                try:
                    t = self.tasks.start("sim.cache", "Build simulator cache", user, build, exclusive="sim.cache")
                    self.building = t.id
                except Exception:
                    self.building = None
                    raise
            return self.state()
        self.dir.mkdir(parents=True, exist_ok=True)
        s = self.cfg.sim
        env = self.env(s.ros_domain)
        py = self.cfg.paths.ros_python
        procs = {
            "board": Proc("sim_board", [py, "-u", str(LOCAL / "sim_board.py"), str(cache), str(s.fake_port), str(s.camera_port)],
                          self.log_dir, env=dict(os.environ), on_line=self._board_line),
            "animator": Proc("sim_animator", [py, "-u", str(project_file(self.cfg.paths, "go2_animator.py"))], self.log_dir, env=env),
            "bridge": Proc("sim_bridge", [py, "-u", str(project_file(self.cfg.paths, "live_gesture_bridge.py")), "--host", "127.0.0.1",
                                          "--port", str(s.fake_port), "--classes", "own",
                                          "--settings", str(self.dir / "bridge_settings.json"),
                                          "--status", str(self.dir / "bridge_status.json")],
                           self.log_dir, env=env, cwd=str(project_file(self.cfg.paths, "live_gesture_bridge.py").parent), stdout_file=self.dir / "bridge.log"),
        }
        for name in ("board", "animator"):
            self.p[name] = procs[name]
            await procs[name].start()
        await asyncio.sleep(1.0)                    # fake board listening before the bridge connects
        self.p["bridge"] = procs["bridge"]
        await procs["bridge"].start()
        await self.set_view(self.view_source)
        self.broker.publish("alert", {"level": "info", "text": "simulator started"})
        return self.state()

    async def stop(self) -> dict:
        if "bridge" in self.p:                   # bridge first: no robot commands after this
            await self.p["bridge"].stop()
        await asyncio.gather(*(self.p[n].stop() for n in ("tap", "animator", "board") if n in self.p))
        if self.rviz_source == "sim":
            await self.rviz("stop", "sim")
        return self.state()

    async def shutdown(self) -> None:
        if "bridge" in self.p:
            await self.p["bridge"].stop()
        await asyncio.gather(*(p.stop() for p in list(self.p.values())), return_exceptions=True)

    def _board_line(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            return
        if msg.get("type") == "state":
            self.person = msg.get("person", self.person)

    async def set_person(self, gesture: str) -> dict:
        if gesture not in PEOPLE:
            raise HTTPException(422, f"gesture must be one of {PEOPLE}")
        b = self.p.get("board")
        if not b or not b.alive:
            raise HTTPException(409, "simulator is not running")
        b.write({"person": gesture})
        await b.drain()
        self.person = gesture
        return self.state()

    async def reset(self) -> dict:
        a = self.p.get("animator")
        if a:
            await a.stop()
            await a.start()
        return self.state()

    # ---------------------------------------------------------------- pose tap (3D view)
    async def set_view(self, source: str) -> dict:
        if source not in ("sim", "board-test"):
            raise HTTPException(422, "source must be sim or board-test")
        self.view_source = source
        old = self.p.pop("tap", None)
        if old:
            await old.stop()
        domain = self.cfg.sim.ros_domain if source == "sim" else self.cfg.sim.board_test_domain
        tap = Proc(f"pose_tap_{source}", [self.cfg.paths.ros_python, "-u", str(LOCAL / "pose_tap.py"), source],
                   self.log_dir, env=self.env(domain), on_line=self._pose_line)
        self.p["tap"] = tap
        await tap.start()
        return self.state()

    def _pose_line(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            return
        if not msg.get("stale"):
            self._last_pose = time.time()
            self.broker.publish("pose", msg)

    # ---------------------------------------------------------------- desktop RViz
    async def rviz(self, action: str, source: str) -> dict:
        if source not in ("sim", "board-test") or action not in ("start", "stop"):
            raise HTTPException(422, "action start|stop, source sim|board-test")
        key = f"rviz_{source}"
        if action == "stop":
            if key in self.p:
                await self.p.pop(key).stop()
            if self.rviz_source == source:
                self.rviz_source = None
            return self.state()
        for other in ("sim", "board-test"):
            if f"rviz_{other}" in self.p:
                await self.p.pop(f"rviz_{other}").stop()
        domain = self.cfg.sim.ros_domain if source == "sim" else self.cfg.sim.board_test_domain
        env = self.env(domain, display=True)
        if not env.get("DISPLAY"):
            raise HTTPException(409, "no desktop display: RViz needs to run on the laptop's screen")
        model_dir = self.model_dir()
        env.update(GO2_VIZ=str(model_dir), RVIZ_EMBED="0")
        # robot_state_publisher + RViz only: the simulator (or the board-test animator) moves the robot.
        # (go2_viz/view_go2.launch.py also starts an animator, which would animate every command twice.)
        launch = Path(__file__).resolve().parent.parent / "docker" / "rviz" / "view_rviz.launch.py"
        self.p[key] = Proc(key, ["ros2", "launch", str(launch)], self.log_dir, env=env, cwd=str(model_dir))
        await self.p[key].start()
        if source == "board-test":
            await self.ensure_board_test_animator()
        self.rviz_source = source
        return self.state()

    async def ensure_board_test_animator(self) -> None:
        """The board's test mode commands an animator on the laptop (domain 42): start one if none runs."""
        if any_animator_on(self.cfg.sim.board_test_domain) or (self.p.get("animator42") and self.p["animator42"].alive):
            return
        env = self.env(self.cfg.sim.board_test_domain)
        self.p["animator42"] = Proc("animator_board_test", [self.cfg.paths.ros_python, "-u", str(project_file(self.cfg.paths, "go2_animator.py"))],
                                    self.log_dir, env=env)
        await self.p["animator42"].start()

    # ---------------------------------------------------------------- state
    def model_dir(self) -> Path:
        """Folder with go2.urdf, meshes/ and go2.rviz (go2_viz, or ros2/rviz after fetch_go2_model.sh)."""
        urdf = project_file(self.cfg.paths, "go2.urdf")
        if not urdf.exists():
            raise HTTPException(503, "Go2 model (go2.urdf) not found: run ros2/rviz/fetch_go2_model.sh once")
        return urdf.parent

    def model(self) -> dict:
        if self._model is None:
            self._model = urdf_model(self.model_dir() / "go2.urdf")
        return self._model

    def state(self) -> dict:
        alive = lambda n: bool(self.p.get(n) and self.p[n].alive)  # noqa: E731
        return {"running": self.running(), "person": self.person, "domain": self.cfg.sim.ros_domain,
                "components": {"fake_board": alive("board"), "bridge": alive("bridge"), "animator": alive("animator"),
                               "rviz": alive("rviz_sim") or alive("rviz_board-test"), "tap": alive("tap")},
                "rviz_source": self.rviz_source if (alive("rviz_sim") or alive("rviz_board-test")) else None,
                "view_source": self.view_source, "people": PEOPLE, "building_task": self.building,
                "pose_fresh": time.time() - self._last_pose < 2.0,
                "logs": {n: p.info()["log"] for n, p in self.p.items()}}


def any_animator_on(domain: int) -> bool:
    """Is a go2_animator.py already running on this ROS domain (e.g. started by hand for RViz)?"""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"go2_animator.py" not in cmd:
                continue
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            if f"ROS_DOMAIN_ID={domain}".encode() in env:
                return True
        except OSError:
            continue
    return False
