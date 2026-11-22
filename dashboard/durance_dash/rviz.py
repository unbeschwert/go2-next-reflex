"""Real RViz in the durance-rviz container, shown in the browser through an authenticated VNC WebSocket."""
from __future__ import annotations

import asyncio
import os
import secrets
import time
from pathlib import Path

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from .config import project_file
from .sshrun import run_local

IMAGE = "durance-rviz:1"
NAME = "durance-rviz"
CONTEXT = Path(__file__).resolve().parent.parent / "docker" / "rviz"
VNC_PORT = 5901
WIDTH, HEIGHT = 1600, 900
ENV_KEYS = ("ROS_DOMAIN_ID", "CYCLONEDDS_URI", "ROS_AUTOMATIC_DISCOVERY_RANGE")


class RViz:
    def __init__(self, cfg, sim, tasks, audit, broker):
        self.cfg, self.sim, self.tasks, self.audit, self.broker = cfg, sim, tasks, audit, broker
        self.password: str | None = None
        self.source: str | None = None
        self.started: float | None = None
        self.build_task: str | None = None
        self.error: str | None = None
        self._docker: tuple[float, bool, str | None] | None = None

    async def docker(self, *args: str, timeout: float = 60, log=None):
        try:
            return await run_local(["docker", *args], timeout=timeout, log=log)
        except FileNotFoundError:
            from .sshrun import Result
            return Result(127, "", "docker is not installed", 0.0)

    async def docker_ok(self) -> tuple[bool, str | None]:
        if self._docker and time.monotonic() - self._docker[0] < 15:
            return self._docker[1], self._docker[2]
        r = await self.docker("version", "--format", "{{.Server.Version}}", timeout=10)
        if r.ok:
            res = (True, None)
        elif "permission denied" in r.out.lower():
            res = (False, "this user may not use Docker: run `sudo usermod -aG docker $USER`, then log out and in "
                          "(and restart the dashboard)")
        elif r.rc == 127:
            res = (False, "Docker is not installed")
        else:
            res = (False, (r.out or r.err).strip()[-300:] or "Docker daemon not reachable")
        self._docker = (time.monotonic(), *res)
        return res

    async def state(self) -> dict:
        ok, detail = await self.docker_ok()
        image = {"name": IMAGE, "present": False, "build_task": self.build_task}
        container = {"running": False, "source": None, "started": None, "error": self.error}
        if ok:
            image["present"] = (await self.docker("image", "inspect", "--format", "{{.Id}}", IMAGE, timeout=10)).ok
            r = await self.docker("inspect", "--format", '{{.State.Running}} {{index .Config.Labels "durance.source"}}',
                                  NAME, timeout=10)
            if r.ok:
                running, _, source = r.out.strip().partition(" ")
                container.update(running=running == "true", source=source or None, started=self.started)
        running = container["running"] and self.password is not None
        if container["running"] and self.password is None:
            container["error"] = "RViz is running from an earlier dashboard session: press Start to reconnect it"
        return {"docker": {"ok": ok, "detail": detail}, "image": image, "container": container,
                "vnc": {"path": "/api/rviz/vnc", "password": self.password, "width": WIDTH, "height": HEIGHT}
                if running else None}

    async def build(self, user: str) -> str:
        ok, detail = await self.docker_ok()
        if not ok:
            raise HTTPException(503, f"Docker not usable: {detail}")

        async def run(log):
            log(f"$ docker build -t {IMAGE} {CONTEXT}   (first build downloads ROS 2 Jazzy, several minutes)")
            r = await self.docker("build", "-t", IMAGE, str(CONTEXT), timeout=3600, log=log)   # legacy builder: no --progress
            self.build_task = None
            r.check("docker build")
            self.audit.log(user, "rviz.build", {"image": IMAGE})
            return {"image": IMAGE}
        t = self.tasks.start("rviz.build", "Build RViz image", user, run, exclusive="rviz.build")
        self.build_task = t.id
        return t.id

    async def start(self, user: str, source: str) -> dict:
        if source not in ("sim", "board-test"):
            raise HTTPException(422, "source must be sim or board-test")
        ok, detail = await self.docker_ok()
        if not ok:
            raise HTTPException(503, f"Docker not usable: {detail}")
        if not (await self.docker("image", "inspect", IMAGE, timeout=10)).ok:
            raise HTTPException(409, "the RViz image is not built yet")
        domain = self.cfg.sim.ros_domain if source == "sim" else self.cfg.sim.board_test_domain
        env = self.sim.env(domain)
        if source == "board-test":
            await self.sim.ensure_board_test_animator()
        urdf = project_file(self.cfg.paths, "go2.urdf")
        if not urdf.exists():
            raise HTTPException(409, "Go2 model (go2.urdf) not found: run ros2/rviz/fetch_go2_model.sh once")
        await self.stop(user, quiet=True)
        password = secrets.token_urlsafe(6)[:8]                # VNC passwords are at most 8 characters
        args = ["run", "-d", "--rm", "--name", NAME, "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
                "--label", f"durance.source={source}", "--security-opt", "no-new-privileges:true", "--cap-drop", "ALL",
                "--read-only", "--tmpfs", "/tmp:rw,size=256m",
                "-v", f"{urdf.parent}:/go2_viz:ro",
                # the launch file from this checkout (no image rebuild when it changes)
                "-v", f"{CONTEXT / 'view_rviz.launch.py'}:/opt/durance/view_rviz.launch.py:ro",
                "-e", f"VNC_PASSWORD={password}", "-e", f"VNC_PORT={VNC_PORT}", "-e", f"WIDTH={WIDTH}", "-e", f"HEIGHT={HEIGHT}"]
        for k in ENV_KEYS:
            if env.get(k):
                args += ["-e", f"{k}={env[k]}"]
        r = await self.docker(*args, IMAGE, timeout=60)
        if not r.ok:
            self.error = (r.out or r.err).strip()[-400:]
            raise HTTPException(502, f"docker run failed: {self.error}")
        self.password, self.source, self.started, self.error = password, source, time.time(), None
        for _ in range(60):                                     # wait for the VNC server
            try:
                _, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", VNC_PORT), 1)
                w.close()
                break
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(0.25)
        else:
            logs = (await self.docker("logs", "--tail", "30", NAME, timeout=10)).out
            self.error = f"VNC server did not come up: {logs[-400:]}"
        self.audit.log(user, "rviz.start", {"source": source})
        return await self.state()

    async def stop(self, user: str | None = None, quiet: bool = False) -> dict:
        r = await self.docker("inspect", NAME, timeout=10)
        if r.ok:
            await self.docker("rm", "-f", NAME, timeout=30)
        self.password, self.source, self.started = None, None, None
        if user and not quiet:
            self.audit.log(user, "rviz.stop")
        return await self.state()

    async def shutdown(self) -> None:
        if self.password is not None:
            await self.docker("rm", "-f", NAME, timeout=20)

    # ---------------------------------------------------------------- VNC over WebSocket
    async def proxy(self, ws: WebSocket) -> None:
        if self.password is None:
            await ws.close(code=1011, reason="RViz is not running")
            return
        proto = "binary" if "binary" in (ws.scope.get("subprotocols") or []) else None
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", VNC_PORT), 3)
        except (OSError, asyncio.TimeoutError):
            await ws.close(code=1011, reason="RViz VNC server not reachable")
            return
        await ws.accept(subprotocol=proto)

        async def ws_to_tcp():
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                data = msg.get("bytes") or (msg.get("text") or "").encode()
                if data:
                    writer.write(data)
                    await writer.drain()

        async def tcp_to_ws():
            while True:
                data = await reader.read(65536)
                if not data:
                    return
                await ws.send_bytes(data)

        tasks = [asyncio.create_task(ws_to_tcp()), asyncio.create_task(tcp_to_ws())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()
            writer.close()
            try:
                await ws.close()
            except RuntimeError:
                pass
