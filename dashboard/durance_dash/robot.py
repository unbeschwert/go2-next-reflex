"""Manual robot control: one dead-man teleop session (on the board over SSH, or local for the simulator)."""
from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from pathlib import Path

from fastapi import HTTPException

from .board import SCRIPTS
from .errors import need
from .sshrun import q

# name: (api_id, label, group, confirm, note)
COMMANDS = {
    "stop": (1003, "Stop moving", "Motion", False, "StopMove"),
    "standup": (1004, "Stand up", "Posture", False, "locked joints"),
    "balance": (1002, "Balance stand", "Posture", False, ""),
    "sit": (1009, "Sit", "Posture", False, ""),
    "risesit": (1010, "Rise from sit", "Posture", False, ""),
    "standdown": (1005, "Lie down", "Posture", True, "the body drops to the floor"),
    "recovery": (1006, "Recovery stand", "Posture", True, "get up after a fall: needs free space around the dog"),
    "hello": (1016, "Hello (wave)", "Tricks", False, ""),
    "stretch": (1017, "Stretch", "Tricks", False, ""),
    "content": (1020, "Content", "Tricks", False, ""),
    "heart": (1036, "Finger heart", "Tricks", False, ""),
    "dance1": (1022, "Dance 1", "Tricks", True, "moves around: keep 1.5 m clear"),
    "dance2": (1023, "Dance 2", "Tricks", True, "moves around: keep 1.5 m clear"),
    "scrape": (1029, "Rear up and greet", "Tricks", True, "stands on the hind legs"),
    "damp": (1001, "Damp (motors limp)", "Danger", True, "the robot COLLAPSES: only when it is lying down"),
}
LOCAL_PY = "/usr/bin/python3"


class Robot:
    def __init__(self, cfg, board, sim, live, settings, audit, broker):
        self.cfg, self.board, self.sim, self.live, self.settings = cfg, board, sim, live, settings
        self.audit, self.broker = audit, broker
        self.proc: asyncio.subprocess.Process | None = None
        self.link: str = "none"
        self.manual = False
        self.prev_paused: bool | None = None
        self.last_cmd: dict | None = None
        self.state: dict = {}
        self.ready = asyncio.Event()
        self._last_move = 0.0
        self._reader: asyncio.Task | None = None
        self.error: str | None = None
        self.estop_active = False

    @property
    def limits(self) -> dict:
        r = self.cfg.robot
        return {"vx": r.max_vx, "vy": r.max_vy, "wz": r.max_wz}

    def teleop_alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def current_link(self) -> str:
        if self.live.source == "sim":
            return "sim"
        st = await self.board.raw_status()
        if not st.get("ssh"):
            return "none"
        mode = str(st.get("mode", ""))
        if mode == "test":
            return "rviz"
        return "go2" if (st.get("helpers") or {}).get("robot_link") else "none"

    def view(self) -> dict:
        return {"manual": self.manual, "link": self.link, "teleop": self.teleop_alive(), "limits": self.limits,
                "deadman_s": self.cfg.robot.deadman_s, "last_cmd": self.last_cmd, "robot_state": self.state,
                "error": self.error}

    def publish(self) -> None:
        self.broker.publish("robot", self.view())

    # ---------------------------------------------------------------- teleop process
    def _teleop_code(self) -> str:
        code = base64.b64encode((SCRIPTS / "teleop.py").read_bytes()).decode()
        return f"import base64;exec(base64.b64decode({code!r}))"

    async def _spawn(self, link: str) -> None:
        r = self.cfg.robot
        args = [str(r.max_vx), str(r.max_vy), str(r.max_wz), str(r.deadman_s)]
        if link == "sim":
            env = self.sim.env(self.cfg.sim.ros_domain)
            self.proc = await asyncio.create_subprocess_exec(
                LOCAL_PY, "-u", "-c", self._teleop_code(), *args, env=env, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        else:
            env = self.cfg.board.robot_env if link == "go2" else self.cfg.board.test_env
            run = f"exec python3 -u -c {q(self._teleop_code())} {q(*args)}"
            self.proc = await self.board.ssh.spawn(f"bash -c {q(f'source {q(env)} >/dev/null 2>&1 || exit 3; {run}')}")
        self.ready.clear()
        self._reader = asyncio.create_task(self._read(self.proc))

    async def _read(self, proc) -> None:
        assert proc.stdout
        async for raw in proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            t = msg.get("type")
            if t == "ready":
                self.ready.set()
            elif t == "state":
                self.state = {k: msg.get(k) for k in ("moving", "yaw", "pos", "mode", "body_height", "vel")}
                self.publish()
            elif t == "deadman":
                self.last_cmd = {"name": "deadman stop", "t": time.time()}
                self.publish()
            elif t == "error":
                self.error = msg.get("text")
                self.publish()
        await proc.wait()
        err = (await proc.stderr.read()).decode(errors="replace").strip() if proc.stderr else ""
        if self.manual and proc is self.proc:
            self.error = ("teleop ended" + (f" (exit {proc.returncode})" if proc.returncode else "")
                          + (": ROS environment missing on the board" if proc.returncode == 3 else "")
                          + (f": {err[-200:]}" if err and proc.returncode not in (0, 3) else ""))
            self.manual = False
            self.broker.publish("alert", {"level": "danger", "text": f"manual control stopped: {self.error}"})
            self.publish()

    async def _send(self, obj: dict) -> None:
        if not self.teleop_alive() or self.proc.stdin is None:
            raise HTTPException(409, "manual control is not active")
        try:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            await asyncio.wait_for(self.proc.stdin.drain(), 1.0)
        except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
            raise HTTPException(503, "teleop link lost")

    async def _close(self) -> None:
        p = self.proc
        if p is None:
            return
        if p.returncode is None:
            try:
                if p.stdin:
                    p.stdin.write(b'{"op": "stop"}\n')
                    await asyncio.wait_for(p.stdin.drain(), 1.0)
                    p.stdin.close()                 # teleop sends StopMove x3 and exits
                await asyncio.wait_for(p.wait(), 3.0)
            except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                p.kill()
        self.proc = None

    # ---------------------------------------------------------------- API
    async def set_manual(self, user: str, enabled: bool, confirm: bool) -> dict:
        if not enabled:
            if self.manual or self.teleop_alive():
                self.manual = False
                await self._close()
                if self.prev_paused is not None and not self.estop_active:
                    await self.settings.set_paused(user, self.prev_paused, "manual control off")
                self.prev_paused = None
                self.audit.log(user, "robot.manual.off")
            self.publish()
            return self.view()
        if self.manual and self.teleop_alive():
            return self.view()
        link = await self.current_link()
        if link == "none":
            raise HTTPException(409, "no robot link: board not reachable, or /root/robot_link.sh missing")
        parts = []
        if link == "go2":
            parts.append("Manual control of the REAL Go2. Gesture control will be paused. Keep the area around the "
                         "dog clear and a hand on the E-STOP.")
        if self.live.source == "board":
            from .board import parse_lock
            lock = parse_lock((await self.board.raw_status(0)).get("lock"))
            if lock and not lock["mine"]:
                parts.append(f"The board is locked by {lock['owner']} ({lock['purpose'] or '-'}); the lock will be taken over.")
        if parts:
            need(confirm, " ".join(parts) + " Continue?")
        if self.live.source == "board":
            await self.board.ensure_lock(user, "manual robot control", confirm=True)
        cur = (await self.settings.get())["values"]
        self.prev_paused = bool(cur.get("paused"))
        await self.settings.set_paused(user, True, "manual control", bypass_lock=True)
        self.link, self.error = link, None
        await self._spawn(link)
        try:
            await asyncio.wait_for(self.ready.wait(), 12.0)
        except asyncio.TimeoutError:
            await self._close()
            raise HTTPException(504, "teleop did not start within 12 s (ROS environment / unitree messages on the target?)")
        self.manual = True
        self.audit.log(user, "robot.manual.on", {"link": link})
        self.publish()
        return self.view()

    async def move(self, vx: float, vy: float, wz: float) -> dict:
        if not self.manual:
            raise HTTPException(409, "enable manual mode first")
        now = time.monotonic()
        if now - self._last_move < 0.03:            # at most ~30 commands per second
            return {"ok": True, "throttled": True}
        self._last_move = now
        lim = self.limits

        def c(v, m):
            v = float(v)
            return 0.0 if not math.isfinite(v) else max(-m, min(m, v))
        v = {"vx": c(vx, lim["vx"]), "vy": c(vy, lim["vy"]), "wz": c(wz, lim["wz"])}
        await self._send({"op": "move", **v})
        self.last_cmd = {"name": "move", **v, "t": time.time()}
        return {"ok": True, "clamped": v}

    async def stop(self) -> dict:
        if self.teleop_alive():
            await self._send({"op": "stop"})
            self.last_cmd = {"name": "stop", "t": time.time()}
        return {"ok": True}

    async def command(self, user: str, name: str, confirm: bool) -> dict:
        if name not in COMMANDS:
            raise HTTPException(422, f"unknown command {name}")
        if not self.manual:
            raise HTTPException(409, "enable manual mode first")
        api_id, label, _, risky, note = COMMANDS[name]
        if risky:
            need(confirm, f"{label}: {note}. Send it to the {'REAL Go2' if self.link == 'go2' else self.link}?")
        await self._send({"op": "stop"} if api_id == 1003 else {"op": "api", "id": api_id})
        self.last_cmd = {"name": name, "label": label, "t": time.time()}
        self.audit.log(user, "robot.command", {"name": name, "api_id": api_id, "link": self.link})
        self.publish()
        return {"ok": True, "sent": label}

    async def estop(self) -> str | None:
        """StopMove burst through the open teleop session (fastest path). Returns what was done."""
        if self.teleop_alive():
            try:
                await self._send({"op": "estop"})
                return f"StopMove burst sent ({self.link})"
            except Exception:  # noqa: BLE001 - fall back to the other stop paths
                return None
        return None

    async def shutdown(self) -> None:
        self.manual = False
        await self._close()

    def commands(self) -> dict:
        return {"commands": [{"name": n, "api_id": a, "label": lbl, "group": g, "confirm": c, "note": note}
                             for n, (a, lbl, g, c, note) in COMMANDS.items()]}
