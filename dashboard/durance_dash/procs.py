"""Local helper processes (simulator, animator, RViz, pose tap, recorder) and their ROS 2 environments."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable


class Proc:
    """One child process in its own process group. stdout lines go to `on_line` and the log file."""

    def __init__(self, name: str, args: list[str], log_dir: Path, env: dict | None = None, cwd: str | None = None,
                 on_line: Callable[[str], None] | None = None, stdout_file: Path | None = None):
        self.name, self.args, self.env, self.cwd, self.on_line = name, args, env, cwd, on_line
        self.log_path = log_dir / f"{name}.log"
        self.stdout_file = stdout_file          # e.g. the bridge log that the status tap follows
        self.p: asyncio.subprocess.Process | None = None
        self.started = 0.0
        self._tasks: list[asyncio.Task] = []

    @property
    def alive(self) -> bool:
        return self.p is not None and self.p.returncode is None

    async def start(self) -> None:
        if self.alive:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.p = await asyncio.create_subprocess_exec(
            *self.args, env=self.env, cwd=self.cwd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        self.started = time.time()
        log = open(self.log_path, "a", buffering=1)
        log.write(f"--- {time.strftime('%H:%M:%S')} start: {' '.join(self.args)[:300]}\n")
        out = open(self.stdout_file, "a", buffering=1) if self.stdout_file else None

        async def pump(stream, is_err):
            async for raw in stream:
                line = raw.decode(errors="replace").rstrip("\n")
                if out is not None and not is_err:
                    out.write(line + "\n")
                else:
                    log.write(line + "\n")
                if self.on_line and not is_err:
                    try:
                        self.on_line(line)
                    except Exception as e:  # noqa: BLE001 - a bad line must not kill the reader
                        log.write(f"(handler error: {e!r})\n")

        async def closer():
            await asyncio.gather(*self._tasks[:2], return_exceptions=True)
            await self.p.wait()
            log.write(f"--- exit {self.p.returncode}\n")
            log.close()
            if out:
                out.close()

        self._tasks = [asyncio.create_task(pump(self.p.stdout, False)), asyncio.create_task(pump(self.p.stderr, True))]
        self._tasks.append(asyncio.create_task(closer()))

    def write(self, obj) -> bool:
        if not self.alive or self.p.stdin is None:
            return False
        try:
            self.p.stdin.write((json.dumps(obj) + "\n").encode())
            return True
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            return False

    async def drain(self) -> None:
        if self.alive and self.p.stdin is not None:
            try:
                await asyncio.wait_for(self.p.stdin.drain(), 2)
            except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                pass

    async def stop(self, timeout: float = 4.0) -> None:
        if not self.alive:
            return
        try:
            if self.p.stdin:
                self.p.stdin.close()               # helpers stop the robot when stdin closes
        except (BrokenPipeError, RuntimeError):
            pass
        try:
            await asyncio.wait_for(self.p.wait(), 0.8)
            return
        except asyncio.TimeoutError:
            pass
        for sig, wait in ((signal.SIGTERM, min(timeout, 2.0)), (signal.SIGKILL, timeout)):
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(self.p.wait(), wait)
                return
            except asyncio.TimeoutError:
                continue                    # e.g. blocked writing to a full pipe: SIGTERM cannot end it

    def info(self) -> dict:
        return {"alive": self.alive, "pid": self.p.pid if self.p else None, "since": self.started,
                "exit": None if self.p is None or self.alive else self.p.returncode, "log": str(self.log_path)}


_env_cache: dict[str, dict] = {}


def sourced_env(script: str) -> dict:
    """Environment after `source script` (cached). Raises RuntimeError if the script fails."""
    if script not in _env_cache:
        if not Path(script).exists():
            raise RuntimeError(f"ROS environment script {script} not found")
        r = subprocess.run(["bash", "-c", f'source "$1" >/dev/null 2>&1 && env -0', "_", script],
                           capture_output=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"sourcing {script} failed")
        env = {}
        for item in r.stdout.split(b"\0"):
            if b"=" in item:
                k, v = item.split(b"=", 1)
                env[k.decode()] = v.decode(errors="replace")
        _env_cache[script] = env
    return dict(_env_cache[script])


def ros_env(laptop_env: str, board_link_env: str, domain: int, display: str | None = None) -> dict:
    """ROS 2 environment for one domain: 42 = the board link (as the existing RViz setup), any other =
    loopback only (the simulator can never talk to the board or the robot)."""
    if domain == 42:
        env = sourced_env(board_link_env)
    else:
        env = sourced_env(laptop_env)
        env.update(ROS_DOMAIN_ID=str(domain), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
        env["CYCLONEDDS_URI"] = (
            f'<CycloneDDS><Domain Id="{domain}"><General><Interfaces><NetworkInterface name="lo"/></Interfaces>'
            '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
            '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
        env.pop("ROS_LOCALHOST_ONLY", None)
    env["PYTHONUNBUFFERED"] = "1"
    if display:
        env["DISPLAY"] = display
    return env
