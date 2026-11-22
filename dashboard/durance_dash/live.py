"""Live gesture-bridge status / log events and the camera view, from the board (SSH) or the simulator."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import httpx
import numpy as np

from .board import py_cmd

KINDS = [("command", re.compile(r"ROBOT COMMAND|DRY-RUN would send")), ("won", re.compile(r"\bwon\b")),
         ("ready", re.compile(r"ready for the next gesture")), ("held", re.compile(r"\bHELD\b")),
         ("turn", re.compile(r"^\[[^\]]*\] turn |follow mode")),
         ("error", re.compile(r"Traceback|Error|error|lost|FAILED|failed"))]
BOUNDARY = b"frame"


def classify(line: str) -> str:
    for kind, rx in KINDS:
        if rx.search(line):
            return kind
    return "info"


def placeholder(text: str) -> bytes:
    img = np.full((240, 320, 3), 24, np.uint8)
    for i, part in enumerate([text[k:k + 38] for k in range(0, min(len(text), 152), 38)]):
        cv2.putText(img, part, (8, 100 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
    cv2.putText(img, "camera not available", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 160, 255), 1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def mjpeg_part(jpg: bytes) -> bytes:
    return b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n"


class Live:
    def __init__(self, cfg, broker, board):
        self.cfg, self.broker, self.board = cfg, broker, board
        self.source = "board"
        self.sim_dir = cfg.state_dir / "sim"
        self.proc: asyncio.subprocess.Process | None = None
        self.last: dict | None = None
        self.last_t = 0.0
        self.last_hb = 0.0
        self.last_command: dict | None = None
        self.logs: deque = deque(maxlen=300)
        self.error: str | None = None
        self._wake = asyncio.Event()
        self._forward_ok = 0.0

    # ---------------------------------------------------------------- status stream
    def set_source(self, source: str) -> None:
        if source != self.source:
            self.source = source
            self.last, self.last_t = None, 0.0
            if self.proc and self.proc.returncode is None:
                self.proc.kill()
            self._wake.set()

    async def run(self) -> None:
        backoff = 1.0
        while True:
            self._wake.clear()
            try:
                if self.source == "board":
                    self.proc = await self.board.ssh.spawn(py_cmd("status_tap.py", self.cfg.board.status_file, "/tmp/bridge.log"))
                else:
                    self.sim_dir.mkdir(parents=True, exist_ok=True)
                    self.proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-u", str(Path(__file__).parent / "board_scripts" / "status_tap.py"),
                        str(self.sim_dir / "bridge_status.json"), str(self.sim_dir / "bridge.log"),
                        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                started = time.monotonic()
                assert self.proc.stdout
                async for raw in self.proc.stdout:
                    self.handle(raw.decode(errors="replace"))
                    backoff = 1.0
                err = (await self.proc.stderr.read()).decode(errors="replace").strip() if self.proc.stderr else ""
                await self.proc.wait()
                if time.monotonic() - started < 3 and err:
                    self.error = err[-300:]
            except Exception as e:  # noqa: BLE001 - keep the stream alive whatever happens
                self.error = str(e)
            self.publish_disconnected()
            try:
                await asyncio.wait_for(self._wake.wait(), backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 10.0)

    def handle(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            return
        kind = msg.get("type")
        now = time.time()
        if kind == "status":
            d = msg.get("data") or {}
            self.last, self.last_t, self.error = d, now, None
            self.broker.publish("live", self.view(d))
        elif kind == "log":
            text = str(msg.get("line", ""))[:2000]
            entry = {"t": now, "source": "sim" if self.source == "sim" else "bridge", "kind": classify(text),
                     "line": text, "history": bool(msg.get("history"))}
            self.logs.append(entry)
            if entry["kind"] == "command" and not entry["history"]:
                label = re.sub(r"^.*(?:ROBOT COMMAND ->|would send ->)\s*", "", text).split(" (")[0].split(":")[0]
                self.last_command = {"label": label.strip(), "t": now}
            self.broker.publish("log", entry)
            if not entry["history"] and time.time() - self.last_t > 3.0:
                self.fallback_rates(text)
        elif kind == "hb":
            self.last_hb = now
            if self.last and now - self.last_t > 2.0:      # stream alive, bridge not writing: bridge stopped
                self.publish_disconnected()

    RATES_RE = re.compile(r"rates\s+o0=([\d.]+)\s+o1=([\d.]+)\s+o2=([\d.]+)\s+o3=([\d.]+)\s+o4=([\d.]+)"
                          r".*?frames=\d+\s+\((\d+)/s camera, (\d+) in window\)")

    def fallback_rates(self, line: str) -> None:
        """Bridges older than the status file (2026-11-22) only log their rates every 2 s: use those."""
        m = self.RATES_RE.search(line)
        if not m:
            return
        rates = [float(m.group(i)) for i in range(1, 6)]
        self.broker.publish("live", {"t": time.time(), "connected": True, "rates": rates, "labels": [],
                                     "fps": int(m.group(6)), "window_frames": int(m.group(7)), "status": "listening",
                                     "candidate": None, "paused": False, "thresholds": {}, "source": self.source,
                                     "coarse": True, "error": "old bridge on the board: live values every 2 s only "
                                                              "(Board page: Install board software)"})

    def view(self, d: dict) -> dict:
        return {"t": d.get("t", time.time()), "connected": bool(d.get("connected")), "rates": d.get("rates", [0] * 5),
                "labels": d.get("labels", []), "fps": d.get("fps", 0), "window_frames": d.get("window_frames", 0),
                "status": d.get("status", "no-board"), "candidate": d.get("candidate"), "candidate_s": d.get("candidate_s", 0),
                "last_trick": d.get("last_trick"), "busy_s": d.get("busy_s", 0), "paused": bool(d.get("paused")),
                "thresholds": d.get("thresholds", {}), "mode": d.get("mode"), "dry_run": d.get("dry_run"),
                "clearance": d.get("clearance"), "source": self.source}

    def publish_disconnected(self) -> None:
        self.broker.publish("live", {"t": time.time(), "connected": False, "status": "no-board", "rates": [0] * 5,
                                     "labels": (self.last or {}).get("labels", []), "fps": 0, "source": self.source,
                                     "error": self.error or "gesture bridge not running"})

    def summary(self) -> dict:
        fresh = self.last is not None and time.time() - self.last_t < 2.0
        d = self.last or {}
        return {"connected": fresh and bool(d.get("connected")), "fps": d.get("fps", 0) if fresh else 0,
                "status": d.get("status", "no-board") if fresh else "no-board", "paused": bool(d.get("paused")) if fresh else None,
                "last_command": self.last_command, "error": None if fresh else (self.error or "gesture bridge not running")}

    # ---------------------------------------------------------------- camera
    async def upstream_url(self) -> tuple[str | None, str]:
        if self.source == "sim":
            return f"http://127.0.0.1:{self.cfg.sim.camera_port}/stream", "simulator not running"
        b = self.cfg.board
        if time.monotonic() - self._forward_ok > 60:
            if not await self.board.ssh.forward(b.local_viewer_port, b.viewer_port):
                return None, "board not reachable over SSH"
            self._forward_ok = time.monotonic()
        return f"http://127.0.0.1:{b.local_viewer_port}/stream", "camera viewer not running on the board (start inference)"

    async def camera(self, is_disconnected):
        """MJPEG generator: upstream frames, or a placeholder frame every second while it is down."""
        while not await is_disconnected():
            url, why = await self.upstream_url()
            if url:
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=10.0)) as client:
                        async with client.stream("GET", url) as r:
                            if r.status_code == 200:
                                async for chunk in r.aiter_bytes():
                                    yield chunk
                                    if await is_disconnected():
                                        return
                except (httpx.HTTPError, OSError):
                    self._forward_ok = 0.0
            yield mjpeg_part(placeholder(why))
            await asyncio.sleep(1.0)

    async def snapshot(self) -> bytes:
        url, why = await self.upstream_url()
        if url:
            try:
                buf = b""
                async with httpx.AsyncClient(timeout=5.0) as client:
                    async with client.stream("GET", url) as r:
                        async for chunk in r.aiter_bytes():
                            buf += chunk
                            a = buf.find(b"\xff\xd8")
                            e = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                            if e > 0:
                                return buf[a:e + 2]
                            if len(buf) > 2_000_000:
                                break
            except (httpx.HTTPError, OSError):
                pass
        return placeholder(why)
