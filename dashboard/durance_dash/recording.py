"""Cued recording of new gesture data (river-tools capture from the board's camera) and cutting it into clips."""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

from .config import project_file
from .errors import need
from .sshrun import run_local

WORKER = Path(__file__).parent / "local_scripts" / "recorder_worker.py"
DEFAULT_STYLES = "normal,fast,large"
NO_ACTION_STYLES = "normal:2,fast:2,still:1,empty:1"


class Recorder:
    def __init__(self, cfg, board, audit, broker):
        self.cfg, self.board, self.audit, self.broker = cfg, board, audit, broker
        self.captures = Path(cfg.paths.captures).expanduser()
        self.dataset = Path(cfg.paths.data).expanduser()
        self.preview = cfg.state_dir / "record_preview.npy"
        self.proc: asyncio.subprocess.Process | None = None
        self.state: dict = {"running": False, "phase": "idle"}
        self.user = "?"

    def gestures(self) -> dict:
        return json.loads((project_file(self.cfg.paths, "gestures.json")).read_text())

    def options(self) -> dict:
        g = self.gestures()
        return {"classes": [{"id": c["id"], "name": c["name"], "how": c["how"], "styles": c.get("styles"),
                             "active": c.get("active", True)} for c in g["classes"]],
                "defaults": {"minutes": 1.0, "ready_s": 6.0, "switch_s": 3.0, "styles": DEFAULT_STYLES,
                             "no_action": True, "shuffle": True},
                "next_session": self.next_session()}

    def next_session(self) -> str:
        used = [int(p.name[1:]) for p in self.captures.glob("s[0-9]*") if p.name[1:].isdigit()]
        used += [int(p.name) for p in self.dataset.glob("[0-9]*") if p.name.isdigit()]
        return "%02d" % max(max(used, default=19) + 1, 20)

    def sessions(self) -> dict:
        out = []
        for d in sorted(self.captures.glob("s[0-9]*")):
            if not d.name[1:].isdigit():
                continue
            s = d.name[1:]
            person = (d / "person.txt").read_text().strip() if (d / "person.txt").exists() else ""
            clips: dict[str, int] = {}
            for f in (self.dataset / s).glob("gesture_*.bin") if (self.dataset / s).is_dir() else []:
                m = re.match(r"gesture_\d+-(.+)-[a-z]+-\d+\.bin$", f.name)
                if m:
                    clips[m.group(1)] = clips.get(m.group(1), 0) + 1
            out.append({"session": s, "person": person, "raw_takes": len(list((d / "raw").glob("*.bin"))),
                        "labeled_takes": len(list((d / "labeled").glob("*.bin"))), "clips": clips,
                        "total_clips": sum(clips.values())})
        return {"sessions": out}

    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self, user: str, body: dict) -> dict:
        if self.running():
            raise HTTPException(409, "a recording is already running")
        g = self.gestures()
        ids = {c["id"] for c in g["classes"]}
        session = str(body.get("session", ""))
        if not re.fullmatch(r"\d{1,3}", session) or not 20 <= int(session) <= 999:
            raise HTTPException(422, "session: a number >= 20 (sessions 01-19 are the organisers' dataset)")
        classes = body.get("classes") or [c["id"] for c in g["classes"] if c.get("active", True) and c["id"] != 5]
        if not isinstance(classes, list) or not all(isinstance(c, int) and c in ids for c in classes):
            raise HTTPException(422, f"classes: list of ids from gestures.json {sorted(ids)}")
        minutes = body.get("minutes", 1.0)
        if not isinstance(minutes, (int, float)) or not 0.2 <= minutes <= 10:
            raise HTTPException(422, "minutes: 0.2..10 per gesture")
        styles = str(body.get("styles") or DEFAULT_STYLES)
        if not re.fullmatch(r"[a-z]+(,[a-z]+){0,5}", styles):
            raise HTTPException(422, "styles: comma list of lowercase words, e.g. normal,fast,large")
        person = re.sub(r"[^\w .-]", "", str(body.get("person", "")))[:40]
        st = await self.board.raw_status(0)
        if not st.get("ssh"):
            raise HTTPException(503, "board not reachable: the camera is on the board")
        busy = [k for k in ("camera", "bridge") if (st.get("processes") or {}).get(k)]
        question = []
        existing = self.captures / f"s{int(session):02d}" / "labeled"
        if existing.exists() and any(existing.glob("*.bin")):
            question.append(f"Session {session} already has recorded takes; new takes are added after them.")
        if busy:
            question.append("Recording needs the board's camera, which can only stream to one program: board "
                            "inference will be stopped (start it again afterwards).")
        lock = st.get("lock")
        if lock and not lock.startswith("dashboard:"):
            question.append(f"The board is locked by someone else ({lock[:80]}); the lock will be taken over.")
        if question:
            need(body.get("confirm", False), " ".join(question) + " Continue?")
        await self.board.ensure_lock(user, f"recording session {session}", confirm=True)
        if busy:
            r = await self.board.ssh.run("bash /root/inference.sh stop", timeout=60)
            r.check("stopping board inference")
            self.board.invalidate()
        conf = {"session": session, "person": person, "classes": classes, "minutes": float(minutes), "styles": styles,
                "default_styles": DEFAULT_STYLES, "no_action_styles": NO_ACTION_STYLES,
                "no_action": bool(body.get("no_action", True)), "shuffle": bool(body.get("shuffle", True)),
                "ready_s": 6.0, "switch_s": 3.0, "seed": 0, "captures": str(self.captures),
                "host": f"{self.cfg.board.user}@{self.cfg.board.host}", "port": 3333,
                "gestures_json": str(project_file(self.cfg.paths, "gestures.json"))}
        py = Path(self.cfg.paths.recorder_python).expanduser()
        if not py.exists():
            raise HTTPException(503, f"river-tools Python not found at {py} (uv tool install river-tools)")
        self.proc = await asyncio.create_subprocess_exec(
            str(py), "-u", str(WORKER), json.dumps(conf), str(self.preview), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        self.user = user
        self.state = {"running": True, "phase": "connecting", "session": f"{int(session):02d}", "saved": [], "error": None}
        asyncio.create_task(self._read(self.proc))
        self.audit.log(user, "record.start", {"session": session, "classes": classes, "minutes": minutes})
        return self.state

    async def _read(self, proc) -> None:
        assert proc.stdout
        async for raw in proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "state":
                msg.pop("type")
                self.state = msg
                self.broker.publish("record", msg)
        await proc.wait()
        err = (await proc.stderr.read()).decode(errors="replace").strip() if proc.stderr else ""
        if proc.returncode and not self.state.get("error"):
            self.state.update(running=False, phase="error", error=f"recorder exited ({proc.returncode}): {err[-300:]}")
        self.state["running"] = False
        self.broker.publish("record", self.state)
        self.audit.log(self.user, "record.end", {"session": self.state.get("session"), "takes": len(self.state.get("saved") or [])},
                       self.state.get("phase", "?"))

    async def control(self, user: str, action: str) -> dict:
        if action not in ("go", "pause", "resume", "redo", "skip", "stop"):
            raise HTTPException(422, "action: go, pause, resume, redo, skip or stop")
        if not self.running() or self.proc.stdin is None:
            raise HTTPException(409, "no recording running")
        self.proc.stdin.write((json.dumps({"cmd": action}) + "\n").encode())
        await self.proc.stdin.drain()
        if action == "stop":
            try:
                await asyncio.wait_for(self.proc.wait(), 10)
            except asyncio.TimeoutError:
                self.proc.kill()
        await asyncio.sleep(0.1)
        return self.state

    def preview_jpeg(self) -> bytes:
        try:
            img = np.load(self.preview)
            if not self.running() or time.time() - self.preview.stat().st_mtime > 3:
                img = (img * 0.25).astype(np.uint8)
        except (OSError, ValueError):
            img = np.zeros((320, 320, 3), np.uint8)
        return cv2.imencode(".jpg", img[..., ::-1])[1].tobytes()

    async def cut(self, user: str, session: str, best: int, recut: bool, log) -> dict:
        if not re.fullmatch(r"\d{2,3}", session):
            raise HTTPException(422, "session")
        if not (self.captures / f"s{session}" / "labeled").exists():
            raise HTTPException(404, f"no labeled takes for session {session}")
        if self.running():
            raise HTTPException(409, "wait until the recording has finished")
        args = [self.cfg.paths.ros_python, str(project_file(self.cfg.paths, "cut_takes.py")), "--session", session,
                "--labeled", "--captures", str(self.captures), "--dataset", str(self.dataset)]
        if best:
            args += ["--best", str(int(best))]
        if recut:
            args += ["--recut"]
        log("$ " + " ".join(args))
        r = await run_local(args, timeout=1800, cwd=str(project_file(self.cfg.paths, "cut_takes.py").parent), log=log)
        self.audit.log(user, "record.cut", {"session": session, "best": best, "recut": recut}, "ok" if r.ok else f"exit {r.rc}")
        r.check("cut_takes.py")
        return {"session": session}

    async def shutdown(self) -> None:
        if self.running():
            try:
                self.proc.stdin.write(b'{"cmd": "stop"}\n')
                await asyncio.wait_for(self.proc.wait(), 8)
            except Exception:  # noqa: BLE001
                self.proc.kill()
