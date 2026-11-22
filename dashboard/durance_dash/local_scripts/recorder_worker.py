"""Headless cued gesture recording for the dashboard (same plan and file names as go2_viz/record_session.py).

    <river-tools python> recorder_worker.py CONFIG_JSON PREVIEW_NPY

Takes go to ~/river-captures/s<session>/labeled/gesture_<NN>-<label>-<style>-take<KK>.bin, the format
cut_takes.py --labeled expects. stdin: {"cmd": "go" | "pause" | "resume" | "redo" | "skip" | "stop"}.
stdout: JSON state 4 times per second and after every change. PREVIEW_NPY: 320x320x3 uint8 live events.
"""
import json
import os
import random
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from river_tools.capture.events import COLORS, EVENT_DTYPE, EVENT_SIZE
from river_tools.capture.remote import start_camera, stop_camera
from river_tools.capture.stream import receive_batches

cfg = json.loads(sys.argv[1])
preview_path = Path(sys.argv[2])
G = json.loads(Path(cfg["gestures_json"]).read_text())
CLASSES = {g["id"]: g["name"] for g in G["classes"]}
HOW = {g["id"]: g["how"] for g in G["classes"]}
CLASS_STYLES = {g["id"]: g["styles"] for g in G["classes"] if g.get("styles")}
HINT_STYLE = {"still": "stand still, small fidgets", "empty": "step OUT of the view",
              "front": "FACE THE CAMERA and walk on the spot, normal pace",
              "frontfast": "FACE THE CAMERA and walk on the spot FAST",
              "back": "TURN AROUND: back to the camera, walk on the spot, normal pace",
              "backfast": "back to the camera, walk on the spot FAST"}

session = "%02d" % int(cfg["session"])
out_dir = Path(cfg["captures"]) / ("s" + session) / "labeled"
out_dir.mkdir(parents=True, exist_ok=True)
if cfg.get("person"):
    (out_dir.parent / "person.txt").write_text(cfg["person"].strip() + "\n")

# ---- plan (identical to record_session.py)
rng = random.Random(cfg.get("seed") or int(session))
order = [c for c in cfg["classes"] if c != 5]
if cfg.get("shuffle", True):
    rng.shuffle(order)
if cfg.get("no_action", True):
    order.insert(len(order) // 2, 5)
total_s = cfg["minutes"] * 60
plan = []
for c in order:
    if c == 5:
        segs = [(st, float(w)) for st, w in (x.split(":") for x in cfg["no_action_styles"].split(",") if x)]
    else:
        own = CLASS_STYLES.get(c)
        use = own if (own and cfg["styles"] == cfg["default_styles"]) else cfg["styles"].split(",")
        segs = [(st, 1.0) for st in use if st]
    wsum = sum(w for _, w in segs)
    for i, (st, w) in enumerate(segs):
        turning = i > 0 and st.startswith("back") and not segs[i - 1][0].startswith("back")
        plan.append((c, st, total_s * w / wsum, cfg["ready_s"] if (i == 0 or turning) else cfg["switch_s"]))


class TakeWriter:
    def __init__(self):
        self.lock = threading.Lock()
        self.file, self.path, self.events = None, None, 0

    def open(self, path):
        with self.lock:
            self.path, self.events = path, 0
            self.file = path.open("xb")
            self.file.write(struct.pack("!I", 0))

    def write(self, batch):
        with self.lock:
            if self.file is not None:
                self.file.write(batch)
                self.events += len(batch) // EVENT_SIZE

    def close(self, keep=True):
        with self.lock:
            if self.file is None:
                return None
            size = self.file.tell() - 4
            self.file.seek(0)
            self.file.write(struct.pack("!I", size))
            self.file.close()
            self.file, path = None, self.path
        if not keep:
            path.unlink()
            return None
        return path


def take_path(cls, style):
    prefix = "gesture_%02d-%s-%s-take" % (cls, CLASSES[cls], style)
    raw = out_dir.parent / "raw"
    used = [int(p.stem.rsplit("take", 1)[1]) for d in (out_dir, raw) if d.exists()
            for p in d.glob(prefix + "*.bin") if p.stem.rsplit("take", 1)[1].isdigit()]
    return out_dir / ("%s%02d.bin" % (prefix, max(used, default=0) + 1))


writer = TakeWriter()
stop, connected = threading.Event(), threading.Event()
img_lock = threading.Lock()
image = np.zeros((320, 320, 3), dtype=np.uint8)
rate = [0]
error = [None]
out_lock = threading.Lock()


def emit(obj):
    with out_lock:
        try:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()
        except OSError:
            stop.set()


def receive():
    camera = None
    try:
        with socket.create_server(("", cfg["port"])) as server:
            server.settimeout(0.2)
            camera = start_camera(cfg["host"], cfg["port"])
            while not stop.is_set():
                if camera.poll() is not None:
                    error[0] = "camera SSH session ended (is board inference stopped? camera cable?)"
                    break
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                connected.set()
                with conn:
                    try:
                        for batch in receive_batches(conn, float("inf"), stop):
                            writer.write(batch)
                            ev = np.frombuffer(batch, dtype=EVENT_DTYPE)
                            with img_lock:
                                image[ev["y"] % 320, ev["x"] % 320] = COLORS[ev["polarity"] % 2]
                                rate[0] += len(ev)
                    except ConnectionError:
                        pass
    except OSError as e:
        error[0] = "recording socket: %s" % e
    finally:
        if camera is not None:
            stop_camera(camera)
        stop.set()


commands = []
cmd_lock = threading.Lock()


def read_stdin():
    for raw in sys.stdin:
        try:
            with cmd_lock:
                commands.append(json.loads(raw).get("cmd"))
        except ValueError:
            pass
    with cmd_lock:
        commands.append("stop")                 # dashboard gone: keep what was recorded, stop


state = {"step": 0, "phase": "waiting", "t0": None, "saved": [], "go_len": None, "resume_left": None, "paused": True}


def finish_go(now, keep_min=5.0):
    elapsed = now - state["t0"]
    path = writer.close(keep=elapsed >= keep_min)
    if path is not None:
        n = (path.stat().st_size - 4) // EVENT_SIZE
        state["saved"].append({"file": path.name, "events": int(n), "seconds": round(elapsed, 1), "step": state["step"]})
    return elapsed


def handle(cmd, now):
    if cmd in ("go", "resume") and state["paused"]:
        state.update(paused=False, phase="ready", t0=now)
    elif cmd == "pause" and not state["paused"]:
        if state["phase"] == "go":
            left = state["go_len"] - finish_go(now)
            state["resume_left"] = left if left > 3 else None
            if state["resume_left"] is None:
                state["step"] += 1
        state.update(paused=True, phase="paused")
    elif cmd == "redo" and state["saved"]:
        writer.close(keep=False)
        last = state["saved"].pop()
        (out_dir / last["file"]).unlink(missing_ok=True)
        state.update(step=last["step"], phase="ready" if not state["paused"] else "paused", t0=now, resume_left=None)
    elif cmd == "skip":
        writer.close(keep=False)
        state.update(step=state["step"] + 1, phase="ready" if not state["paused"] else "paused", t0=now, resume_left=None)
    elif cmd == "stop":
        return False
    return True


def view(now, ev_rate):
    step = min(state["step"], len(plan) - 1)
    cls, style, secs, ready = plan[step] if plan else (5, "normal", 0, 0)
    left = None
    if state["phase"] in ("ready", "go") and state["t0"] is not None:
        length = (cfg["ready_s"] if state["resume_left"] is not None else ready) if state["phase"] == "ready" else state["go_len"]
        left = max(0.0, length - (now - state["t0"]))
    phase = state["phase"]
    if not connected.is_set() and phase not in ("done", "error"):
        phase = "connecting"
    return {"type": "state", "running": phase not in ("done", "error"), "connected": connected.is_set(), "phase": phase,
            "step": min(state["step"] + 1, len(plan)), "steps": len(plan), "session": session,
            "gesture": {"id": cls, "name": CLASSES.get(cls, "?"), "how": HOW.get(cls, "")}, "style": style,
            "hint": HINT_STYLE.get(style) or HOW.get(cls, "repeat it non-stop"),
            "seconds_left": None if left is None else round(left, 1), "events_per_s": int(ev_rate),
            "saved": state["saved"], "error": error[0],
            "plan": [{"cls": c, "name": CLASSES.get(c, "?"), "style": s, "seconds": round(t, 1)} for c, s, t, _ in plan]}


threading.Thread(target=receive, daemon=True).start()
threading.Thread(target=read_stdin, daemon=True).start()
last_emit, last_preview, last_rate_t, ev_rate = 0.0, 0.0, time.monotonic(), 0.0
try:
    while True:
        now = time.monotonic()
        with cmd_lock:
            pending, commands[:] = list(commands), []
        if not all(handle(c, now) for c in pending):
            break
        if stop.is_set() and state["phase"] not in ("done",):
            state["phase"] = "error" if error[0] else "done"
        if not connected.is_set():
            state["t0"] = now if state["t0"] is None or state["phase"] == "waiting" else state["t0"]
        elif state["step"] >= len(plan):
            state["phase"] = "done"
        elif not state["paused"]:
            cls, style, secs, ready = plan[state["step"]]
            if state["resume_left"] is not None:
                ready = cfg["ready_s"]
            length = ready if state["phase"] == "ready" else state["go_len"]
            left = length - (now - state["t0"])
            if state["phase"] == "ready" and left <= 0:
                writer.open(take_path(cls, style))
                go_len = state["resume_left"] if state["resume_left"] is not None else secs
                state.update(phase="go", t0=now, go_len=go_len, resume_left=None)
            elif state["phase"] == "go" and left <= 0:
                finish_go(now, keep_min=0)
                state.update(step=state["step"] + 1, phase="ready", t0=now)
        if now - last_rate_t >= 0.5:
            with img_lock:
                ev_rate, rate[0] = rate[0] / (now - last_rate_t), 0
            last_rate_t = now
        if now - last_preview >= 0.2:
            with img_lock:
                snap = image.copy()
                image[:] = 0
            tmp = preview_path.with_suffix(".tmp.npy")
            np.save(tmp, snap)
            os.replace(tmp, preview_path)
            last_preview = now
        if now - last_emit >= 0.25:
            emit(view(now, ev_rate))
            last_emit = now
        if state["phase"] in ("done", "error") and stop.is_set():
            break
        time.sleep(0.03)
finally:
    if state["phase"] == "go" and writer.file is not None:
        finish_go(time.monotonic())
    writer.close(keep=False)
    stop.set()
    final = view(time.monotonic(), 0)
    final.update(running=False, phase="error" if error[0] else "done")
    emit(final)
