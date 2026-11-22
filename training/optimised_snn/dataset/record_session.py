#!/usr/bin/env python3
"""Cued gesture recording: one continuous session, labels come from the on-screen cue.

Each gesture is recorded for --minutes (default 1) without stopping; the speed/size cue changes
every 20 s (normal -> fast -> large; no-action: normal, fast, still, empty). The screen shows
next: HEART (fast) 3..2..1   then   GO HEART 0:59   and the events that arrive while GO is shown go to
    ~/river-captures/s<session>/labeled/gesture_<NN>-<label>-<style>-take<KK>.bin
Then cut them into 1 s clips with:
    python3 cut_takes.py --session <session> --labeled

Run with the river-tools Python (it has the stream code):
    ~/.local/share/uv/tools/river-tools/bin/python record_session.py --session 20
(stop board inference first:  ssh root@192.168.77.1 'cd /root && bash inference.sh stop')

Keys:  space = pause/resume (what was recorded is kept, the segment continues after the pause)
       r = redo the previous take (deletes it)    n = skip this step    q = quit
Face the laptop screen to the person but keep it OUT of the camera's view.
"""
import argparse
import json
import random
import textwrap
import socket
import struct
import threading
import time
from pathlib import Path

GESTURES = json.loads((Path(__file__).resolve().parent / "gestures.json").read_text())
CLASSES = {g["id"]: g["name"] for g in GESTURES["classes"]}
HOW = {g["id"]: g["how"] for g in GESTURES["classes"]}
CLASS_STYLES = {g["id"]: g["styles"] for g in GESTURES["classes"] if g.get("styles")}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session", required=True)
ap.add_argument("--captures", default=str(Path.home() / "river-captures"))
ap.add_argument("--host", default="root@192.168.77.1")
ap.add_argument("--port", type=int, default=3333)
ap.add_argument("--minutes", type=float, default=1.0, help="recording time per gesture, split over the styles (1 min = 20 s normal/fast/large)")
ap.add_argument("--ready", type=float, default=6.0, help="countdown before a new gesture (not recorded)")
ap.add_argument("--switch", type=float, default=3.0, help="countdown when only the style changes")
ap.add_argument("--styles", default="normal,fast,large")
ap.add_argument("--no-action-styles", default="normal:2,fast:2,still:1,empty:1",
                help="no-action segments as style:weight (time is split by weight)")
ap.add_argument("--classes", default="all",
                help="preset from gestures.json (all) or a comma list of ids, e.g. 6,8,9")
ap.add_argument("--no-action", type=int, default=1, help="record no-action as well")
ap.add_argument("--shuffle", type=int, default=1, help="shuffle the gesture order")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--auto-close", action="store_true", help="close the window when the plan is done")
ap.add_argument("--person", default="",
                help="who records: the person's earlier session number (e.g. 21) or a new name. Sessions of "
                     "the same person must stay in the same train/val/test split")
a = ap.parse_args()

import numpy as np
import matplotlib.pyplot as plt

from river_tools.capture.events import COLORS, EVENT_DTYPE, EVENT_SIZE
from river_tools.capture.remote import start_camera, stop_camera
from river_tools.capture.stream import receive_batches

session = f"{int(a.session):02d}"
out_dir = Path(a.captures) / f"s{session}" / "labeled"
out_dir.mkdir(parents=True, exist_ok=True)
person_file = out_dir.parent / "person.txt"
if a.person:
    person = a.person.strip()
    if not person:
        ap.error("--person must have a non-empty value")
    person = f"{int(person):02d}" if person.isdecimal() else person
    if person_file.exists():
        recorded = person_file.read_text().strip()
        recorded = f"{int(recorded):02d}" if recorded.isdecimal() else recorded
        if person != recorded:
            ap.error(f"session {session} already belongs to person {recorded!r}")
    else:
        person_file.write_text(person + "\n")
elif not person_file.exists():
    print(f"note: no --person given; session {session} counts as a new person in the split")

rng = random.Random(a.seed or int(session))
plan = []
chosen = GESTURES["presets"].get(a.classes) or [int(c) for c in a.classes.split(",")]
unknown = [c for c in chosen if c not in CLASSES]
if unknown:
    raise SystemExit(f"unknown class ids {unknown} - add them to gestures.json")
order = [c for c in chosen if c != 5]
if a.shuffle:
    rng.shuffle(order)
if a.no_action:
    order.insert(len(order) // 2, 5)          # no-action in the middle, as a rest between gestures
total_s = a.minutes * 60
for c in order:
    if c == 5:
        segs = [(st, float(w)) for st, w in (x.split(":") for x in a.no_action_styles.split(",") if x)]
    else:
        own = CLASS_STYLES.get(c)                   # per-class plan from gestures.json (e.g. front/back)
        use = own if (own and a.styles == ap.get_default("styles")) else a.styles.split(",")
        segs = [(st, 1.0) for st in use if st]
    wsum = sum(w for _, w in segs)
    for i, (st, w) in enumerate(segs):
        turning = i > 0 and st.startswith("back") and not segs[i - 1][0].startswith("back")
        plan.append((c, st, total_s * w / wsum, a.ready if (i == 0 or turning) else a.switch))
HINT_STYLE = {"still": "stand still, small fidgets", "empty": "step OUT of the view",
              "front": "FACE THE CAMERA and walk on the spot, normal pace",
              "frontfast": "FACE THE CAMERA and walk on the spot FAST",
              "back": "TURN AROUND: back to the camera, walk on the spot, normal pace",
              "backfast": "back to the camera, walk on the spot FAST"}


class TakeWriter:
    """Routes incoming event batches into the currently open take file."""

    def __init__(self):
        self.lock = threading.Lock()
        self.file = None
        self.path = None
        self.events = 0

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
    prefix = f"gesture_{cls:02d}-{CLASSES[cls]}-{style}-take"
    raw = out_dir.parent / "raw"
    used = [int(p.stem.rsplit("take", 1)[1]) for d in (out_dir, raw) if d.exists()
            for p in d.glob(prefix + "*.bin") if p.stem.rsplit("take", 1)[1].isdigit()]
    return out_dir / f"{prefix}{max(used, default=0) + 1:02d}.bin"


writer = TakeWriter()
stop, connected = threading.Event(), threading.Event()
image_lock = threading.Lock()
image = np.zeros((320, 320, 3), dtype=np.uint8)
rate = [0]


def receive():
    camera = None
    try:
        with socket.create_server(("", a.port)) as server:
            server.settimeout(0.2)
            camera = start_camera(a.host, a.port)
            while not stop.is_set():
                if camera.poll() is not None:
                    print("camera SSH session ended")
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
                            with image_lock:
                                image[ev["y"], ev["x"]] = COLORS[ev["polarity"]]
                                rate[0] += len(ev)
                    except ConnectionError:
                        pass
    finally:
        if camera is not None:
            stop_camera(camera)
        stop.set()


# ------------------------------------------------------------------ UI
fig = plt.figure("Gesture recording - session " + session, figsize=(13, 7.5), facecolor="#202020")
ax_img = fig.add_axes([0.62, 0.12, 0.36, 0.62])
ax_img.set_xticks([]); ax_img.set_yticks([])
shown = ax_img.imshow(image.copy(), interpolation="nearest")
t_title = fig.text(0.31, 0.80, "", ha="center", fontsize=50, weight="bold", color="w")
t_style = fig.text(0.31, 0.66, "", ha="center", fontsize=34, color="w")
t_count = fig.text(0.31, 0.33, "", ha="center", fontsize=100, weight="bold", color="w")
t_hint = fig.text(0.31, 0.25, "", ha="center", va="top", fontsize=19, color="#dddddd", linespacing=1.4)
t_info = fig.text(0.02, 0.03, "", fontsize=12, color="#bbbbbb", family="monospace")
t_rate = fig.text(0.62, 0.77, "", fontsize=12, color="#bbbbbb")

# start paused so the person can get into position; space starts the first countdown
state = {"step": 0, "phase": "ready", "t0": None, "paused": not a.auto_close, "saved": [], "quit": False,
         "go_len": None, "resume_left": None}


def finish_go(now, keep_min=5.0):
    """Close the running take; keep it if it is long enough. Returns seconds recorded."""
    elapsed = now - state["t0"]
    path = writer.close(keep=elapsed >= keep_min)
    if path is not None:
        n = (path.stat().st_size - 4) // EVENT_SIZE
        print(f"saved {path.name}: {elapsed:.0f} s, {n} events")
        state["saved"].append((path, state["step"]))
    return elapsed


def on_key(e):
    k = e.key
    if k == " ":
        if not state["paused"]:
            if state["phase"] == "go":       # keep what was recorded, continue the segment later
                left = state["go_len"] - finish_go(time.monotonic())
                state["resume_left"] = left if left > 3 else None
                if state["resume_left"] is None:
                    state["step"] += 1
            state.update(paused=True, phase="ready")
        else:
            state.update(paused=False, phase="ready", t0=time.monotonic())
    elif k == "r" and state["saved"]:
        writer.close(keep=False)
        path, step = state["saved"].pop()
        path.unlink(missing_ok=True)
        print(f"deleted {path.name} - redoing it")
        state.update(step=step, phase="ready", t0=time.monotonic(), resume_left=None)
    elif k == "n":
        writer.close(keep=False)
        state.update(step=state["step"] + 1, phase="ready", t0=time.monotonic(), resume_left=None)
    elif k == "q":
        state["quit"] = True


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("close_event", lambda _: state.update(quit=True))
worker = threading.Thread(target=receive, daemon=True)
worker.start()
plt.show(block=False)

print(f"plan ({len(plan)} segments, ~{sum(p[2] + p[3] for p in plan) / 60:.0f} min):",
      ", ".join(f"{CLASSES[c]}/{s} {sec:.0f}s" for c, s, sec, _ in plan))
last_rate_t, shown_rate = time.monotonic(), 0
try:
    while not state["quit"] and not stop.is_set() and plt.fignum_exists(fig.number):
        now = time.monotonic()
        with image_lock:
            shown.set_data(image.copy()); image[:] = 0
            if now - last_rate_t >= 0.5:
                shown_rate, rate[0], last_rate_t = rate[0] / (now - last_rate_t), 0, now
        t_rate.set_text(f"live view   {shown_rate / 1e3:6.0f} k events/s")

        if state["step"] >= len(plan):
            if a.auto_close:
                break
            fig.set_facecolor("#203a5a")
            t_title.set_text("DONE"); t_style.set_text(f"{len(state['saved'])} takes saved")
            t_count.set_text(""); t_hint.set_text("close the window")
        elif not connected.is_set():
            t_title.set_text("connecting..."); t_style.set_text(""); t_count.set_text(""); t_hint.set_text("")
            state["t0"] = now
        elif state["paused"]:
            fig.set_facecolor("#404040")
            first = state["step"] == 0 and not state["saved"]
            t_title.set_text("READY?" if first else "PAUSED")
            c0, s0 = plan[min(state["step"], len(plan) - 1)][:2]
            t_style.set_text(f"{'first' if first else 'next'}: {CLASSES[c0].upper()} ({s0})")
            t_count.set_text(""); t_hint.set_text("space = start" if first else "space = continue")
        else:
            cls, style, secs, ready = plan[state["step"]]
            name = CLASSES[cls].upper()
            if state["t0"] is None:
                state["t0"] = now
            if state["resume_left"] is not None:
                ready = a.ready
            left = (ready if state["phase"] == "ready" else state["go_len"]) - (now - state["t0"])
            if state["phase"] == "ready" and left <= 0:
                writer.open(take_path(cls, style))
                go_len = state["resume_left"] if state["resume_left"] is not None else secs
                state.update(phase="go", t0=now, go_len=go_len, resume_left=None)
                continue
            if state["phase"] == "go" and left <= 0:
                finish_go(now, keep_min=0)
                state.update(step=state["step"] + 1, phase="ready", t0=now)
                continue
            if state["phase"] == "ready":
                fig.set_facecolor("#5a4a10")
                t_title.set_text(f"next: {name}")
                t_count.set_text(f"{int(np.ceil(left))}")
            else:
                fig.set_facecolor("#14502a")
                t_title.set_text(f"GO  {name}")
                sec_left = int(np.ceil(left))
                t_count.set_text(f"{sec_left // 60}:{sec_left % 60:02d}")
            t_style.set_text(style)
            t_hint.set_text(textwrap.fill(HINT_STYLE.get(style) or HOW.get(cls, "repeat it non-stop"), 46))
        t_info.set_text(f"session {session}   segment {min(state['step'] + 1, len(plan))}/{len(plan)}   "
                        f"saved {len(state['saved'])}   [space] pause  [r] redo last  [n] skip  [q] quit")
        fig.canvas.draw_idle()
        plt.pause(0.03)
finally:
    if state["phase"] == "go" and writer.file is not None:
        finish_go(time.monotonic())            # keep a partly recorded segment (>= 5 s) on quit
    writer.close(keep=False)
    stop.set()
    worker.join(timeout=8)
    plt.close("all")
    print(f"{len(state['saved'])} takes in {out_dir}")
    print(f"next:  python3 cut_takes.py --session {session} --labeled")
