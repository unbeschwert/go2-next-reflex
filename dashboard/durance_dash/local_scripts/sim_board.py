"""Fake Kria board for the dashboard's simulator.

    python3 sim_board.py CACHE.npz OUT_PORT CAMERA_PORT

* OUT_PORT (127.0.0.1): the spleat-server output stream the gesture bridge connects to: 4-byte big-endian
  length + CBOR {"payload": 5 x float32} per frame, 100 frames/s (quiet scene: ~15/s, like the real camera
  client, which only sends frames that contain events).
* CAMERA_PORT (127.0.0.1): GET /stream (MJPEG) and /snapshot (JPEG) of the event frames that produced
  those outputs (recorded clips of the person in CACHE, decoded at the model's input size).
* stdin: JSON lines {"person": "none" | "heart" | "left" | "right" | "empty"}; stdout: JSON state lines.
The outputs are the FPGA integer model's (go2_viz/spleat_sim.py) per-frame spikes for those clips.
"""
import json
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

cache_path, out_port, cam_port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = np.load(cache_path)
PEOPLE = ["none", "heart", "left", "right", "empty"]
pools = {p: (d[f"out_{p}"], d[f"frames_{p}"]) for p in PEOPLE if f"out_{p}" in d}
size = int(d["size"])
lock = threading.Lock()
state = {"person": "none", "clip": 0, "frame": 0, "clients": 0, "sent": 0}
latest = {"jpeg": None, "counts": np.zeros((2, size, size), np.float32)}


def emit(obj):
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
    except OSError:
        pass


def next_frame():
    """(outputs[5], counts[2,S,S], quiet) for the current person, cycling through its clips."""
    with lock:
        p = state["person"]
        if p == "empty" or p not in pools:
            return np.zeros(5, np.float32), np.zeros((2, size, size), np.float32), True
        outs, frames = pools[p]
        c, f = state["clip"] % len(outs), state["frame"]
        o, fr = outs[c][f].astype(np.float32), frames[c][f].astype(np.float32)
        state["frame"] += 1
        if state["frame"] >= outs.shape[1]:
            state["frame"], state["clip"] = 0, state["clip"] + 1
        return o, fr, False


def message(o):
    body = b"\xa1\x67payload\x54" + struct.pack("<5f", *[float(x) for x in o])
    return struct.pack(">I", len(body)) + body


def serve_outputs():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", out_port))
    srv.listen(4)
    clients = []
    srv.settimeout(0.0)
    t0, k, quiet_skip = time.time(), 0, 0
    while True:
        try:
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            clients.append(conn)
            state["clients"] = len(clients)
        except BlockingIOError:
            pass
        o, fr, quiet = next_frame()
        with lock:
            latest["counts"] = 0.6 * latest["counts"] + fr
        send = not quiet or quiet_skip % 7 == 0          # quiet scene: ~15 frames/s
        quiet_skip += 1
        if send:
            msg = message(o)
            for c in list(clients):
                try:
                    c.sendall(msg)
                except OSError:
                    clients.remove(c)
                    state["clients"] = len(clients)
            state["sent"] += 1
        k += 1
        delay = t0 + k / 100 - time.time()
        if delay > 0:
            time.sleep(delay)
        elif delay < -1.0:                                 # fell behind (laptop busy): resync
            t0, k = time.time(), 0


def render():
    while True:
        with lock:
            c = latest["counts"].copy()
            person = state["person"]
        img = np.zeros((size, size, 3), np.float32)
        img[..., 0] = np.clip(c[0] * 60, 0, 255)           # OFF -> blue (BGR)
        img[..., 2] = np.clip(c[1] * 60, 0, 255)           # ON -> red
        img[..., 1] = np.clip(np.minimum(c[0], c[1]) * 60, 0, 255)
        big = cv2.resize(img.astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"SIMULATOR  person: {person}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        ok, jpg = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            latest["jpeg"] = jpg.tobytes()
        time.sleep(0.04)


class Cam(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/snapshot"):
            j = latest["jpeg"] or b""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(j)))
            self.end_headers()
            self.wfile.write(j)
            return
        if not self.path.startswith("/stream"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                j = latest["jpeg"]
                if j:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                time.sleep(0.05)
        except OSError:
            return


threading.Thread(target=serve_outputs, daemon=True).start()
threading.Thread(target=render, daemon=True).start()
httpd = ThreadingHTTPServer(("127.0.0.1", cam_port), Cam)
httpd.daemon_threads = True
threading.Thread(target=httpd.serve_forever, daemon=True).start()
emit({"type": "ready", "people": [p for p in PEOPLE if p in pools or p == "empty"], "size": size})
for raw in sys.stdin:
    try:
        msg = json.loads(raw)
        p = msg.get("person")
        if p in PEOPLE:
            with lock:
                state.update(person=p, clip=0, frame=0)
        emit({"type": "state", "person": state["person"], "clients": state["clients"], "sent": state["sent"]})
    except ValueError:
        emit({"type": "error", "text": "bad message"})
