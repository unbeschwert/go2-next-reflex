#!/usr/bin/env python3
"""Runs ON THE KRIA. Shows what the SNN sees and what it decides, in a browser.

Passively listens on the loopback interface to the TCP connection between the
camera client (despot_send_and_forget) and the SPLEAT server (127.0.0.1:3333):
  * upstream:   input frames  {id, payload = float32[2][80][80]}  at 100 Hz
  * downstream: model outputs {id, payload = float32[5], latency} at 100 Hz
Nothing is modified - inference keeps running exactly as before.

    python3 /root/snn_input_viewer.py [port]     ->  http://192.168.77.1:8080
"""
import socket
import struct
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
SERVER_PORT = 3333
H = W = 80
SCALE = 6
N_OUT = 5
LABELS = ["o0 Sit?", "o1 Hello?", "o2 Dance?", "o3 ?", "o4 none"]
COLORS = [(40, 115, 240), (255, 190, 75), (230, 90, 215), (115, 230, 90), (150, 150, 150)]  # BGR

lock = threading.Lock()
frames = deque(maxlen=5)             # last 50 ms of input frames
outs = deque(maxlen=50)              # last 0.5 s of model outputs
stats = {"in": 0, "out": 0, "lat": 0, "t": time.time(), "in_hz": 0.0, "resync": 0}
latest_jpeg = None


def parse_payload(body):
    k = body.find(b"\x67payload")
    if k < 0:
        return None, body
    h = body[k + 8]
    if h == 0x59:
        n, off = struct.unpack(">H", body[k + 9:k + 11])[0], k + 11
    elif h == 0x58:
        n, off = body[k + 9], k + 10
    elif 0x40 <= h < 0x58:
        n, off = h - 0x40, k + 9
    else:
        return None, body
    return body[off:off + n], body


class Stream:
    """Reassembles one direction of the TCP stream into length-prefixed frames."""

    def __init__(self, on_frame, max_frame):
        self.buf = bytearray()
        self.on_frame, self.max_frame = on_frame, max_frame

    def feed(self, data):
        self.buf += data
        while len(self.buf) >= 4:
            n = struct.unpack(">I", self.buf[:4])[0]
            if n == 0 or n > self.max_frame or (len(self.buf) > 4 and self.buf[4] & 0xE0 != 0xA0):
                # lost sync (capture dropped a packet): skip to the next CBOR map header
                j = self.buf.find(b"\x62id", 5)
                stats["resync"] += 1
                if j < 5:
                    self.buf.clear()
                    return
                del self.buf[:j - 5]
                continue
            if len(self.buf) < 4 + n:
                return
            body = bytes(self.buf[4:4 + n])
            del self.buf[:4 + n]
            self.on_frame(body)


def on_input(body):
    p, _ = parse_payload(body)
    if p is None or len(p) != 2 * H * W * 4:
        return
    a = np.frombuffer(p, "<f4").reshape(2, H, W)
    with lock:
        frames.append(a)
        stats["in"] += 1


def on_output(body):
    p, _ = parse_payload(body)
    if p is None or len(p) != 4 * N_OUT:
        return
    sc = struct.unpack(f"<{N_OUT}f", p)
    j = body.find(b"\x67latency")
    with lock:
        outs.append([1.0 if x > 0 else 0.0 for x in sc])
        stats["out"] += 1
        if j >= 0 and body[j + 8] == 0x19:
            stats["lat"] = struct.unpack(">H", body[j + 9:j + 11])[0]


def sniff():
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
    s.bind(("lo", 0))
    up = {}      # client port -> Stream (input frames)
    down = {}    # client port -> Stream (model outputs)
    while True:
        pkt, addr = s.recvfrom(70000)
        if addr[2] == socket.PACKET_OUTGOING:      # loopback shows every packet twice
            continue
        ip = pkt[14:]
        if len(ip) < 20 or ip[9] != 6:
            continue
        ihl = (ip[0] & 15) * 4
        tcp = ip[ihl:]
        sp, dp = struct.unpack(">HH", tcp[:4])
        data = tcp[(tcp[12] >> 4) * 4:]
        if not data:
            continue
        if dp == SERVER_PORT:
            up.setdefault(sp, Stream(on_input, 200_000)).feed(data)
        elif sp == SERVER_PORT:
            down.setdefault(dp, Stream(on_output, 4096)).feed(data)


def render():
    global latest_jpeg
    while True:
        time.sleep(1 / 20)
        with lock:
            fr = list(frames)
            o = list(outs)
            now = time.time()
            if now - stats["t"] >= 1:
                stats["in_hz"] = stats["in"] / (now - stats["t"])
                stats["in"], stats["out"], stats["t"] = 0, 0, now
        img = np.full((H, W, 3), 18, np.uint8)
        if fr:
            acc = np.sum(fr, axis=0)                 # [2, 80, 80] over last 50 ms
            img[acc[1] > 0] = (255, 120, 40)         # channel 1 (OFF?) blue
            img[acc[0] > 0] = (255, 255, 255)        # channel 0 (ON?) white
        big = cv2.resize(img, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
        rates = [sum(f[i] for f in o) / len(o) for i in range(N_OUT)] if o else [0.0] * N_OUT
        panel = np.full((170, W * SCALE, 3), 28, np.uint8)
        top = int(np.argmax(rates))
        for i, r in enumerate(rates):
            y = 14 + i * 31
            cv2.putText(panel, LABELS[i], (8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (120, y), (120 + int(330 * r), y + 20), COLORS[i], -1)
            cv2.rectangle(panel, (120, y), (450, y + 20), (90, 90, 90), 1)
            cv2.putText(panel, f"{r:.2f}", (455, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255) if i == top and r > 0.3 else (160, 160, 160), 1, cv2.LINE_AA)
        out = np.vstack([big, panel])
        cv2.putText(out, f"SNN input 2x80x80 @ {stats['in_hz']:.0f} Hz   FPGA latency {stats['lat']/1000:.1f} ms",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            latest_jpeg = jpg.tobytes()


PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SNN Live View</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;display:flex;
flex-direction:column;align-items:center;padding:16px}img{max-width:100%;border:1px solid #333}
p{color:#999;font-size:14px;max-width:480px}</style></head><body>
<h3>What the SNN sees (top) and decides (bars) - live from the Kria</h3>
<img id="v" src="/stream" alt="live SNN view">
<p>Top: the exact 80x80 event frames sent to the FPGA (white = brighter, blue = darker, last 50 ms).
Bars: how often each FPGA output fired in the last 0.5 s. A gesture triggers a trick when its bar
stays above 0.35 and clearly beats the others.</p>
<script>const v=document.getElementById('v');v.onerror=()=>setTimeout(()=>{v.src='/stream?'+Date.now()},1000);</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.split("?")[0] == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    j = latest_jpeg
                    if j is not None and j is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        last = j
                    time.sleep(0.025)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


if __name__ == "__main__":
    threading.Thread(target=sniff, daemon=True).start()
    threading.Thread(target=render, daemon=True).start()
    print(f"SNN live view on http://0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
