#!/usr/bin/env python3
"""Live browser viewer for the GenX320 event camera on the Kria.

Reads EVT 2.1 events from /dev/video0 (via v4l2-ctl), accumulates them into
frames and serves an MJPEG stream.  Open http://<kria-ip>:8080 in a browser.

Robustness:
  * the sensor's Event Rate Controller (ERC) is enabled so bursts cannot
    overflow the MIPI receiver ("Stream Line Buffer Full"),
  * a dedicated thread only drains the camera pipe, so slow decoding never
    back-pressures the DMA; if decoding falls behind, old data is dropped,
  * a watchdog restarts the capture (re-initialising the pipeline with
    despot_test_camera) when no data arrives for STALL_S seconds.
"""
import collections, subprocess, threading, time, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2

W = H = 320
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
FPS = 30
SCALE = 2
ERC_RATE = 5_000_000          # max events/s let through by the sensor
STALL_S = 2.0                 # no bytes for this long -> restart capture
MAX_BACKLOG = 4 * 1024 * 1024 # bytes of undecoded data kept; older is dropped
SENSOR = "/dev/v4l-subdev3"

BIT = np.arange(32, dtype=np.uint64)
OFFS = np.arange(32, dtype=np.int32)
chunks = collections.deque()
chunks_lock = threading.Lock()
backlog = [0]
last_data = [time.time()]
latest_jpeg = None
status = {"ev_per_s": 0, "restarts": 0, "dropped": 0}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup_sensor():
    subprocess.run(["v4l2-ctl", "-d", SENSOR, "-c",
                    f"erc_enable=1,erc_rate={ERC_RATE}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_capture():
    return subprocess.Popen(
        ["v4l2-ctl", "-d", "/dev/video0", "--stream-mmap=32",
         "--stream-count=0", "--stream-to=-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


def capture_loop():
    """Own the v4l2-ctl process; drain its pipe; restart it on stalls."""
    first = True
    while True:
        if not first:
            status["restarts"] += 1
            log("capture stalled - reinitialising camera pipeline")
            subprocess.run(["despot_test_camera"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        first = False
        setup_sensor()
        p = start_capture()
        last_data[0] = time.time()
        stop = threading.Event()

        def watchdog():
            while not stop.is_set():
                time.sleep(0.5)
                if time.time() - last_data[0] > STALL_S:
                    p.kill()
                    return
        threading.Thread(target=watchdog, daemon=True).start()

        while True:
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            last_data[0] = time.time()
            with chunks_lock:
                chunks.append(chunk)
                backlog[0] += len(chunk)
                while backlog[0] > MAX_BACKLOG:
                    old = chunks.popleft()
                    backlog[0] -= len(old)
                    status["dropped"] += len(old)
        stop.set()
        p.kill()
        p.wait()
        time.sleep(0.5)


def decode(data, on, off):
    w = np.frombuffer(data, dtype="<u8")
    t = w >> np.uint64(60)
    n = 0
    for kind, img in ((1, on), (0, off)):
        e = w[t == kind]
        if e.size == 0:
            continue
        x = ((e >> np.uint64(43)) & np.uint64(0x7FF)).astype(np.int32)
        y = ((e >> np.uint64(32)) & np.uint64(0x7FF)).astype(np.int32)
        bits = ((e[:, None] >> BIT) & np.uint64(1)).astype(bool)  # low 32 bits = valid mask
        xs = (x[:, None] + OFFS)[bits]
        ys = np.broadcast_to(y[:, None], bits.shape)[bits]
        ok = (xs < W) & (ys < H)
        img[ys[ok], xs[ok]] = True
        n += int(ok.sum())
    return n


def renderer():
    global latest_jpeg
    leftover = b""
    count = 0
    last_stat = time.time()
    while True:
        t0 = time.time()
        with chunks_lock:
            parts = list(chunks)
            chunks.clear()
            backlog[0] = 0
        data = leftover + b"".join(parts)
        n8 = len(data) // 8 * 8
        leftover = data[n8:]
        on = np.zeros((H, W), bool)
        off = np.zeros((H, W), bool)
        if n8:
            count += decode(data[:n8], on, off)

        img = np.full((H, W, 3), 30, np.uint8)
        img[off] = (255, 120, 40)    # OFF events: blue (BGR)
        img[on] = (255, 255, 255)    # ON events: white
        img = cv2.resize(img, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)

        now = time.time()
        if now - last_stat >= 1.0:
            status["ev_per_s"] = int(count / (now - last_stat))
            count, last_stat = 0, now
        stalled = now - last_data[0] > 1.0
        txt = "NO DATA - restarting camera..." if stalled else f"{status['ev_per_s']:,} ev/s"
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if stalled else (0, 200, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            latest_jpeg = jpg.tobytes()
        time.sleep(max(0.0, 1.0 / FPS - (time.time() - t0)))


PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kria Event Camera</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;
display:flex;flex-direction:column;align-items:center;padding:16px}
img{max-width:100%;image-rendering:pixelated;border:1px solid #333}
p{font-size:14px;color:#999}</style></head><body>
<h2>GenX320 event camera - live</h2>
<img id="v" src="/stream" width="640" height="640" alt="live event stream">
<p>White = brighter (ON), blue = darker (OFF). Only moving things show up.</p>
<script>
// reconnect automatically if the stream drops
const v=document.getElementById('v');
v.onerror=()=>setTimeout(()=>{v.src='/stream?'+Date.now()},1000);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"{status}\n".encode())
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = None
            try:
                while True:
                    j = latest_jpeg
                    if j is not None and j is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(j)}\r\n\r\n".encode()
                                         + j + b"\r\n")
                        last = j
                    time.sleep(1.0 / FPS / 2)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=renderer, daemon=True).start()
    log(f"Serving on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
