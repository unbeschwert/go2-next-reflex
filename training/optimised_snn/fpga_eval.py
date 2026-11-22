#!/usr/bin/env python3
"""FPGA-in-the-loop evaluation: replay dataset clips into the board's SPLEAT server and score
what the FPGA itself outputs - real accuracy, robot-style trigger statistics and latency.

Runs on the laptop (numpy only, no camera needed). The board must run the SPLEAT server with
the firmware + weights under test, and nothing else may feed it:
    ssh root@192.168.77.1 '/root/inference.sh stop'          # camera client off
    python3 fpga_eval.py --sessions 04,05,06 --limit 20       # 20 clips per session, ~1 min
    python3 fpga_eval.py --sessions 04,05,06                  # full test set, ~15 min
    python3 fpga_eval.py --selftest                           # protocol check without a board

Per clip it sends --gap empty frames (so the neurons forget the previous clip) and then the
100 frames of 10 ms, exactly as binned for training: 2 polarity channels, SIZE x SIZE, value =
events in the block / block area (--input-scale changes that). Each frame is one request
{id, payload float32[2][S][S]}; the answer carries the output spikes and the FPGA latency.
--save writes the per-frame output spikes to an .npz for comparison with the simulation.
"""
import argparse
import os
import re
import socket
import struct
import sys
import threading
import time
import zipfile

import numpy as np

CLASS_NAMES = ("Sit", "Lay down", "Hello", "Dance", "No action")
EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
CAM, T = 320, 100

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="192.168.77.1")
ap.add_argument("--port", type=int, default=3333)
ap.add_argument("--zip", default=os.path.expanduser("~/Downloads/durance-ai-handgesture-recognition (1).zip"))
ap.add_argument("--sessions", default="04,05,06")
ap.add_argument("--size", type=int, default=80, help="model input size (80 for best_model.json)")
ap.add_argument("--outputs", type=int, default=5)
ap.add_argument("--limit", type=int, default=0, help="max clips per session (spread over the classes)")
ap.add_argument("--gap", type=int, default=40, help="empty frames between clips")
ap.add_argument("--input-scale", type=float, default=0.0,
                help="multiply the frames (value = events / block area); default = block area, i.e. send the\n"
                     "block SUMS like the live camera client despot_send_and_forget does. The board config must then\n"
                     "have inputs.scaling = 4096 / block area (64 for 40x40, 256 for 80x80), measured 2026-11-22")
ap.add_argument("--save", default="", help="write per-frame outputs to this .npz")
ap.add_argument("--timeout", type=float, default=5.0)
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--class-names", default=",".join(CLASS_NAMES),
                help="comma list, one per output, last = no gesture; our gestures: "
                     "'Heart,Turn left,Turn right,Namaskaram,No action'")
ap.add_argument("--as-none", action="store_true",
                help="count every clip as the last class (no gesture): false-trigger test on foreign "
                     "motion, e.g. the organisers' zip with our model")
ap.add_argument("--silent-thr", type=float, default=0.0,
                help="models trained with --silent-noaction (last output never fires): score the last "
                     "class as if it fired at this rate, e.g. 0.4 -> 'no gesture' unless a gesture output "
                     "fires in more than 40 %% of the frames")
args = ap.parse_args()
CLASS_NAMES = tuple(args.class_names.split(","))
assert len(CLASS_NAMES) == args.outputs, "one class name per output"
S = args.size
SCALE = CAM // S
if not args.input_scale:
    args.input_scale = float(SCALE * SCALE)   # block sums, as the camera client sends them


# ---------------------------------------------------------------- protocol
def cbor_uint(n):
    if n < 24:
        return bytes([n])
    if n < 1 << 8:
        return b"\x18" + bytes([n])
    if n < 1 << 16:
        return b"\x19" + struct.pack(">H", n)
    return b"\x1a" + struct.pack(">I", n)


def cbor_bytes_header(n):
    head = cbor_uint(n)
    return bytes([head[0] | 0x40]) + head[1:]


def request(msg_id, frame):
    payload = np.ascontiguousarray(frame, dtype="<f4").tobytes()
    body = b"\xa2\x62id" + cbor_uint(msg_id) + b"\x67payload" + cbor_bytes_header(len(payload)) + payload
    return struct.pack(">I", len(body)) + body


def read_uint(buf, pos):
    b = buf[pos]
    if b < 24:
        return b
    if b == 0x18:
        return buf[pos + 1]
    if b == 0x19:
        return struct.unpack(">H", buf[pos + 1:pos + 3])[0]
    if b == 0x1a:
        return struct.unpack(">I", buf[pos + 1:pos + 5])[0]
    if b == 0x1b:
        return struct.unpack(">Q", buf[pos + 1:pos + 9])[0]
    raise ValueError(f"unexpected CBOR byte {b:#x}")


def parse_reply(body):
    """-> (id, output spikes, latency in microseconds or None)"""
    i = body.find(b"\x62id")
    msg_id = read_uint(body, i + 3) if i >= 0 else None
    k = body.find(b"\x67payload")
    h = body[k + 8]
    if 0x40 <= h < 0x58:
        n, off = h - 0x40, k + 9
    elif h == 0x58:
        n, off = body[k + 9], k + 10
    else:
        n, off = struct.unpack(">H", body[k + 9:k + 11])[0], k + 11
    out = np.frombuffer(body[off:off + n], "<f4")
    j = body.find(b"\x67latency")
    lat = read_uint(body, j + 8) if j >= 0 else None
    return msg_id, out, lat


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection")
        buf += chunk
    return bytes(buf)


def fake_server(port, n_out):
    """Protocol self-test: answers every request with a spike on output (id % n_out)."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    try:
        while True:
            n = struct.unpack(">I", recv_exact(conn, 4))[0]
            body = recv_exact(conn, n)
            msg_id = read_uint(body, body.find(b"\x62id") + 3)
            out = np.zeros(n_out, "<f4")
            out[msg_id % n_out] = 1.0
            rep = (b"\xa4\x62id" + cbor_uint(msg_id) + b"\x67payload" + cbor_bytes_header(out.nbytes) + out.tobytes()
                   + b"\x67latency" + cbor_uint(7000 + msg_id % 500) + b"\x65error\x60")
            conn.sendall(struct.pack(">I", len(rep)) + rep)
    except (ConnectionError, OSError):
        pass


# ---------------------------------------------------------------- data
def load_clips():
    sessions = args.sessions.split(",")
    per = {}
    with zipfile.ZipFile(args.zip) as zf:
        for n in sorted(zf.namelist()):
            m = re.match(r"^motion_buddy_dataset/(\d+)/(gesture_(\d+)-[^/]+\.bin)$", n)
            if m and not m.group(2).startswith(".") and m.group(1) in sessions:
                cls = len(CLASS_NAMES) - 1 if args.as_none else int(m.group(3)) - 1
                per.setdefault((m.group(1), cls), []).append(n)
        items = []
        for (sess, cls), names in sorted(per.items()):
            if args.limit:
                names = names[:max(1, args.limit // len(CLASS_NAMES))]
            items += [(n, cls, int(sess)) for n in names]
        clips = []
        for name, cls, sess in items:
            data = zf.read(name)
            ev = np.frombuffer(data, EV, offset=4)
            t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])) // 10_000
            ok = (t >= 0) & (t < T) & (ev["x"] < CAM) & (ev["y"] < CAM) & (ev["polarity"] < 2)
            idx = ((t[ok] * 2 + ev["polarity"][ok].astype(np.int64)) * S + ev["y"][ok].astype(np.int64) // SCALE) * S \
                + ev["x"][ok].astype(np.int64) // SCALE
            frames = np.bincount(idx, minlength=T * 2 * S * S).reshape(T, 2, S, S).astype(np.float32)
            clips.append((frames / (SCALE * SCALE) * args.input_scale, cls, sess))
    return clips


def main():
    if args.selftest:
        args.host, args.port = "127.0.0.1", 39333
        threading.Thread(target=fake_server, args=(args.port, args.outputs), daemon=True).start()
        time.sleep(0.3)
        rng = np.random.default_rng(0)
        clips = [(rng.random((T, 2, S, S), dtype=np.float32), c, 1) for c in range(len(CLASS_NAMES))]
    else:
        print(f"loading clips of sessions {args.sessions} ...", flush=True)
        clips = load_clips()
    print(f"{len(clips)} clips, {S}x{S} input, {args.gap} empty frames between clips -> "
          f"{len(clips) * (T + args.gap)} frames to {args.host}:{args.port}", flush=True)

    sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(args.timeout)
    blank = np.zeros((2, S, S), np.float32)
    next_id = 1_000_000
    lat, outputs, t0 = [], [], time.time()

    def infer(frame):
        nonlocal next_id
        next_id += 1
        sock.sendall(request(next_id, frame))
        while True:  # replies are broadcast to every client: skip any that are not ours
            n = struct.unpack(">I", recv_exact(sock, 4))[0]
            msg_id, out, l = parse_reply(recv_exact(sock, n))
            if msg_id == next_id:
                if l is not None:
                    lat.append(l)
                return out

    for ci, (frames, cls, sess) in enumerate(clips):
        for _ in range(args.gap):
            infer(blank)
        spikes = np.stack([infer(f) for f in frames])           # [T, outputs]
        outputs.append(spikes)
        if (ci + 1) % 25 == 0 or ci + 1 == len(clips):
            el = time.time() - t0
            print(f"  {ci + 1}/{len(clips)} clips, {el:.0f}s, {(ci + 1) * (T + args.gap) / el:.0f} frames/s", flush=True)
    sock.close()

    out = np.stack(outputs)                                      # [N, T, outputs]
    y = np.array([c for _, c, _ in clips])
    sess = np.array([s for _, _, s in clips])
    counts = (out > 0).sum(1)
    if args.silent_thr:
        counts[:, -1] = round(args.silent_thr * T)       # 'no gesture' = no gesture output above the threshold
    pred = counts.argmax(1)
    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), int)
    for t_, p_ in zip(y, pred):
        conf[t_, p_] += 1
    g = counts[:, :-1] / T
    order = np.sort(g, axis=1)
    fire = (order[:, -1] >= 0.5) & (order[:, -1] - order[:, -2] >= 0.2)
    gest, hit = y < len(CLASS_NAMES) - 1, g.argmax(1) == y
    ng, nn = max(1, gest.sum()), max(1, (~gest).sum())
    print(f"\nFPGA accuracy {np.mean(pred == y):.1%} | per session "
          + "  ".join(f"{k:02d}: {np.mean(pred[sess == k] == y[sess == k]):.1%}" for k in sorted(set(sess.tolist()))))
    print(f"trigger rule: ok {np.sum(gest & fire & hit) / ng:.1%} wrong {np.sum(gest & fire & ~hit) / ng:.1%} "
          f"missed {np.sum(gest & ~fire) / ng:.1%} | false trigger on No action {np.sum(~gest & fire) / nn:.1%}")
    print("mean output firing rate per class (rows true class, cols outputs):")
    for c in range(len(CLASS_NAMES)):
        if (y == c).any():
            print(f"  {CLASS_NAMES[c]:10s} " + " ".join(f"{r:5.2f}" for r in counts[y == c].mean(0) / T))
    print("confusion (rows true, cols predicted):")
    print(conf)
    if lat:
        a = np.array(lat) / 1000.0
        print(f"FPGA latency per frame (ms): median {np.median(a):.2f}  mean {a.mean():.2f}  p90 {np.percentile(a, 90):.2f}  "
              f"p99 {np.percentile(a, 99):.2f}  max {a.max():.2f}   ({np.mean(a > 10):.1%} of frames over the 10 ms frame period)")
    if args.save:
        np.savez_compressed(args.save, outputs=out, labels=y, sessions=sess, latency_us=np.array(lat))
        print("saved", args.save)


if __name__ == "__main__":
    main()
