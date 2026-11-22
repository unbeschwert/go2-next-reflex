#!/usr/bin/env python3
"""Runs ON THE KRIA. Records what the SNN really gets and answers during live inference.

Passively sniffs the loopback TCP connection camera client -> SPLEAT server (like
snn_input_viewer.py, which keeps running): every input frame (float32[2][S][S]) and every
output (5 spike flags) with its arrival time. Nothing is modified.

    python3 /root/snn_record.py SECONDS /tmp/live.npz
    -> inputs [N,2,S,S] float16, t_in [N], outputs [M,5] uint8, t_out [M]
"""
import sys
import threading
import time

import numpy as np

secs, out_path = float(sys.argv[1]), sys.argv[2]
sys.argv = [sys.argv[0], "8099", "own"]          # the viewer module reads argv at import
sys.path.insert(0, "/root")
import snn_input_viewer as V                       # noqa: E402  (no server is started on import)

ins, t_in, outs, t_out = [], [], [], []


def on_input(body):
    p, _ = V.parse_payload(body)
    if p is None or len(p) % 8:
        return
    side = int(round((len(p) // 8) ** 0.5))
    if side < 8 or side * side * 8 != len(p):
        return
    ins.append(np.frombuffer(p, "<f4").reshape(2, side, side).astype(np.float16))
    t_in.append(time.time())


def on_output(body):
    p, _ = V.parse_payload(body)
    if p is None or len(p) != 4 * V.N_OUT:
        return
    sc = np.frombuffer(p, "<f4")
    outs.append((sc > 0).astype(np.uint8))
    t_out.append(time.time())


V.on_input, V.on_output = on_input, on_output
threading.Thread(target=V.sniff, daemon=True).start()
print(f"recording {secs:.0f} s ...", flush=True)
t0 = time.time()
while time.time() - t0 < secs:
    time.sleep(1)
    print(f"  {time.time() - t0:3.0f} s: {len(ins)} input frames, {len(outs)} outputs", flush=True)
np.savez_compressed(out_path, inputs=np.stack(ins) if ins else np.zeros((0, 2, 1, 1), np.float16),
                    t_in=np.array(t_in), outputs=np.stack(outs) if outs else np.zeros((0, 5), np.uint8),
                    t_out=np.array(t_out))
print("saved", out_path, flush=True)
