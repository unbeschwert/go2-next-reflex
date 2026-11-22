#!/usr/bin/env python3
"""Run a weights file through the integer FPGA model (spleat_sim.py) on dataset clips.

    python3 sim_check.py WEIGHTS.txt [--arch nb40] [--sessions 04,05,06] [--per-class 20] [--gain 1]
                         [--fpga fpga_outputs.npz]    # compare with outputs recorded from the board

A pre-flight check before deploying: accuracy and trigger statistics as the hardware model predicts
them (8-bit firmware: weights and biases = integer / 64, saturating potentials).
"""
import argparse, os, re, zipfile
import numpy as np
from spleat_sim import ARCHS, Spleat

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("--arch", default="nb40")
ap.add_argument("--zip", default=os.path.expanduser("~/Downloads/durance-ai-handgesture-recognition (1).zip"))
ap.add_argument("--sessions", default="04,05,06")
ap.add_argument("--per-class", type=int, default=20, help="clips per class and session (0 = all)")
ap.add_argument("--gain", type=float, default=1.0, help="input gain (board config scaling / 4096)")
ap.add_argument("--fpga", default="", help=".npz from fpga_eval.py --save for the same sessions (all clips)")
ap.add_argument("--save", default="", help="write per-frame outputs + labels + sessions to this .npz (as fpga_eval.py --save)")
a = ap.parse_args()
S = ARCHS[a.arch][0]; SC = 320 // S
EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
z = zipfile.ZipFile(a.zip)
per = {}
for n in sorted(z.namelist()):
    m = re.match(r"^motion_buddy_dataset/(\d+)/(gesture_(\d+)-[^/]+\.bin)$", n)
    if m and not m.group(2).startswith(".") and m.group(1) in a.sessions.split(","):
        per.setdefault((m.group(1), int(m.group(3)) - 1), []).append(n)
items, pos = [], 0
for (sess, cls), names in sorted(per.items()):
    for j, n in enumerate(names):
        if not a.per_class or j < a.per_class:
            items.append((n, cls, pos + j))
    pos += len(names)
fp = (np.load(a.fpga)["outputs"] > 0).astype(int) if a.fpga else None
blank = np.zeros((2, S, S), np.float32)
conf = np.zeros((5, 5), int); counts, same, bits, frames_out = [], 0, [], []
for n, cls, idx in items:
    ev = np.frombuffer(z.read(n), EV, offset=4)
    t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])) // 10000
    ok = (t >= 0) & (t < 100) & (ev["x"] < 320) & (ev["y"] < 320)
    i = ((t[ok] * 2 + ev["polarity"][ok].astype(np.int64)) * S + ev["y"][ok].astype(np.int64) // SC) * S + ev["x"][ok].astype(np.int64) // SC
    clip = np.bincount(i, minlength=100 * 2 * S * S).reshape(100, 2, S, S).astype(np.float32) / (SC * SC) * a.gain
    m = Spleat(a.weights, a.arch, w_shift=6, b_shift=6, in_shift=6, leak_mode="retain")
    m.run([blank] * 60)
    out = m.run(clip)
    c = out.sum(0); counts.append(c); conf[cls, c.argmax()] += 1
    frames_out.append(out.astype(np.uint8))
    if fp is not None:
        same += int(c.argmax() == fp[idx].sum(0).argmax()); bits.append(np.mean(out == fp[idx]))
y = np.array([c for _, c, _ in items]); C = np.array(counts)
g = C[:, :4] / 100; srt = np.sort(g, 1); fire = (srt[:, -1] >= 0.5) & (srt[:, -1] - srt[:, -2] >= 0.2)
gest, hit = y < 4, g.argmax(1) == y
print(f"{len(items)} clips | hardware-model accuracy {np.mean(C.argmax(1) == y):.1%} | rule 0.5: ok {np.sum(gest & fire & hit) / max(1, gest.sum()):.1%} "
      f"wrong {np.sum(gest & fire & ~hit) / max(1, gest.sum()):.1%} missed {np.sum(gest & ~fire) / max(1, gest.sum()):.1%} "
      f"false {np.sum(~gest & fire) / max(1, (~gest).sum()):.1%} | mean output rate {C.mean() / 100:.3f}")
if fp is not None:
    print(f"vs FPGA on the same clips: same winner {same / len(items):.1%}, identical output bits {np.mean(bits):.1%}, "
          f"FPGA accuracy on them {np.mean([fp[i].sum(0).argmax() == c for _, c, i in items]):.1%}")
print(conf)
if a.save:
    np.savez_compressed(a.save, outputs=np.stack(frames_out), labels=y,
                        sessions=np.array([int(n.split("/")[1]) for n, _, _ in items]))
