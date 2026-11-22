#!/usr/bin/env python3
"""Find the number interpretation under which the simulation reproduces the real FPGA.

Input: an exported weights file, the architecture, and the FPGA's own outputs for dataset clips
(recorded with go2_viz/fpga_eval.py --save). Simulates a grid of hypotheses - gain on the input
of the first layer, scale of the biases, scale of the other layers' weights - and ranks them by
how closely the simulated spike counts match the FPGA's, clip by clip.

    python fit_fpga.py --params ref/board40_weights_q8.txt --frac 7 --arch nb40 \
        --fpga ref/fpga_board40_test.npz --sessions 04,05,06
"""
import argparse
import itertools
import os
import re
import struct
import zipfile
from multiprocessing import Pool

import numpy as np
import torch

from riverops.nn import FixedPointFormat, QuantConfig, SpikingConv2d, prepare_fixed_point

EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
CAM, T = 320, 100
ARCHS = {
    "best80": (80, [(2, 16, 5, 2, 1), (16, 32, 3, 1, 1), (32, 32, 6, 2, 2),
                    (32, 256, 6, 2, 0), (256, 128, 3, 2, 0), (128, 5, 3, 1, 0)]),
    "nb40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 9, 1, 0), (256, 5, 1, 1, 0)]),
    "notebook32": (32, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 6, 2, 0), (256, 5, 1, 1, 0)]),
}
ap = argparse.ArgumentParser()
ap.add_argument("--zip", default="data/motion_buddy_full.zip")
ap.add_argument("--params", required=True)
ap.add_argument("--arch", choices=list(ARCHS), default="nb40")
ap.add_argument("--frac", type=int, default=7)
ap.add_argument("--fpga", required=True, help=".npz from fpga_eval.py --save")
ap.add_argument("--sessions", default="04,05,06")
ap.add_argument("--clips", type=int, default=0, help="use only the first N clips (0 = all)")
ap.add_argument("--in-gains", default="1,2,4,8,16,32,64,128,256")
ap.add_argument("--bias-scales", default="1,0.5,0.25,0.125,0.0625,0.03125,0")
ap.add_argument("--weight-scales", default="1")
ap.add_argument("--fixed", action="store_true", help="saturating fixed-point potentials (Q4.12) as in the hardware")
ap.add_argument("--pot-bits", default="16,12", help="potential format: total,frac bits")
ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
args = ap.parse_args()
SIZE, LAYERS = ARCHS[args.arch]
SCALE = CAM // SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vals = [int(v) for v in open(args.params).read().split()]
n_w = [ci * co * k * k for (ci, co, k, _, _) in LAYERS]
n_b = [co for (_, co, _, _, _) in LAYERS]
assert len(vals) == sum(n_w) + sum(n_b) + 2 * len(LAYERS), "file does not fit the architecture"
w_vals, rest = vals[:sum(n_w)], vals[sum(n_w):]
b_vals, neuron = rest[:sum(n_b)], rest[sum(n_b):]
scale = float(1 << args.frac)


def build(bias_scale, weight_scale):
    layers, wo, bo = [], 0, 0
    for i, (ci, co, k, s, p) in enumerate(LAYERS):
        thr, leak = neuron[2 * i] / 4096.0, neuron[2 * i + 1] / 4096.0
        layer = SpikingConv2d(ci, co, kernel_size=k, stride=s, padding=p, bias=True, first_layer=(i == 0),
                              initial_tau=1.0 / leak, learnable_leak=False, v_threshold=thr, reset_mode="HARD")
        w = torch.tensor(w_vals[wo:wo + n_w[i]], dtype=torch.float32).reshape(co, ci, k, k) / scale
        b = torch.tensor(b_vals[bo:bo + n_b[i]], dtype=torch.float32) / scale
        with torch.no_grad():
            layer.conv.weight.copy_(w * (weight_scale if i > 0 else 1.0))
            layer.conv.bias.copy_(b * bias_scale)
        wo, bo = wo + n_w[i], bo + n_b[i]
        layers.append(layer)
    net = torch.nn.Sequential(*layers)
    if args.fixed:
        tb, fb = (int(v) for v in args.pot_bits.split(","))
        qc = QuantConfig(weight_format=FixedPointFormat(16, 12), potential_format=FixedPointFormat(tb, fb),
                         enable_weight_quant=True, enable_potential_quant=True)
        net = prepare_fixed_point(net, qc)
    return net.to(device).eval()


_zip = None


def _init(path):
    global _zip
    _zip = zipfile.ZipFile(path)


def bin_clip(name):
    data = _zip.read(name)
    ev = np.frombuffer(data, EV, offset=4)
    t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])) // 10_000
    ok = (t >= 0) & (t < T) & (ev["x"] < CAM) & (ev["y"] < CAM) & (ev["polarity"] < 2)
    idx = ((t[ok] * 2 + ev["polarity"][ok].astype(np.int64)) * SIZE + ev["y"][ok].astype(np.int64) // SCALE) * SIZE \
        + ev["x"][ok].astype(np.int64) // SCALE
    return np.minimum(np.bincount(idx, minlength=T * 2 * SIZE * SIZE), 32767).astype(np.int16).reshape(T, 2, SIZE, SIZE)


sessions = args.sessions.split(",")
per = {}
with zipfile.ZipFile(args.zip) as zf:
    for n in sorted(zf.namelist()):
        m = re.match(r"^motion_buddy_dataset/(\d+)/(gesture_(\d+)-[^/]+\.bin)$", n)
        if m and not m.group(2).startswith(".") and m.group(1) in sessions:
            per.setdefault((m.group(1), int(m.group(3)) - 1), []).append(n)
items = [(n, cls) for (sess, cls), names in sorted(per.items()) for n in names]
fp = np.load(args.fpga)
y = fp["labels"]
assert len(items) == len(y) and all(c == l for (_, c), l in zip(items, y)), "clip list differs from the FPGA run"
N = args.clips or len(items)
with Pool(args.workers, initializer=_init, initargs=(args.zip,)) as pool:
    X = torch.from_numpy(np.stack(pool.map(bin_clip, [n for n, _ in items[:N]], chunksize=4)))
y = torch.from_numpy(y[:N])
fpga_train = torch.from_numpy((fp["outputs"][:N] > 0).astype(np.float32))      # [N, T, 5] spikes per frame
fpga_counts = fpga_train.sum(1)
print(f"{N} clips; FPGA accuracy {float((fpga_counts.argmax(1) == y).float().mean()):.1%}; "
      f"FPGA mean output rate {float(fpga_counts.mean()) / T:.3f}", flush=True)


@torch.no_grad()
def simulate(model, gain):
    outs = []
    for i in range(0, N, 64):
        x = X[i:i + 64].to(device).float() / (SCALE * SCALE) * gain
        for layer in model:
            x = layer(x)
        outs.append(x.flatten(2).cpu())          # [B, T, 5]
    return torch.cat(outs)


rows = []
for bs, ws in itertools.product([float(v) for v in args.bias_scales.split(",")], [float(v) for v in args.weight_scales.split(",")]):
    model = build(bs, ws)
    for g in [float(v) for v in args.in_gains.split(",")]:
        tr = simulate(model, g)
        c = tr.sum(1)
        rows.append({"gain": g, "bias": bs, "w": ws,
                     "acc": float((c.argmax(1) == y).float().mean()),
                     "agree": float((c.argmax(1) == fpga_counts.argmax(1)).float().mean()),
                     "count_err": float((c - fpga_counts).abs().mean()),
                     "frame_match": float((tr == fpga_train).float().mean()),
                     "exact_clips": float((tr == fpga_train).flatten(1).all(1).float().mean()),
                     "rate": float(c.mean()) / T})
rows.sort(key=lambda r: r["count_err"])
print(f"{'in gain':>8s} {'bias x':>8s} {'w x':>5s} | {'sim acc':>7s} {'same pred':>9s} {'count err':>9s} {'frame match':>11s} {'exact clips':>11s} {'mean rate':>9s}")
for r in rows[:25]:
    print(f"{r['gain']:8.0f} {r['bias']:8.4f} {r['w']:5.2f} | {r['acc']:7.1%} {r['agree']:9.1%} {r['count_err']:9.2f} "
          f"{r['frame_match']:11.2%} {r['exact_clips']:11.1%} {r['rate']:9.3f}")
print("count err = mean |simulated - FPGA| spikes per output over 100 steps; frame match = identical output bits")
