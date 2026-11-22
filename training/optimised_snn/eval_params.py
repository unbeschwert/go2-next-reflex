#!/usr/bin/env python3
"""Score an exported SPLEAT parameter file (weights.txt layout) in simulation.

Rebuilds the fused network from the flat integer file (all weights, all biases, then one
threshold/leak pair per layer) and evaluates it on dataset sessions, in float arithmetic and in
riverops' fixed-point emulation: accuracy, per-session accuracy, confusion, firing rates,
SynOps per 10 ms step, and the robot trigger rule - at the bridge's 0.5 threshold and at the
threshold tuned on --val-sessions for <= --max-false false triggers on No-action clips.

    python eval_params.py --params runs/<run>/weights_q16.txt --frac 12 --sessions 04,05,06 --val-sessions 13
    python eval_params.py --params ref/weights_shipped_q8.txt --frac 7        # 8-bit Q1.7 weights
    python eval_params.py ... --save sim_outputs.npz    # spike counts per clip (same clip order as fpga_eval.py)

Use it to check an export before deploying it, and to compare models on equal terms.
"""
import argparse
import os
import re
import struct
import zipfile
from multiprocessing import Pool

import numpy as np
import torch

from riverops.nn import FixedPointFormat, QuantConfig, SpikingConv2d, prepare_fixed_point

CLASS_NAMES = ("Sit", "Lay down", "Hello", "Dance", "No action")
EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
CAM, T = 320, 100
ARCHS = {
    "best80": (80, [(2, 16, 5, 2, 1), (16, 32, 3, 1, 1), (32, 32, 6, 2, 2),
                    (32, 256, 6, 2, 0), (256, 128, 3, 2, 0), (128, 5, 3, 1, 0)]),
    "notebook32": (32, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2),
                        (8, 256, 6, 2, 0), (256, 5, 1, 1, 0)]),
    "slim80": (80, [(2, 8, 5, 2, 1), (8, 16, 3, 1, 1), (16, 16, 6, 2, 2),
                    (16, 128, 6, 2, 0), (128, 64, 3, 2, 0), (64, 5, 3, 1, 0)]),
    "c40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 2, 1), (16, 64, 6, 2, 0), (64, 5, 3, 1, 0)]),
    # notebook network adapted to 40x40 (4th kernel 9): the 8-bit firmware found on the board on 2026-11-21
    "nb40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 9, 1, 0), (256, 5, 1, 1, 0)]),
    "s32": (32, [(2, 16, 5, 2, 1), (16, 32, 3, 2, 1), (32, 64, 3, 2, 0), (64, 5, 3, 1, 0)]),
}

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default="data/motion_buddy_full.zip")
ap.add_argument("--params", required=True)
ap.add_argument("--arch", choices=list(ARCHS), default="best80")
ap.add_argument("--frac", type=int, default=12, help="fractional bits of weights/biases: 12 (16-bit) or 7 (8-bit)")
ap.add_argument("--sessions", default="04,05,06")
ap.add_argument("--val-sessions", default="", help="tune the trigger threshold on these sessions")
ap.add_argument("--max-false", type=float, default=0.01)
ap.add_argument("--input-scale", type=float, default=1.0, help="multiply the input frames (e.g. 16 if the board sums instead of averaging)")
ap.add_argument("--limit", type=int, default=0, help="max clips per session (spread over the classes)")
ap.add_argument("--save", default="")
ap.add_argument("--quiet", action="store_true", help="only the summary lines")
ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
ap.add_argument("--allow-cpu", action="store_true")
args = ap.parse_args()
SIZE, LAYERS = ARCHS[args.arch]
SCALE = CAM // SIZE
FANOUT = [co * k * k / (s * s) for (_, co, k, s, _) in LAYERS]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and not args.allow_cpu:
    raise SystemExit("CUDA not available (pass --allow-cpu)")

# ---------------------------------------------------------------- parameters -> network
vals = [int(v) for v in open(args.params).read().split()]
n_w = [ci * co * k * k for (ci, co, k, _, _) in LAYERS]
n_b = [co for (_, co, _, _, _) in LAYERS]
expected = sum(n_w) + sum(n_b) + 2 * len(LAYERS)
if len(vals) != expected:
    raise SystemExit(f"{args.params}: {len(vals)} values, but {args.arch} needs {expected}")
w_vals, rest = vals[:sum(n_w)], vals[sum(n_w):]
b_vals, neuron = rest[:sum(n_b)], rest[sum(n_b):]
scale = float(1 << args.frac)
layers, wo, bo = [], 0, 0
for i, (ci, co, k, s, p) in enumerate(LAYERS):
    thr, leak = neuron[2 * i] / 4096.0, neuron[2 * i + 1] / 4096.0
    layer = SpikingConv2d(ci, co, kernel_size=k, stride=s, padding=p, bias=True, first_layer=(i == 0),
                          initial_tau=1.0 / leak, learnable_leak=False, v_threshold=thr, reset_mode="HARD")
    w = torch.tensor(w_vals[wo:wo + n_w[i]], dtype=torch.float32).reshape(co, ci, k, k) / scale
    b = torch.tensor(b_vals[bo:bo + n_b[i]], dtype=torch.float32) / scale
    with torch.no_grad():
        layer.conv.weight.copy_(w)
        layer.conv.bias.copy_(b)
    wv, bv = np.array(w_vals[wo:wo + n_w[i]]), b_vals[bo:bo + n_b[i]]
    wo, bo = wo + n_w[i], bo + n_b[i]
    layers.append(layer)
    if not args.quiet:
        print(f"layer {i}: {ci:3d}->{co:3d} k{k} s{s}  weights [{wv.min():7d}, {wv.max():6d}] zero {np.mean(wv == 0):5.1%}"
              f"  bias [{min(bv):6d}, {max(bv):6d}]  threshold {thr:.3f} leak {leak:.4f}")
net = torch.nn.Sequential(*layers).to(device).eval()
qc = QuantConfig(weight_format=FixedPointFormat(16 if args.frac == 12 else 8, args.frac),
                 potential_format=FixedPointFormat(16, 12), enable_weight_quant=True, enable_potential_quant=True)
net_fixed = prepare_fixed_point(net, qc).to(device).eval()

# ---------------------------------------------------------------- data
_zip = None


def _init(path):
    global _zip
    _zip = zipfile.ZipFile(path)


def bin_clip(name):
    data = _zip.read(name)
    n = struct.unpack("!I", data[:4])[0]
    if n != len(data) - 4 or n % EV.itemsize or n == 0:
        return None
    ev = np.frombuffer(data, EV, offset=4)
    t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])) // 10_000
    ok = (t >= 0) & (t < T) & (ev["x"] < CAM) & (ev["y"] < CAM) & (ev["polarity"] < 2)
    idx = ((t[ok] * 2 + ev["polarity"][ok].astype(np.int64)) * SIZE + ev["y"][ok].astype(np.int64) // SCALE) * SIZE \
        + ev["x"][ok].astype(np.int64) // SCALE
    return np.minimum(np.bincount(idx, minlength=T * 2 * SIZE * SIZE), 32767).astype(np.int16).reshape(T, 2, SIZE, SIZE)


def load(sessions):
    """Clips ordered by session, class, name (the order fpga_eval.py uses)."""
    per = {}
    with zipfile.ZipFile(args.zip) as zf:
        for n in sorted(zf.namelist()):
            m = re.match(r"^motion_buddy_dataset/(\d+)/(gesture_(\d+)-[^/]+\.bin)$", n)
            if m and not m.group(2).startswith(".") and m.group(1) in sessions:
                per.setdefault((m.group(1), int(m.group(3)) - 1), []).append(n)
    items = []
    for (sess, cls), names in sorted(per.items()):
        if args.limit:
            names = names[:max(1, args.limit // len(CLASS_NAMES))]
        items += [(n, cls, int(sess)) for n in names]
    with Pool(args.workers, initializer=_init, initargs=(args.zip,)) as pool:
        frames = pool.map(bin_clip, [n for n, _, _ in items], chunksize=4)
    keep = [i for i, f in enumerate(frames) if f is not None]
    return (torch.from_numpy(np.stack([frames[i] for i in keep])), torch.tensor([items[i][1] for i in keep]),
            torch.tensor([items[i][2] for i in keep]), [items[i][0] for i in keep])


@torch.no_grad()
def forward(model, X):
    """-> spike counts per class [N, 5], SynOps/step, firing rate per layer"""
    counts, syn, rate_sum, neurons = [], 0.0, None, None
    for i in range(0, len(X), 32):
        x = X[i:i + 32].to(device).float() / (SCALE * SCALE) * args.input_scale
        active = [(x > 0).flatten(1).sum(1).float()]
        neurons = []
        for layer in model:
            x = layer(x)
            active.append(x.flatten(1).sum(1))
            neurons.append(x[0, 0].numel())
        counts.append(x.sum(1).flatten(1).cpu())
        syn += (sum(a * f for a, f in zip(active[:-1], FANOUT)) / T).sum().item()
        r = torch.stack([a.sum() for a in active[1:]])
        rate_sum = r if rate_sum is None else rate_sum + r
    rates = [float(rs) / (len(X) * T * nn_) for rs, nn_ in zip(rate_sum, neurons)]
    return torch.cat(counts), syn / len(X), rates


def trigger(counts, y, thr, margin=0.2):
    g = counts[:, :-1] / T
    top2 = g.topk(2, dim=1)
    fire = (top2.values[:, 0] >= thr) & (top2.values[:, 0] - top2.values[:, 1] >= margin)
    gest, hit = y < 4, top2.indices[:, 0] == y
    ng, nn = max(1, int(gest.sum())), max(1, int((~gest).sum()))
    return (int((gest & fire & hit).sum()) / ng, int((gest & fire & ~hit).sum()) / ng,
            int((gest & ~fire).sum()) / ng, int((~gest & fire).sum()) / nn)


def tuned_threshold(counts, y):
    for thr in [x / 100 for x in range(30, 100, 2)]:
        if trigger(counts, y, thr)[3] <= args.max_false:
            return thr
    return 0.98


X, Y, S, names = load(args.sessions.split(","))
val = load(args.val_sessions.split(",")) if args.val_sessions else None
print(f"{len(Y)} test clips (sessions {args.sessions})" + (f", {len(val[1])} validation clips ({args.val_sessions})" if val else "")
      + f" | params {args.params} ({len(vals)} values, frac {args.frac})")
saved = {}
for name, model in (("float", net), ("fixed-point", net_fixed)):
    counts, syn, rates = forward(model, X)
    pred = counts.argmax(1)
    acc = float((pred == Y).float().mean())
    ok, wrong, missed, false = trigger(counts, Y, 0.5)
    line = (f"[{name:11s}] acc {acc:.1%} (" + " ".join(f"{k:02d}:{float((pred[S == k] == Y[S == k]).float().mean()):.0%}"
                                                      for k in sorted(set(S.tolist()))) + ")"
            f" | rule 0.5: ok {ok:.1%} wrong {wrong:.1%} missed {missed:.1%} false {false:.1%}")
    if val:
        vc, _, _ = forward(model, val[0])
        thr = tuned_threshold(vc, val[1])
        ok, wrong, missed, false = trigger(counts, Y, thr)
        line += (f" | tuned thr {thr:.2f}: ok {ok:.1%} wrong {wrong:.1%} false {false:.1%}"
                 f" | val acc {float((vc.argmax(1) == val[1]).float().mean()):.1%}")
    print(line + f" | SynOps {syn / 1e3:.0f}k | rates% " + " ".join(f"{100 * r:.1f}" for r in rates))
    if not args.quiet:
        conf = torch.zeros(5, 5, dtype=torch.long)
        for t_, p_ in zip(Y, pred):
            conf[t_, p_] += 1
        print(conf.numpy())
    saved[name.replace("-", "_")] = counts.numpy()
if args.save:
    np.savez_compressed(args.save, labels=Y.numpy(), sessions=S.numpy(), names=np.array(names), **saved)
    print("saved", args.save)
