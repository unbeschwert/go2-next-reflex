#!/usr/bin/env python3
"""Motion Buddy gesture SNN - accuracy + efficiency training for the SPLEAT FPGA (v9).

v9: options taken from the team's training/snn package (origin/main, 2026-11-22), all off by default:
  --arch v3        the v3 network (16-16-16-128-5, kernels 5,3,6,6,2). Layer 3 uses kernel 6 stride 2
                   instead of kernel 9 stride 1: 1,152 instead of 20,736 SynOps per layer-2 spike.
                   Needs a new firmware from model.json.
  --initial-tau    starting neuron time constant (v8: always 14.3). With --learnable-leak 1 every layer
                   learns its own leak (exported per layer in weights.txt).
  --layer-dropout  channel dropout on the currents of --dropout-layers during training and QAT; fusion
                   removes it, so the exported network is unchanged
  --binom-drop     remove up to this fraction of the events (binomial thinning of the event counts)
  --time-shift     move the clip content up to +-N frames, --speed play it faster / slower
  --polarity-p     swap the ON/OFF channels
  --mixup-p        blend two clips of the batch; their losses are blended with the same weight

v7: trains and scores the way the board runs. The FPGA never starts from zero potentials - it
runs on a continuous stream - but a network started from zero fires ~20 extra spikes on every
output before its inner layers settle (measured against the FPGA on 2026-11-21). So every clip
is now preceded by --warmup-steps frames of context (empty frames, or the tail of another clip
as in a real stream) and only the clip itself is counted. Scores use empty context, like
go2_viz/fpga_eval.py; 'ctx' is the accuracy when the context is another clip's tail.

v6: weight formats are named. '8h' = 8-bit weights as the 8-bit SPLEAT firmware really reads
them (value = integer / 64, range +-2; measured on the FPGA on 2026-11-21 - the notebook's Q1.7
reading, '8', does not reproduce the hardware). '16' = Q4.12 for the 16-bit firmware.
--input-gain trains on frames multiplied by a constant; deploy with `scaling: 4096 * gain` in
the board's config.yaml (keeps the first layer's folded weights inside the 8-bit range).

v5: also reports the robot operating point - the firing-rate threshold that keeps false
triggers on No-action clips <= --max-false on the validation session, and the gestures still
caught with it on the test set; evaluates the QAT model in float arithmetic too; saves the
per-clip output spike counts (outputs.npz).

v4: clips are exactly 1 s long, so only 1 s is stored; reports accuracy from the first
0.5 s (early decision) and robot-style trigger statistics (rate >= 0.5 and 0.2 lead, the
rule of kria/live_gesture_bridge.py); --noaction-weight; test split may hold several sessions.

v2 (all sessions, streaming crops, augmentation, AdamW/cosine/EMA, CE term, fusion-aware
QAT, fixed-point evaluation) plus what is needed to search for the best *and* cheapest net:

  efficiency   every evaluation reports spike rates per layer and synaptic operations per
               10 ms step (SynOps = active inputs of a layer x its fan-out). On the FPGA the
               inference latency follows spike activity, so SynOps is the cost we minimise.
  --spike-reg  penalises SynOps during training (sparser firing at equal accuracy)
  --aug strong zoom / rotation / shift / flip in one affine warp, random event dropout and
               noise events (the board's camera client filters noise; other rooms differ)
  architectures  best80 (= firmware already on the board) and compact nets that need a new
               firmware from the riversdk.dev generator: slim80, c40, s32, notebook32
  --qat-bits   QAT + export for 16-bit and 8-bit weights, each from its own fine-tuned model
  --init       start from a previous checkpoint (e.g. sparsity fine-tuning)
  shifted eval accuracy when the 1 s window starts 0.5 s into the clip (streaming robustness)

Outputs in --out: model.json, weights_q16.txt/.json, weights_q8.txt/.json, float_best.pt,
qat16_best.pt, qat8_best.pt, metrics.json.
"""
import argparse
import json
import math
import os
import re
import struct
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel

from riverops.nn import FixedPointFormat, QuantConfig, SpikingConvBN2d, SpikingConvBNDropout2d, prepare_ptq, prepare_qat

EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
CAM = 320
ARCHS = {  # input size, [(in, out, kernel, stride, padding), ...]
    # identical to best_model.json: runs on the firmwares already on the board
    "best80": (80, [(2, 16, 5, 2, 1), (16, 32, 3, 1, 1), (32, 32, 6, 2, 2),
                    (32, 256, 6, 2, 0), (256, 128, 3, 2, 0), (128, 5, 3, 1, 0)]),
    # the organisers' notebook network
    "notebook32": (32, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2),
                        (8, 256, 6, 2, 0), (256, 5, 1, 1, 0)]),
    # compact candidates (kernel sizes limited to those used by working SPLEAT models)
    "slim80": (80, [(2, 8, 5, 2, 1), (8, 16, 3, 1, 1), (16, 16, 6, 2, 2),
                    (16, 128, 6, 2, 0), (128, 64, 3, 2, 0), (64, 5, 3, 1, 0)]),
    "c40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 2, 1), (16, 64, 6, 2, 0), (64, 5, 3, 1, 0)]),
    # notebook network adapted to 40x40 (4th kernel 9): the 8-bit firmware found on the board on 2026-11-21
    "nb40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 9, 1, 0), (256, 5, 1, 1, 0)]),
    "s32": (32, [(2, 16, 5, 2, 1), (16, 32, 3, 2, 1), (32, 64, 3, 2, 0), (64, 5, 3, 1, 0)]),
    # 80x80 input at the cost of the 40x40 network: the first layer reduces by 4 (kernel 6, stride 4)
    # to the same 19x19 maps as nb40, the rest is identical. On the FPGA the latency follows the
    # number of neurons (nb40: 12k neurons, 0.74 ms; best80: 98k neurons, 7.5 ms). Needs a new firmware.
    "fast80": (80, [(2, 16, 6, 4, 0), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 9, 1, 0), (256, 5, 1, 1, 0)]),
    # v3 network of training/snn (motion_buddy_40_v3.yaml): 40 -> 19 -> 19 -> 9 -> 2 -> 1, 88.6k weights.
    # Layer 3 (kernel 6, stride 2) has 18x less fan-out than nb40's kernel 9. Needs a new firmware.
    "v3": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 16, 6, 2, 2), (16, 128, 6, 2, 0), (128, 5, 2, 1, 0)]),
}
def _qc(total, frac):
    return QuantConfig(weight_format=FixedPointFormat(total, frac), potential_format=FixedPointFormat(16, 12),
                       enable_weight_quant=True, enable_potential_quant=True)


QCONF = {"16": _qc(16, 12),   # 16-bit firmware: Q4.12
         "8h": _qc(8, 6),     # 8-bit firmware as measured on the FPGA: integer / 64
         "8": _qc(8, 7)}      # notebook's Q1.7 (does not match the hardware)

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default="data/motion_buddy_full.zip")
ap.add_argument("--out", default="runs/v3-local")
ap.add_argument("--arch", choices=list(ARCHS), default="best80")
ap.add_argument("--init", default="", help="checkpoint (state dict) to start from")
ap.add_argument("--val", default="04")
ap.add_argument("--test", default="05")
ap.add_argument("--train", default="auto", help="comma list or 'auto' = all other sessions")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--qat-epochs", type=int, default=8)
ap.add_argument("--qat-lr-scale", type=float, default=0.1, help="QAT learning rate = lr * this")
ap.add_argument("--qat-bits", default="16", help="weight formats to fine-tune: 16, 8h, 8 (comma list)")
ap.add_argument("--input-gain", type=float, default=1.0, help="multiply the input frames (deploy with scaling = 4096 * gain)")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--weight-decay", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=2, help="warmup epochs")
ap.add_argument("--ce-weight", type=float, default=0.5, help="0 = notebook loss only")
ap.add_argument("--ce-scale", type=float, default=10.0)
ap.add_argument("--spike-reg", type=float, default=0.0, help="weight of the SynOps penalty (per 1e6 SynOps/step)")
ap.add_argument("--noaction-weight", type=float, default=1.0, help="loss weight of No-action clips")
ap.add_argument("--max-false", type=float, default=0.01, help="allowed false-trigger rate for the tuned threshold")
ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--time-steps", type=int, default=100)
ap.add_argument("--warmup-steps", type=int, default=40, help="context frames before each clip (0 = start from zero state)")
ap.add_argument("--ctx-blank", type=float, default=0.3, help="training: share of clips preceded by empty frames (rest: tail of another clip)")
ap.add_argument("--store-seconds", type=float, default=1.0, help="dataset clips are exactly 1 s long")
ap.add_argument("--crop-max-s", type=float, default=0.5, help="max random start offset of the 1 s window")
ap.add_argument("--aug", choices=("basic", "strong"), default="basic")
ap.add_argument("--max-shift", type=int, default=-1, help="pixels; -1 = size/16")
ap.add_argument("--flip", type=int, default=1)
ap.add_argument("--zoom", type=float, default=0.15, help="strong: zoom range +-")
ap.add_argument("--rot", type=float, default=10.0, help="strong: rotation range +- degrees")
ap.add_argument("--event-drop", type=float, default=0.0, help="max fraction of events randomly removed")
ap.add_argument("--noise", type=float, default=0.0, help="max probability of a noise event per pixel and step")
ap.add_argument("--rate-jitter", type=float, default=0.25)
ap.add_argument("--learnable-leak", type=int, default=0)
ap.add_argument("--initial-tau", type=float, default=1.0 / (1.0 - 0.93),
                help="starting neuron time constant (14.3 = keep 93 %% per step, 2 = keep 50 %%)")
ap.add_argument("--layer-dropout", type=float, default=0.0, help="channel dropout on the currents of --dropout-layers")
ap.add_argument("--dropout-layers", default="1,2,3")
ap.add_argument("--binom-drop", type=float, default=0.0, help="max fraction of events removed (binomial thinning)")
ap.add_argument("--binom-drop-p", type=float, default=0.5)
ap.add_argument("--time-shift", type=int, default=0, help="max frames the clip content is moved (0 = off)")
ap.add_argument("--time-shift-p", type=float, default=0.5)
ap.add_argument("--speed", default="", help="playback speed range, e.g. 0.8,1.25 ('' = off)")
ap.add_argument("--speed-p", type=float, default=0.5)
ap.add_argument("--polarity-p", type=float, default=0.0, help="probability of swapping the ON/OFF channels")
ap.add_argument("--mixup-p", type=float, default=0.0, help="probability per batch of mixing two clips")
ap.add_argument("--mixup-alpha", type=float, default=0.4)
ap.add_argument("--limit", type=int, default=0, help="max clips per session (smoke tests)")
ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--allow-cpu", action="store_true")
ap.add_argument("--class-names", default="Sit,Lay down,Hello,Dance,No action",
                help="comma list, one per output; the last one is 'no gesture'")
ap.add_argument("--unused", default="", help="outputs without training clips (kept silent), e.g. 3")
ap.add_argument("--mirror", default="", help="labels swapped by the flip augmentation, e.g. 1:2 (left/right gestures)")
ap.add_argument("--neg-zip", default="", help="second dataset whose clips all count as 'no gesture' (foreign motion)")
ap.add_argument("--neg-train", default="", help="sessions of --neg-zip added to the training set")
ap.add_argument("--neg-eval", default="", help="sessions of --neg-zip kept apart: false-trigger test on foreign motion")
ap.add_argument("--neg-limit", type=int, default=0, help="max clips per --neg-zip training session (0 = all)")
ap.add_argument("--quiet-prob", type=float, default=0.0,
                help="share of training samples replaced by a quiet scene (a few scattered events per frame, "
                     "sometimes a flickering spot), labelled 'no gesture': the camera client sends such frames "
                     "when nothing moves and an untrained network fires on them by itself")
ap.add_argument("--quiet-events", type=float, default=12.0, help="quiet scene: up to this many camera events per frame")
ap.add_argument("--on-target", type=float, default=0.8, help="target firing rate of the correct output")
ap.add_argument("--off-target", type=float, default=0.2, help="target firing rate of the other outputs")
ap.add_argument("--silent-noaction", type=int, default=0,
                help="1 = 'no gesture' means all outputs silent (last output unused): the network idles when nothing happens")
ap.add_argument("--silent-thr", type=float, default=0.4,
                help="with --silent-noaction: a gesture counts when its output fires above this rate")
args = ap.parse_args()

CLASS_NAMES = tuple(args.class_names.split(","))
NEG = len(CLASS_NAMES) - 1
UNUSED = [int(v) for v in args.unused.split(",") if v]
SIZE, LAYERS = ARCHS[args.arch]
SCALE = CAM // SIZE
DENOM = float(SCALE * SCALE) / args.input_gain
NBINS = int(args.store_seconds * 100)
T = args.time_steps
W = args.warmup_steps
MAX_SHIFT = SIZE // 16 if args.max_shift < 0 else args.max_shift
FANOUT = [co * k * k / (s * s) for (_, co, k, s, _) in LAYERS]  # SynOps per active input of each layer
QAT_BITS = [b for b in args.qat_bits.split(",") if b]
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

assert LAYERS[-1][1] == len(CLASS_NAMES), "one class name per network output"
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
elif args.allow_cpu:
    device = torch.device("cpu")
else:
    raise SystemExit("CUDA not available - refusing to train on CPU (pass --allow-cpu)")
print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "",
      "| arch", args.arch, f"{SIZE}x{SIZE}", "| args", vars(args), flush=True)


# ------------------------------------------------------------------ data
_zip = None


def _init_worker(path):
    global _zip
    _zip = zipfile.ZipFile(path)


def bin_clip(name):
    data = _zip.read(name)
    n = struct.unpack("!I", data[:4])[0]
    if n != len(data) - 4 or n % EV.itemsize or n == 0:
        return name, None, 0
    ev = np.frombuffer(data, EV, offset=4)
    t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])) // 10_000
    ok = (t >= 0) & (t < NBINS) & (ev["x"] < CAM) & (ev["y"] < CAM) & (ev["polarity"] < 2)
    xs = ev["x"][ok].astype(np.int64) // SCALE
    ys = ev["y"][ok].astype(np.int64) // SCALE
    idx = ((t[ok] * 2 + ev["polarity"][ok].astype(np.int64)) * SIZE + ys) * SIZE + xs
    counts = np.bincount(idx, minlength=NBINS * 2 * SIZE * SIZE)  # block sums == area mean * SCALE^2
    dtype = np.uint8 if SCALE * SCALE <= 16 else np.int16  # CUDA cannot index uint16
    frames = np.minimum(counts, np.iinfo(dtype).max).astype(dtype).reshape(NBINS, 2, SIZE, SIZE)
    length = int(min(NBINS, t[ok].max() + 1)) if ok.any() else 0
    return name, frames, length


def list_members(zf):
    out = {}
    for n in zf.namelist():
        m = re.match(r"^motion_buddy_dataset/(\d+)/(gesture_(\d+)-[^/]+\.bin)$", n)
        if m and not m.group(2).startswith("."):
            out.setdefault(m.group(1), []).append((n, int(m.group(3)) - 1))
    return {s: sorted(v) for s, v in sorted(out.items())}


def load_split(members, sessions, pool, force_label=None, limit=0, sess_offset=0):
    rng = np.random.default_rng(args.seed)
    items = []
    for s in sessions:
        its = members[s][:args.limit] if args.limit else members[s]
        if limit and len(its) > limit:                       # random subset, all classes of that session
            its = [its[i] for i in sorted(rng.choice(len(its), limit, replace=False))]
        items += its
    names = [n for n, _ in items]
    lab = dict(items) if force_label is None else {n: force_label for n in names}
    t0 = time.time()
    X, L, Y, S = [], [], [], []
    for name, frames, length in pool.imap(bin_clip, names, chunksize=4):
        if frames is None:
            print("  skip invalid", name)
            continue
        X.append(frames); L.append(length); Y.append(lab[name]); S.append(int(name.split("/")[1]) + sess_offset)
    X = torch.from_numpy(np.stack(X))
    print(f"  sessions {','.join(sessions)}: {len(Y)} clips {tuple(X.shape)} {X.dtype} "
          f"{X.numel() * X.element_size() / 1e9:.1f} GB, {time.time() - t0:.0f}s", flush=True)
    return X, torch.tensor(L), torch.tensor(Y), torch.tensor(S)


with zipfile.ZipFile(args.zip) as zf:
    members = list_members(zf)
sessions = list(members)
val_s, test_s = args.val.split(","), args.test.split(",")
train_s = [s for s in sessions if s not in val_s + test_s] if args.train == "auto" else args.train.split(",")
print("sessions", sessions, "| train", train_s, "| val", val_s, "| test", test_s, flush=True)
with Pool(args.workers, initializer=_init_worker, initargs=(args.zip,)) as pool:
    data = {k: load_split(members, s, pool) for k, s in (("train", train_s), ("val", val_s), ("test", test_s))}
if args.neg_zip:                                  # foreign motion: every clip is 'no gesture'
    with zipfile.ZipFile(args.neg_zip) as zf:
        neg_members = list_members(zf)
    with Pool(args.workers, initializer=_init_worker, initargs=(args.neg_zip,)) as pool:
        if args.neg_train:
            extra = load_split(neg_members, args.neg_train.split(","), pool, NEG, args.neg_limit, 100)
            data["train"] = tuple(torch.cat([a, b]) for a, b in zip(data["train"], extra))
        if args.neg_eval:
            data["neg"] = load_split(neg_members, args.neg_eval.split(","), pool, NEG, 0, 100)
for k, v in data.items():
    print(f"  {k}: {len(v[2])} clips, per class {torch.bincount(v[2], minlength=len(CLASS_NAMES)).tolist()}", flush=True)
sess_of = {k: v[3] for k, v in data.items()}      # session number of every clip
data = {k: v[:3] for k, v in data.items()}


def to_device(split):
    X, L, Y = data[split]
    try:
        return X.to(device), L.to(device), Y.to(device)
    except torch.cuda.OutOfMemoryError:
        print(f"  {split} stays in CPU memory", flush=True)
        return X, L.to(device), Y.to(device)


data = {k: to_device(k) for k in data}


def quiet_counts(m, steps, gen=None, dev=None):
    """Event counts [m, steps, 2, S, S] of a quiet scene: up to --quiet-events scattered events per
    frame, and for half of the samples a small spot that flickers in every frame (lamp, screen)."""
    dev = dev or device
    r = lambda *shape: torch.rand(*shape, generator=gen, device=dev)
    lam = r(m, 1, 1, 1, 1) * args.quiet_events / (2 * SIZE * SIZE)
    x = torch.poisson(lam.expand(m, steps, 2, SIZE, SIZE), generator=gen)
    yy, xx = torch.meshgrid(torch.arange(SIZE, device=dev), torch.arange(SIZE, device=dev), indexing="ij")
    cy, cx = (r(m) * SIZE).floor(), (r(m) * SIZE).floor()
    rad = 1 + (r(m) * max(1, SIZE // 10)).floor()
    spot = ((yy[None] - cy[:, None, None]).abs() <= rad[:, None, None]) & ((xx[None] - cx[:, None, None]).abs() <= rad[:, None, None])
    rate = torch.where(r(m) < 0.5, r(m) * 0.6 * SCALE * SCALE / 16, torch.zeros(m, device=dev))   # events per block and frame
    x = x + torch.poisson((spot[:, None, None].float() * rate[:, None, None, None, None]).expand(m, steps, 2, SIZE, SIZE), generator=gen)
    return x


if args.quiet_prob > 0:                                 # fixed set for the report: nothing here may trigger
    _g = torch.Generator().manual_seed(1234)
    _xq = quiet_counts(200, NBINS, _g, torch.device("cpu")).clamp(max=127).to(data["train"][0].dtype)
    data["quiet"] = (_xq.to(data["train"][0].device), torch.full((200,), NBINS, device=device),
                     torch.full((200,), NEG, device=device))
    sess_of["quiet"] = torch.zeros(200, dtype=torch.long)


def resample_time(x, speed, shift):
    """Play each clip `speed` times faster, delayed by `shift` frames (both [B], may be fractional).
    Integrates the cumulative event count, so no event is invented; gaps become empty frames.
    (From training/snn/augment.py.)"""
    b, t = x.shape[:2]
    cumulative = torch.cat([torch.zeros_like(x[:, :1]), x.cumsum(dim=1)], dim=1)
    steps = torch.arange(t + 1, device=x.device, dtype=x.dtype)
    centre = t / 2
    tau = (centre + (steps[None] - centre) * speed[:, None] - shift[:, None]).clamp(0, t)
    lo = tau.floor().long().clamp(max=t - 1)
    frac = (tau - lo).view(b, t + 1, *([1] * (x.dim() - 2)))
    index = lo.view(b, t + 1, *([1] * (x.dim() - 2))).expand_as(cumulative)
    low, high = cumulative.gather(1, index), cumulative.gather(1, index + 1)
    return (low + frac * (high - low)).diff(dim=1).clamp_min(0)


def augment(xb):
    """xb: [n, W+T, 2, S, S] event-count frames (already divided by the block area), W context frames first.
    Returns the frames and which samples were mirrored left-right."""
    n = xb.shape[0]
    flipped = torch.zeros(n, dtype=torch.bool, device=device)
    if args.binom_drop:                                  # thin out the events themselves
        on = torch.rand(n, device=device) < args.binom_drop_p
        if on.any():
            keep = 1 - args.binom_drop * torch.rand(int(on.sum()), 1, 1, 1, 1, device=device)
            counts = (xb[on] * DENOM).round()
            xb[on] = torch.binomial(counts, keep.expand_as(counts).contiguous()) / DENOM
    if args.time_shift or args.speed:                    # move / stretch the clip in time, context untouched
        speed, shift = torch.ones(n, device=device), torch.zeros(n, device=device)
        if args.speed:
            lo, hi = (float(v) for v in args.speed.split(","))
            on = torch.rand(n, device=device) < args.speed_p
            speed = torch.where(on, lo + (hi - lo) * torch.rand(n, device=device), speed)
        if args.time_shift:
            on = torch.rand(n, device=device) < args.time_shift_p
            m = args.time_shift
            shift = torch.where(on, torch.randint(-m, m + 1, (n,), device=device).float(), shift)
        moved = (speed != 1) | (shift != 0)
        if moved.any():
            clip = xb[:, W:]
            clip[moved] = resample_time(clip[moved], speed[moved], shift[moved])
    if args.event_drop:
        p = torch.rand(n, 1, 1, 1, 1, device=device) * args.event_drop
        xb = xb * (torch.rand_like(xb) >= p)
    if args.noise:
        q = torch.rand(n, 1, 1, 1, 1, device=device) * args.noise
        xb = xb + (torch.rand_like(xb) < q).float() / DENOM
    if args.aug == "strong":
        zoom = 1 + args.zoom * (2 * torch.rand(n, device=device) - 1)
        rot = math.radians(args.rot) * (2 * torch.rand(n, device=device) - 1)
        flip = (torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0) if args.flip
                else torch.ones(n, device=device))
        flipped = flip < 0
        shift = (2 * torch.rand(n, 2, device=device) - 1) * (2.0 * MAX_SHIFT / SIZE)
        cos, sin = torch.cos(rot) / zoom, torch.sin(rot) / zoom
        theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], 1),
                             torch.stack([sin * flip, cos, shift[:, 1]], 1)], 1)
        tt = xb.shape[1]
        grid = F.affine_grid(theta, (n, tt * 2, SIZE, SIZE), align_corners=False)
        xb = F.grid_sample(xb.reshape(n, tt * 2, SIZE, SIZE), grid, mode="nearest",
                           padding_mode="zeros", align_corners=False).reshape(n, tt, 2, SIZE, SIZE)
    else:
        if MAX_SHIFT:
            padded = F.pad(xb, (MAX_SHIFT,) * 4)
            off = torch.randint(0, 2 * MAX_SHIFT + 1, (n, 2))
            xb = torch.stack([padded[i, ..., a:a + SIZE, b:b + SIZE] for i, (a, b) in enumerate(off.tolist())])
        if args.flip:
            flipped = torch.rand(n, device=device) < 0.5
            xb[flipped] = xb[flipped].flip(-1)
    if args.polarity_p:                                  # swap ON / OFF
        on = torch.rand(n, device=device) < args.polarity_p
        xb = torch.where(on[:, None, None, None, None], xb.flip(2), xb)
    if args.rate_jitter:
        xb = xb * (1 + args.rate_jitter * (2 * torch.rand(n, 1, 1, 1, 1, device=device) - 1))
    return xb, flipped


MIRROR = torch.arange(len(CLASS_NAMES), device=device)
for pair in [v for v in args.mirror.split(",") if v]:
    a, b = (int(v) for v in pair.split(":"))
    MIRROR[a], MIRROR[b] = b, a


def context(split, idx, train, mode):
    """[n, W, 2, S, S]: what the camera saw before the clip. 'blank' = empty frames; 'clip' = the
    last W frames of another clip (random in training, a fixed partner in evaluation)."""
    X, L, Y = data[split]
    n, N = len(idx), len(Y)
    if mode == "blank":
        return torch.zeros(n, W, 2, SIZE, SIZE, device=device)
    j = torch.randint(0, N, (n,), device=device) if train else (idx * 7919 + 13) % N
    t_idx = torch.arange(T - W, T, device=device)[None, :].expand(n, W)
    ctx = X[j.to(X.device)[:, None], t_idx.to(X.device)].to(device).float() / DENOM
    if train and args.ctx_blank > 0:
        ctx = ctx * (torch.rand(n, 1, 1, 1, 1, device=device) >= args.ctx_blank)
    return ctx


def make_batch(split, idx, train, start_bin=0, ctx="blank"):
    X, L, Y = data[split]
    n = len(idx)
    if train and args.crop_max_s > 0:
        max_start = (L[idx] - T).clamp(min=0).clamp(max=int(args.crop_max_s * 100))
        start = (torch.rand(n, device=device) * (max_start + 1)).floor().long()
    else:
        start = torch.full((n,), start_bin, dtype=torch.long, device=device)
    t_idx = (start[:, None] + torch.arange(T, device=device)[None, :]).clamp(max=NBINS - 1)
    xb = X[idx.to(X.device)[:, None], t_idx.to(X.device)].to(device).float() / DENOM  # [n,T,2,S,S]
    if W:
        xb = torch.cat([context(split, idx, train, "clip" if train else ctx), xb], dim=1)   # [n,W+T,2,S,S]
    yb = Y[idx]
    if train:
        xb, flipped = augment(xb)
        yb = torch.where(flipped, MIRROR[yb], yb)        # a mirrored left gesture is a right gesture
        if args.quiet_prob > 0:                          # some samples become a quiet scene = no gesture
            q = torch.rand(n, device=device) < args.quiet_prob
            if q.any():
                xb[q] = quiet_counts(int(q.sum()), xb.shape[1]) / DENOM
                yb = torch.where(q, torch.full_like(yb, NEG), yb)
    return xb, yb


# ------------------------------------------------------------------ model
DROPOUT_LAYERS = [int(v) for v in args.dropout_layers.split(",") if v]


class GestureSNN(nn.Module):
    def __init__(self, dropout=True):
        """dropout=False: the plain architecture (model.json). The state dict is the same either way."""
        super().__init__()
        lif = {"initial_tau": args.initial_tau, "learnable_leak": bool(args.learnable_leak),
               "v_threshold": 1.0, "reset_mode": "HARD"}

        def layer(i, ci, co, k, s, p):
            kw = dict(kernel_size=k, stride=s, padding=p, bias=False, first_layer=(i == 0), **lif)
            if dropout and args.layer_dropout and i in DROPOUT_LAYERS:
                return SpikingConvBNDropout2d(ci, co, dropout_p=args.layer_dropout, **kw)
            return SpikingConvBN2d(ci, co, **kw)
        self.net = nn.Sequential(*[layer(i, *spec) for i, spec in enumerate(LAYERS)])

    def forward(self, x, with_stats=False, skip=0):
        """Returns spike counts per class over the frames after `skip` (the context frames only
        set the state); with_stats also the active inputs of every layer (per sample, summed over
        the counted frames) and the number of neurons per layer."""
        active = [(x[:, skip:] > 0).flatten(1).sum(1).float()]
        neurons = []
        for layer in self.net:
            x = layer(x)
            active.append(x[:, skip:].flatten(1).sum(1))
            neurons.append(x[0, 0].numel())
        counts = x[:, skip:].sum(dim=1).flatten(start_dim=1)
        return (counts, active, neurons) if with_stats else counts


def synops_per_step(active):
    """Synaptic operations per 10 ms step and sample: active inputs x fan-out, all layers."""
    return sum(a * f for a, f in zip(active[:-1], FANOUT)) / T


def decide(counts, steps):
    """Spike counts the decision is taken on. With --silent-noaction the last output is replaced by
    a fixed firing-rate threshold: 'no gesture' unless a gesture output fires above it."""
    if not args.silent_noaction:
        return counts
    out = counts.clone()
    out[:, NEG] = args.silent_thr * steps
    return out


def loss_fn(counts, labels, active=None):
    targets = torch.full_like(counts, int(T * args.off_target))
    if UNUSED:
        targets[:, UNUSED] = 0
    if args.silent_noaction:                             # no-gesture clips: every output stays quiet
        targets[:, NEG] = 0
        on = labels != NEG
        targets[torch.arange(len(labels), device=labels.device)[on], labels[on]] = int(T * args.on_target)
    else:
        targets[torch.arange(len(labels)), labels] = int(T * args.on_target)
    w = torch.where(labels == len(CLASS_NAMES) - 1, args.noaction_weight, 1.0)
    w = w / w.mean()
    loss = (w * (counts - targets).square().mean(dim=1)).mean() / T
    if args.ce_weight:
        ce = F.cross_entropy(decide(counts, T) / T * args.ce_scale, labels, reduction="none")
        loss = loss + args.ce_weight * (w * ce).mean()
    if args.spike_reg and active is not None:
        hidden = sum(a * f for a, f in zip(active[1:-1], FANOUT[1:])) / T   # trainable part only
        loss = loss + args.spike_reg * hidden.mean() / 1e6
    return loss


def clamp_weights(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "leak" not in name:
                p.clamp_(-1.0, 1.0)


def ema_update_fn(decay):
    """EMA update with warm-up: follows the model closely at first, then settles at `decay`."""
    @torch.no_grad()
    def update(ema_params, model_params, num_averaged):
        n = float(num_averaged)
        d = min(decay, (1.0 + n) / (10.0 + n))
        for e, m in zip(ema_params, model_params):
            if e.is_floating_point():
                e.lerp_(m, 1.0 - d)
            else:
                e.copy_(m)
    return update


TRIG_RATE, TRIG_MARGIN = 0.5, 0.2   # decision rule of the robot bridge


@torch.no_grad()
def evaluate(model, split, start_bin=0, steps=None, ctx="blank"):
    """Accuracy, loss, confusion, SynOps/step, firing rate per layer and robot-style trigger
    statistics. steps < T evaluates only the first part of every clip (early decision)."""
    steps = steps or T
    model.eval()
    X, L, Y = data[split]
    conf = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.long)
    total_loss, syn, rate_sum, neurons, preds, all_counts = 0.0, 0.0, None, None, [], []
    trig = torch.zeros(4)   # gesture clips: correct / wrong / missed trigger; no-action clips: false trigger
    for i in range(0, len(Y), 32):
        idx = torch.arange(i, min(i + 32, len(Y)), device=device)
        xb, yb = make_batch(split, idx, train=False, start_bin=start_bin, ctx=ctx)
        counts, active, neurons = model(xb[:, :W + steps], with_stats=True, skip=W)
        total_loss += loss_fn(counts, yb).item() * len(yb) if steps == T else 0.0
        g = counts[:, :-1] / steps                       # firing rate of the gesture outputs
        top2 = g.topk(2, dim=1)
        fire = (top2.values[:, 0] >= TRIG_RATE) & (top2.values[:, 0] - top2.values[:, 1] >= TRIG_MARGIN)
        gest = yb < len(CLASS_NAMES) - 1
        hit = top2.indices[:, 0] == yb
        trig += torch.stack([(gest & fire & hit).sum(), (gest & fire & ~hit).sum(),
                             (gest & ~fire).sum(), (~gest & fire).sum()]).cpu()
        syn += synops_per_step(active).sum().item()
        r = torch.stack([a.sum() for a in active[1:]])
        rate_sum = r if rate_sum is None else rate_sum + r
        preds.append(decide(counts, steps).argmax(1).cpu())
        all_counts.append(counts.cpu())
        for t, p in zip(yb.cpu(), preds[-1]):
            conf[t, p] += 1
    n = len(Y)
    rates = [float(rs) / (n * steps * nn_) for rs, nn_ in zip(rate_sum, neurons)]
    n_gest = max(1, int((Y < len(CLASS_NAMES) - 1).sum())); n_na = max(1, n - n_gest)
    trig = {"correct": trig[0].item() / n_gest, "wrong": trig[1].item() / n_gest,
            "missed": trig[2].item() / n_gest, "false": trig[3].item() / n_na}
    return {"acc": conf.diag().sum().item() / conf.sum().item(), "loss": total_loss / n,
            "conf": conf.tolist(), "synops": syn / n * T / steps, "rates": rates, "pred": torch.cat(preds),
            "trig": trig, "counts": torch.cat(all_counts)}


def fixed_point_eval(model, bits, split, start_bin=0, steps=None, ctx="blank"):
    return evaluate(prepare_ptq(model, QCONF[bits]).to(device), split, start_bin, steps, ctx)


def train_phase(model, epochs, lr, tag, select_fn):
    """Train; keep the weights (live or EMA) with the best validation accuracy."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=args.weight_decay)
    n_train = len(data["train"][2])
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total = epochs * steps_per_epoch
    warm = min(args.warmup, epochs) * steps_per_epoch if tag == "float" and not args.init else 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    # EMA of the weights with warm-up; BatchNorm statistics stay in sync with the live model
    # (use_buffers=False) - averaging them from their defaults breaks a spiking net.
    ema = AveragedModel(model, multi_avg_fn=ema_update_fn(args.ema), use_buffers=False) if args.ema else None
    best, best_state, hist = (-1.0, -math.inf), None, []

    def rank(sc):
        return (round(sc["acc"], 4), -(sc["loss"] + args.spike_reg * sc["synops"] / 1e6))
    for ep in range(1, epochs + 1):
        model.train()
        t0, tl, tc, tn, ts = time.time(), 0.0, 0, 0, 0.0
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = make_batch("train", idx, train=True)
            mix = None
            if args.mixup_p and torch.rand(()) < args.mixup_p:    # blend two clips of the batch
                lam = float(torch.distributions.Beta(args.mixup_alpha, args.mixup_alpha).sample())
                mix_perm = torch.randperm(len(yb), device=device)   # (not 'perm': that is the epoch order)
                xb = lam * xb + (1 - lam) * xb[mix_perm]
                mix = (mix_perm, lam)
            counts, active, _ = model(xb, with_stats=True, skip=W)
            loss = loss_fn(counts, yb, active)
            if mix is not None:                                  # = lam * L(y) + (1 - lam) * L(y[perm]) + spike penalty
                loss = loss + (1 - mix[1]) * (loss_fn(counts, yb[mix[0]]) - loss_fn(counts, yb))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            clamp_weights(model)
            if ema is not None:
                ema.update_parameters(model)
            tl += loss.item() * len(yb); tc += (decide(counts, T).argmax(1) == yb).sum().item(); tn += len(yb)
            ts += synops_per_step(active).sum().item()
        cands = {"live": model}
        if ema is not None:
            cands["ema"] = ema.module
        scores = {name: select_fn(m) for name, m in cands.items()}
        pick = max(scores, key=lambda name: rank(scores[name]))
        score = scores[pick]["acc"]
        rec = {"phase": tag, "epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn,
               "train_synops": ts / tn, "val_acc": score, "val_loss": scores[pick]["loss"],
               "val_synops": scores[pick]["synops"], "val_pick": pick,
               "val_acc_live": scores["live"]["acc"], "val_acc_ema": scores.get("ema", {}).get("acc"),
               "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        mark = ""
        if rank(scores[pick]) > best:
            best, mark = rank(scores[pick]), " *"
            best_state = {k: v.detach().clone() for k, v in cands[pick].state_dict().items()}
        both = " ".join(f"{name} {s['acc']:.1%}" for name, s in scores.items())
        both += f" loss {scores[pick]['loss']:.3f}"
        print(f"[{tag}] epoch {ep}/{epochs} train {tl / tn:.4f} / {tc / tn:.1%} | val {both}"
              f" | synops/step {scores[pick]['synops'] / 1e3:.0f}k | lr {rec['lr']:.2e} | {rec['sec']:.0f}s{mark}",
              flush=True)
    model.load_state_dict(best_state)
    return model, best, hist


# ------------------------------------------------------------------ run
model = GestureSNN().to(device)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    print("initialised from", args.init, flush=True)
n_params = sum(p.numel() for n_, p in model.named_parameters() if "bn" not in n_)
print(f"model {args.arch}: {n_params} weights, fan-out per layer {[round(f) for f in FANOUT]}, "
      f"train clips {len(data['train'][2])}", flush=True)

HALF = T // 2  # early decision: accuracy from the first 0.5 s only


def trigger_stats(counts, y, rate_thr, margin=TRIG_MARGIN):
    """Robot rule on spike counts of a 1 s clip: a gesture output must reach rate_thr and lead by margin."""
    g = counts[:, :-1] / T
    top2 = g.topk(2, dim=1)
    fire = (top2.values[:, 0] >= rate_thr) & (top2.values[:, 0] - top2.values[:, 1] >= margin)
    gest, hit = y < len(CLASS_NAMES) - 1, top2.indices[:, 0] == y
    ng, nn = max(1, int(gest.sum())), max(1, int((~gest).sum()))
    return {"correct": int((gest & fire & hit).sum()) / ng, "wrong": int((gest & fire & ~hit).sum()) / ng,
            "missed": int((gest & ~fire).sum()) / ng, "false": int((~gest & fire).sum()) / nn}


def tuned_threshold(counts, y):
    """Lowest firing-rate threshold with false triggers <= --max-false on this split."""
    for thr in [x / 100 for x in range(30, 100, 2)]:
        if trigger_stats(counts, y, thr)["false"] <= args.max_false:
            return thr
    return 0.98


def by_session(pred, split):
    y, sess = data[split][2].cpu(), sess_of[split]
    return {f"{k:02d}": float((pred[sess == k] == y[sess == k]).float().mean()) for k in sorted(set(sess.tolist()))}


SAVED = {}


def full_eval(m, bits=None, name=None):
    ev = ((lambda split, steps=None, ctx="blank": fixed_point_eval(m, bits, split, 0, steps, ctx)) if bits
          else (lambda split, steps=None, ctx="blank": evaluate(m, split, 0, steps, ctx)))
    val, test, half = ev("val"), ev("test"), ev("test", HALF)
    train_acc = ev("train")["acc"]                       # training clips without augmentation
    neg = None
    quiet = None
    if "quiet" in data:                                  # synthetic quiet scenes: nothing may trigger
        qv = ev("quiet")
        quiet = {"as_no_gesture": qv["acc"], "false": qv["trig"]["false"],
                 "rates": [round(float(v), 3) for v in (qv["counts"].float().mean(0) / T)]}
    if "neg" in data:                                    # foreign motion: nothing may trigger
        nv = ev("neg")
    test_ctx = ev("test", None, "clip")["acc"] if W else None
    yv, yt = data["val"][2].cpu(), data["test"][2].cpu()
    thr = tuned_threshold(val["counts"], yv)
    if "neg" in data:
        yn = data["neg"][2].cpu()
        neg = {"n": len(yn), "as_no_gesture": nv["acc"], "false": trigger_stats(nv["counts"], yn, TRIG_RATE)["false"],
               "false_tuned": trigger_stats(nv["counts"], yn, thr)["false"]}
    SAVED[name or ("fixed" if bits else "float")] = {"val": val["counts"].numpy(), "test": test["counts"].numpy()}
    return {"train": train_acc, "neg": neg, "quiet": quiet, "val_by_session": by_session(val["pred"], "val"),
            "val": val["acc"], "test": test["acc"], "test_half": half["acc"], "test_ctx": test_ctx, "trig": test["trig"],
            "tuned_thr": thr, "trig_tuned": trigger_stats(test["counts"], yt, thr),
            "val_trig_tuned": trigger_stats(val["counts"], yv, thr),
            "synops": test["synops"], "rates": test["rates"], "test_confusion": test["conf"],
            "test_by_session": by_session(test["pred"], "test")}


hist = []
res = {}
if args.epochs > 0:
    model, _, h = train_phase(model, args.epochs, args.lr, "float", lambda m: evaluate(m, "val"))
    hist += h
torch.save(model.state_dict(), os.path.join(args.out, "float_best.pt"))
res["float"] = full_eval(model, name="float")
for bits in QCONF:
    res[f"float_ptq_q{bits}"] = full_eval(model, bits, name=f"float_ptq_q{bits}")
print(" | ".join(f"{k}: val {v['val']:.1%} test {v['test']:.1%}" for k, v in res.items()), flush=True)

qat_models = {}
for bits in QAT_BITS:
    if args.qat_epochs <= 0:
        break
    qat = prepare_qat(model, QCONF[bits], fused_bn=True).to(device)
    qat, _, h = train_phase(qat, args.qat_epochs, args.lr * args.qat_lr_scale, f"qat{bits}",
                            lambda m, b=bits: fixed_point_eval(m, b, "val"))
    hist += h
    torch.save(qat.state_dict(), os.path.join(args.out, f"qat{bits}_best.pt"))
    res[f"qat_q{bits}"] = full_eval(qat, bits, name=f"qat_q{bits}")
    res[f"qat_q{bits}_float"] = full_eval(qat, name=f"qat_q{bits}_float")   # same weights, float arithmetic
    qat_models[bits] = qat

# ------------------------------------------------------------------ export
from riverops.export import export_for_river

example = make_batch("test", torch.arange(1, device=device), train=False)[0]
with open(os.path.join(args.out, "model.json"), "w") as f:
    f.write(export_for_river(GestureSNN(dropout=False).to(device).eval().net, example))
for bits in QCONF:
    src = qat_models.get(bits, model)
    m = prepare_qat(src, QCONF[bits]).cpu().eval()
    weights, biases, neurons = {}, {}, {}
    for i, layer in enumerate(m.net):
        p = layer.fuse().export_params()                 # (the dropout layer type has no export_params)
        weights[f"weights_{i}"], biases[f"bias_{i}"] = p["weights"], p["bias"]
        neurons[f"threshold_{i}"], neurons[f"leak_{i}"] = p["threshold"], p["leak"]
    params = {**weights, **biases, **neurons}
    with open(os.path.join(args.out, f"weights_q{bits}.json"), "w") as f:
        json.dump(params, f)
    vals = [v for x in params.values() for v in (x if isinstance(x, list) else [x])]
    with open(os.path.join(args.out, f"weights_q{bits}.txt"), "w") as f:
        f.write("\n".join(map(str, vals)))
    nz = sum(1 for v in vals if v != 0)
    print(f"exported weights_q{bits}.txt ({'QAT' if bits in qat_models else 'PTQ'}): {len(vals)} values, "
          f"{nz} non-zero, range [{min(vals)}, {max(vals)}]", flush=True)

deploy = ({k: v for k, v in res.items() if k.startswith("qat_q") and not k.endswith("_float")}
          or {k: v for k, v in res.items() if "ptq" in k})
best_key = max(deploy, key=lambda k: deploy[k]["val"])
json.dump({"args": vars(args), "arch": {"name": args.arch, "size": SIZE, "layers": LAYERS, "weights": n_params,
                                         "fanout": FANOUT},
           "classes": CLASS_NAMES, "sessions": {"train": train_s, "val": val_s, "test": test_s},
           "n": {k: len(v[2]) for k, v in data.items()}, "results": res, "deployable_pick": best_key,
           "history": hist}, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

np.savez_compressed(os.path.join(args.out, "outputs.npz"), y_val=data["val"][2].cpu().numpy(),
                    y_test=data["test"][2].cpu().numpy(), s_test=sess_of["test"].numpy(),
                    **{f"{k}__{sp}": v[sp] for k, v in SAVED.items() for sp in ("val", "test")})
print("\n=== RESULTS ===")
print(f"{'model':16s} {'train':>6s} {'val':>6s} {'test':>6s} {'ctx':>6s} {'0.5s':>6s} | rule 0.5: {'ok':>6s} {'wrong':>6s} {'miss':>6s} {'false':>6s} |"
      f" tuned {'thr':>4s} {'ok':>6s} {'wrong':>6s} {'false':>6s} | {'SynOps/step':>11s}  spike rate per layer (%)")
for k, v in res.items():
    tr, tt = v["trig"], v["trig_tuned"]
    ctx_acc = "   -  " if v["test_ctx"] is None else f"{v['test_ctx']:6.1%}"
    print(f"{k:16s} {v['train']:6.1%} {v['val']:6.1%} {v['test']:6.1%} {ctx_acc} {v['test_half']:6.1%} |           {tr['correct']:6.1%} {tr['wrong']:6.1%} "
          f"{tr['missed']:6.1%} {tr['false']:6.1%} |       {v['tuned_thr']:4.2f} {tt['correct']:6.1%} {tt['wrong']:6.1%} {tt['false']:6.1%} |"
          f" {v['synops'] / 1e3:10.0f}k  " + " ".join(f"{100 * r:.2f}" for r in v["rates"]))
print("tuned = firing-rate threshold chosen on the validation session for <= "
      f"{args.max_false:.0%} false triggers, applied to the test set")
print("trigger = robot rule on the test set: gesture clips triggered correctly / wrongly / not at all; "
      "false = No-action clips that trigger a gesture")
print(f"weights: {n_params} | deployable pick (by val): {best_key} ->",
      "weights_" + best_key.split("_")[-1] + ".txt",
      "(firmware exists: best80 = firmware (3) / shipped, nb40 = 8-bit firmware seen on the board)"
      if args.arch in ("best80", "nb40") else "(needs a new firmware from model.json)")
if "neg" in data:
    ng = res[best_key]["neg"]
    print(f"foreign motion ({ng['n']} clips of --neg-zip, sessions {args.neg_eval}): {ng['as_no_gesture']:.1%} classified as "
          f"'{CLASS_NAMES[-1]}'; false triggers {ng['false']:.1%} (rule 0.5), {ng['false_tuned']:.1%} (tuned threshold)")
if "quiet" in data:
    qt = res[best_key]["quiet"]
    print(f"quiet scene (200 synthetic clips): false triggers {qt['false']:.1%} (rule 0.5); firing per output {qt['rates']}")
print("test accuracy per session:", "  ".join(f"{k}: {v:.1%}" for k, v in res[best_key]["test_by_session"].items()))
print("test confusion of", best_key, "(rows true, cols predicted):", CLASS_NAMES)
print(np.array(res[best_key]["test_confusion"]))
