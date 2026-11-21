"""Is the FPGA/emulation gap fixed-point arithmetic? Precision ladder, rounding variants, hardware noise floor."""
import sys, random, copy
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
import riverops.nn.quantization as rq
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts
from training.snn.model import build_model
from training.snn.train import load_qat_model

run = Path("runs/motion-buddy-40-v1")
cfg = load_config(run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
saved = np.load(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/fpga_debug") / "fpga_diag.npz"); F = saved["F"]          # FPGA outputs from the last run
files = sorted(Path(cfg.data.root, "13").glob("gesture_*.bin")); random.Random(1).shuffle(files)
X = [events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2 for f in files[:10]]

def spikes(net, frames):
    with torch.no_grad():
        return net(torch.from_numpy(frames)[None])[0, :, :, 0, 0].numpy()
def run_all(net):
    return np.array([spikes(net, x) for x in X])
def agree(a, b):
    return f"{np.mean(a == b):6.1%} spikes, {np.mean(a.sum(1).argmax(-1) == b.sum(1).argmax(-1)):5.0%} predictions"

qat, formats = load_qat_model(run / "qat_best.pt", cfg)
flt = build_model(cfg); flt.load_state_dict(torch.load(run / "float_best.pt", map_location="cpu", weights_only=True)["state_dict"]); flt.eval()
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()
R = {"float (float_best)": run_all(flt.net), "QAT fake-quant": run_all(qat.net), "fixed-point, round-nearest": run_all(emu.net)}

# Rounding variants: riverops rounds with torch.round inside every fixed-point resize.
original = rq.RoundSTE.forward
for name, fn in (("fixed-point, round-down (floor)", torch.floor), ("fixed-point, truncate (toward 0)", torch.trunc)):
    rq.RoundSTE.forward = staticmethod(lambda ctx, x, fn=fn: fn(x))
    R[name] = run_all(emu.net)
rq.RoundSTE.forward = original

E = R["fixed-point, round-nearest"]
print("1+2. each model vs fixed-point (round-nearest) | vs FPGA        (spike totals)")
for name, r in R.items():
    print(f"  {name:34s} {agree(r, E)} | {agree(r, F)}   ({int(r.sum())})")
print(f"  {'FPGA':34s} {agree(F, E)} |                                  ({int(F.sum())})")

# 3. Hardware noise floor: same clips, fresh connection each time, 3 repeats.
def stream(frames):
    c = FPGAClient("durance-kria.local", 3333, shape_in=(2, S, S), shape_out=(cfg.n_classes, 1, 1)); c.connect()
    out = []
    for f in frames:
        c.send_tensor(f); out.append(c.receive_tensor()[0].ravel())
    c.close(); return np.array(out)
reps = [np.array([stream(x) for x in X[:4]]) for _ in range(3)]
pairs = [(0, 1), (0, 2), (1, 2)]
print("3. FPGA vs itself (same 4 clips, 3 runs):", ", ".join(agree(reps[a], reps[b]) for a, b in pairs))
zeros = np.zeros((T, 2, S, S), np.float32)
zf, ze = stream(zeros), spikes(emu.net, zeros)
print(f"   all-zero input, 100 frames: FPGA {int(zf.sum())} output spikes, fixed-point emulation {int(ze.sum())}")
