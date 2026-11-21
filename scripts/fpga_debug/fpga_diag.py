"""Record FPGA outputs per clip (fresh connection each), then test time shifts and reset behaviour offline."""
import sys, random
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts
from training.snn.train import load_qat_model

run = Path("runs/motion-buddy-40-v1")
cfg = load_config(run / "config.yaml")
qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
files = sorted(Path(cfg.data.root, "13").glob("gesture_*.bin")); random.Random(1).shuffle(files)

def stream(frames):
    c = FPGAClient("durance-kria.local", 3333, shape_in=(2, S, S), shape_out=(cfg.n_classes, 1, 1)); c.connect()
    out = []
    for f in frames:
        c.send_tensor(f); out.append(c.receive_tensor()[0].ravel())
    c.close()
    return np.array(out)

F, E, X = [], [], []
for f in files[:10]:
    frames = events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
    X.append(frames); F.append(stream(frames))
    with torch.no_grad():
        E.append(emu.net(torch.from_numpy(frames)[None])[0, :, :, 0, 0].numpy())
F, E = np.array(F), np.array(E)                  # (clips, T, 5)

# 1. Does a fresh connection reset? Send the same clip twice on two connections: identical outputs → reset.
again = stream(X[0])
print("same clip, two fresh connections → identical FPGA outputs:", np.array_equal(again, F[0]))

# 2. Time shift: FPGA[t + k] vs emulation[t]
for k in range(0, 8):
    m = np.mean(F[:, k:, :] == E[:, :T - k, :])
    print(f"  FPGA lag {k} steps: identical spikes {m:.1%}")

# 3. When do they first differ?
first = [next((t for t in range(T) if not np.array_equal(F[i, t], E[i, t])), None) for i in range(len(F))]
print("first differing step per clip:", first)
print("total spikes FPGA vs emulation:", int(F.sum()), int(E.sum()))
print("first 12 steps, clip 0, FPGA:\n", F[0, :12].astype(int).T, "\nemulation:\n", E[0, :12].astype(int).T)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/fpga_debug"); OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "fpga_diag.npz", F=F, E=E)
print("saved", OUT / "fpga_diag.npz")
