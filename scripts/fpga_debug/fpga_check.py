"""Stream test clips to the Kria and compare with riverops fixed-point emulation."""
import sys, time, random
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts, NAME_RE
from training.snn.train import load_qat_model

run = Path("runs/motion-buddy-40-v1")
cfg = load_config(run / "config.yaml")
qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling

files = sorted(Path(cfg.data.root, "13").glob("gesture_*.bin"))
random.Random(0).shuffle(files)
picked, seen = [], set()
for f in files:                      # one clip per class
    name = NAME_RE.match(f.name).group(2)
    if name not in seen:
        seen.add(name); picked.append(f)

client = FPGAClient("durance-kria.local", 3333, shape_in=(2, S, S), shape_out=(cfg.n_classes, 1, 1))
client.connect()
zero = np.zeros((2, S, S), np.float32)
latencies = []
def settle(n=150):                   # let membrane potentials decay to ~0 between clips (0.93^150 ≈ 2e-5)
    for _ in range(n):
        client.send_tensor(zero); client.receive_tensor()
settle()
print(f"{'clip':40s} {'true':10s} {'FPGA':10s} {'emulation':10s} FPGA spike counts | emulation spike counts")
for f in picked:
    frames = (events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2)
    outs = []
    for t in range(T):
        client.send_tensor(frames[t])
        out, lat = client.receive_tensor()
        outs.append(out.ravel()); latencies.append(lat)
    fpga = np.array(outs)                                   # (T, 5)
    with torch.no_grad():
        spikes = emu.net(torch.from_numpy(frames)[None])    # (1, T, 5, 1, 1)
    e = spikes[0, :, :, 0, 0].numpy()
    true = NAME_RE.match(f.name).group(2)
    cls = cfg.data.classes
    print(f"{f.name:40s} {true:10s} {cls[fpga.sum(0).argmax()]:10s} {cls[e.sum(0).argmax()]:10s} "
          f"{np.round(fpga.sum(0), 1)} | {e.sum(0)}   per-step match {np.mean(np.isclose(fpga, e)):.0%}")
    settle()
client.close()
print("distinct per-frame output values from FPGA (first clip):", np.unique(np.array(outs))[:10])
print(f"latency field: median {np.median(latencies)}, max {np.max(latencies)} (units as reported by server)")
