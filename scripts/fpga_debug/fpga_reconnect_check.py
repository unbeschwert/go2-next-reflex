"""Which emulation variant matches the FPGA? as-is / biases zero / biases / 64 (read as Q4.12)."""
import sys, copy, random
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
base = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()
def variant(scale):
    m = copy.deepcopy(base)
    with torch.no_grad():
        for layer in m.net:
            layer.conv.bias.mul_(scale)
    return m
variants = {"as exported": base, "biases x0": variant(0.0), "biases /64": variant(1 / 64)}
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
files = sorted(Path(cfg.data.root, "13").glob("gesture_*.bin")); random.Random(1).shuffle(files)
files = files[:30]

zero = np.zeros((2, S, S), np.float32)
def settle(n=150):
    for _ in range(n): client.send_tensor(zero); client.receive_tensor()
match = {k: [] for k in variants}; agree = {k: 0 for k in variants}; fpga_correct = 0
for f in files:
    frames = events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
    client = FPGAClient("durance-kria.local", 3333, shape_in=(2, S, S), shape_out=(cfg.n_classes, 1, 1))
    client.connect()                 # fresh connection per clip → fresh potentials
    out = []
    for t in range(T):
        client.send_tensor(frames[t]); out.append(client.receive_tensor()[0].ravel())
    fpga = np.array(out); client.close()
    label = cfg.data.classes.index(NAME_RE.match(f.name).group(2))
    fpga_correct += fpga.sum(0).argmax() == label
    for k, m in variants.items():
        with torch.no_grad():
            e = m.net(torch.from_numpy(frames)[None])[0, :, :, 0, 0].numpy()
        match[k].append(np.mean(fpga == e)); agree[k] += fpga.sum(0).argmax() == e.sum(0).argmax()
print(f"{len(files)} clips from person 13; FPGA correct {fpga_correct}/{len(files)}")
for k in variants:
    print(f"  emulation {k:12s}: identical spikes in {np.mean(match[k]):.1%} of (step, class) slots, same prediction {agree[k]}/{len(files)}")
