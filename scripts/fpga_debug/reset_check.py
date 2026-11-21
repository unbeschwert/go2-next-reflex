"""Does disconnect/reconnect reset the potentials? Zeros → reconnect → zeros, repeated."""
import sys, random
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts
from training.snn.train import load_qat_model

run = Path("runs/motion-buddy-40-v1"); cfg = load_config(run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()

def stream(frames):
    c = FPGAClient("durance-kria.local", 3333, shape_in=(2, S, S), shape_out=(cfg.n_classes, 1, 1)); c.connect()
    out = []
    for f in frames:
        c.send_tensor(f); out.append(c.receive_tensor()[0].ravel())
    c.close(); return np.array(out).astype(int)

zeros = np.zeros((T, 2, S, S), np.float32)
runs = [stream(zeros) for _ in range(5)]
print("zeros, 5 separate connections, output spikes per class:")
for i, r in enumerate(runs):
    print(f"  run {i}: {r.sum(0)}  identical to run 0: {np.array_equal(r, runs[0])}")
with torch.no_grad():
    ze = emu.net(torch.from_numpy(zeros)[None])[0, :, :, 0, 0].numpy().astype(int)
print(f"  emulation: {ze.sum(0)}  identical to run 0: {np.array_equal(ze, runs[0])}")
print("first 15 frames, run 0 (rows = classes):\n", runs[0][:15].T)
print("first 15 frames, emulation:\n", ze[:15].T)

# Variant: a real clip on one connection, then zeros on a new connection.
f = sorted(Path(cfg.data.root, "13").glob("gesture_*.bin"))[0]
clip = events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
stream(clip)
after = stream(zeros)
print(f"zeros after a real clip (new connection): {after.sum(0)}  identical to run 0: {np.array_equal(after, runs[0])}")
