"""Offline follow-up to concat_check.py: more LIF/input variants against the saved FPGA outputs
(runs/fpga_debug/concat_fpga.npz), no board access needed."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["."]
from riverops.nn import prepare_ptq
from riverops.nn.spiking_conv import SpikingConv2d
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts
from training.snn.train import load_qat_model

run = Path("runs/motion-buddy-40-v1"); cfg = load_config(run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
saved = np.load("runs/fpga_debug/concat_fpga.npz"); F = saved["F"]; files = list(saved["files"])
seq = np.concatenate([events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
                      for f in files])
N = len(files)
qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()
for i, layer in enumerate(emu.net):
    layer.layer_index = i
P = formats.potential; LSB = 1 / P.scale
leak = round(P.scale / cfg.model.neuron.initial_tau) / P.scale
original = SpikingConv2d._fixed_point_neuron_step

print(f"input values: max {seq.max():.3f} (Q4.12 max {P.max_value:.3f}); "
      f"frames with an input pixel > Q4.12 max: {int((seq.reshape(len(seq), -1).max(1) > P.max_value).sum())}/{len(seq)}")

def q(x, rnd=torch.round):
    return rnd(x * P.scale).clamp(P.min_int, P.max_int) / P.scale

def step_factory(order):
    def step(self, current, v, threshold):
        current = q(current)
        if v is None:
            v = torch.zeros_like(current)
        if order == "leak then add":                      # riverops order
            updated = q(q(v - q(v * leak)) + current)
        elif order == "add then leak":                    # leak applied to v + input
            s = q(v + current); updated = q(s - q(s * leak))
        spikes = (updated >= threshold).to(updated.dtype)
        return spikes, q((1 - spikes) * updated)
    return step

def simulate(step, x):
    SpikingConv2d._fixed_point_neuron_step = step
    try:
        with torch.no_grad():
            return emu.net(torch.from_numpy(x)[None])[0, :, :, 0, 0].numpy().astype(int)
    finally:
        SpikingConv2d._fixed_point_neuron_step = original

def compare(a, b):
    first = next((t for t in range(len(a)) if not np.array_equal(a[t], b[t])), None)
    pa = a.reshape(N, T, -1).sum(1).argmax(-1); pb = b.reshape(N, T, -1).sum(1).argmax(-1)
    return f"{np.mean(a == b):7.2%}  first diff {str(first):>5s}  same prediction {int((pa == pb).sum())}/{N}"

clamped = np.minimum(seq, P.max_value).astype(np.float32)
inputs_q = (np.round(seq * P.scale).clip(P.min_int, P.max_int) / P.scale).astype(np.float32)
for name, step, x in [
    ("leak then add (riverops), raw input",           step_factory("leak then add"), seq),
    ("leak then add, input clamped to Q4.12",          step_factory("leak then add"), clamped),
    ("add then leak, raw input",                       step_factory("add then leak"), seq),
    ("add then leak, input clamped to Q4.12",          step_factory("add then leak"), clamped),
]:
    print(f"  {name:42s} {compare(F, simulate(step, x))}")

print("\nframes 0-5, output spikes (rows = classes):")
E = simulate(step_factory("leak then add"), seq)
print("FPGA\n", F[:6].T, "\nsimulation\n", E[:6].T)

# --- Single rounding at the end of the update (wide accumulator), with different rounding modes ------
half_up = lambda x: torch.floor(x + 0.5)  # noqa: E731
def single_round(rnd, leak_value):
    def step(self, current, v, threshold):
        current = q(current)
        if v is None:
            v = torch.zeros_like(current)
        updated = q(v - v * leak_value + current, rnd)      # one rounding for the whole update
        spikes = (updated >= threshold).to(updated.dtype)
        return spikes, q((1 - spikes) * updated)
    return step
print("\nsingle rounding of v - v*leak + I:")
for leak_name, leak_value in (("leak 287/4096", leak), ("leak 0.07", 1 / cfg.model.neuron.initial_tau)):
    for rnd_name, rnd in (("nearest", torch.round), ("half up", half_up), ("floor", torch.floor), ("truncate", torch.trunc)):
        print(f"  {leak_name:14s} {rnd_name:9s} {compare(F, simulate(single_round(rnd, leak_value), seq))}")
