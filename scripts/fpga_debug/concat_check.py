"""Is the remaining FPGA/simulation mismatch arithmetic? One restart, many clips back-to-back.

Both sides start from zero potentials (restart / fresh simulation) and see the same concatenated input,
so any difference must come from how they compute. We
  1. compare the FPGA with several fixed-point variants of the LIF update (leak constant and rounding),
  2. check how close to the threshold the potentials are where the two first diverge,
  3. measure how far the simulation drifts from itself after a single 1-LSB nudge (sensitivity).
"""
import random, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from riverops.nn.spiking_conv import SpikingConv2d
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts
from training.snn.train import load_qat_model

KRIA, N_CLIPS = "root@durance-kria.local", 20
run = Path("runs/motion-buddy-40-v1"); cfg = load_config(run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()
for i, layer in enumerate(emu.net):
    layer.layer_index = i
P = formats.potential                     # Q4.12
LSB = 1.0 / P.scale

files = sorted(Path(cfg.data.root, "13").glob("gesture_*.bin")); random.Random(1).shuffle(files)
clips = [events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
         for f in files[:N_CLIPS]]
seq = np.concatenate(clips)               # (N_CLIPS * T, 2, S, S)

# --- FPGA: one restart, then the whole sequence on one connection --------------------------------
subprocess.run(["ssh", "-o", "BatchMode=yes", KRIA,
                "systemctl reset-failed spleat-server.service; systemctl restart spleat-server.service"], check=True)
for _ in range(50):
    try:
        client = FPGAClient("durance-kria.local", 3333, (2, S, S), (cfg.n_classes, 1, 1)); client.connect(); break
    except OSError:
        time.sleep(0.2)
out = []
for frame in seq:
    client.send_tensor(frame); out.append(client.receive_tensor()[0].ravel())
client.close()
F = np.array(out).astype(int)
OUT = Path("runs/fpga_debug"); OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "concat_fpga.npz", F=F, files=[str(f) for f in files[:N_CLIPS]])
print(f"FPGA: {len(seq)} frames ({N_CLIPS} clips back-to-back) after one restart")

# --- Simulation variants of the LIF step ----------------------------------------------------------
original_step = SpikingConv2d._fixed_point_neuron_step
leak_q = round(formats.potential.scale / cfg.model.neuron.initial_tau) / P.scale   # 287/4096, as exported
half_up = lambda x: torch.floor(x + 0.5)  # noqa: E731

def make_step(leak, rnd, form, record=None, nudge=None, strict=False):
    def q(x):
        return rnd(x * P.scale).clamp(P.min_int, P.max_int) / P.scale
    def step(self, current, v, threshold):
        current = q(current)
        if v is None:
            v = torch.zeros_like(current)
            if nudge is not None and self.layer_index == 0:
                v = v.clone(); v.view(-1)[0] += nudge          # one LSB on one neuron at frame 0
        v = q(v)
        decayed = q(v - q(v * leak)) if form == "subtract" else q(v * (1 - leak))
        updated = q(decayed + current)
        spikes = ((updated > threshold) if strict else (updated >= threshold)).to(updated.dtype)
        if record is not None:
            record[self.layer_index].append(((updated - threshold) / LSB).round().to(torch.int64).flatten())
        return spikes, q((1.0 - spikes) * updated)                # hard reset
    return step

def simulate(step):
    SpikingConv2d._fixed_point_neuron_step = step
    try:
        with torch.no_grad():
            return emu.net(torch.from_numpy(seq)[None])[0, :, :, 0, 0].numpy().astype(int)
    finally:
        SpikingConv2d._fixed_point_neuron_step = original_step

def compare(a, b):
    first = next((t for t in range(len(a)) if not np.array_equal(a[t], b[t])), None)
    pa = a.reshape(N_CLIPS, T, -1).sum(1).argmax(-1); pb = b.reshape(N_CLIPS, T, -1).sum(1).argmax(-1)
    return np.mean(a == b), first, int((pa == pb).sum())

variants = {
    "riverops default: v*0.07, round-nearest": (1 / cfg.model.neuron.initial_tau, torch.round, "subtract", False),
    "leak 287/4096, round-nearest":            (leak_q, torch.round, "subtract", False),
    "leak 287/4096, round half up":            (leak_q, half_up, "subtract", False),
    "leak 287/4096, floor (v*287 >> 12)":      (leak_q, torch.floor, "subtract", False),
    "leak 287/4096, truncate":                 (leak_q, torch.trunc, "subtract", False),
    "v*(4096-287)/4096, floor":                (leak_q, torch.floor, "multiply", False),
    "v*(4096-287)/4096, round-nearest":        (leak_q, torch.round, "multiply", False),
    "STRICT v > threshold, v*0.07, round-nearest":    (1 / cfg.model.neuron.initial_tau, torch.round, "subtract", True),
    "STRICT v > threshold, leak 287/4096, nearest":   (leak_q, torch.round, "subtract", True),
    "STRICT v > threshold, leak 287/4096, half up":   (leak_q, half_up, "subtract", True),
    "STRICT v > threshold, leak 287/4096, floor":     (leak_q, torch.floor, "subtract", True),
    "STRICT v > threshold, leak 287/4096, truncate":  (leak_q, torch.trunc, "subtract", True),
    "STRICT v > threshold, v*(1-leak), floor":        (leak_q, torch.floor, "multiply", True),
}
stock = simulate(original_step)
print(f"\n{'simulation variant':42s} identical spikes  first diff frame  same prediction")
results = {}
for name, (leak, rnd, form, strict) in variants.items():
    E = simulate(make_step(leak, rnd, form, strict=strict))
    results[name] = E
    same, first, pred = compare(F, E)
    print(f"  {name:40s} {same:8.2%}  {str(first):>14s}  {pred:6d}/{N_CLIPS}")
same_stock = np.array_equal(stock, results["riverops default: v*0.07, round-nearest"])
print(f"  (sanity: re-implemented default == riverops prepare_ptq: {same_stock})")

# --- 2. Near-threshold check at the first divergence (best-matching variant) -----------------------
best = max(results, key=lambda k: compare(F, results[k])[0])
leak, rnd, form, strict = variants[best]
record = {i: [] for i in range(len(emu.net))}
simulate(make_step(leak, rnd, form, record=record, strict=strict))
first = compare(F, results[best])[1]
print(f"\nbest variant: {best}")
if first is not None:
    print(f"first FPGA/simulation difference at frame {first} (clip {first // T}, frame {first % T} of that clip)")
    for i in range(len(emu.net)):
        margins = torch.stack(record[i][: first + 1])               # (frames, neurons) in LSB
        window = margins[max(0, first - 2): first + 1].abs()
        print(f"  layer {i}: closest |potential - threshold| in frames {max(0, first - 2)}..{first}: "
              f"{int(window.min())} LSB;  neuron-steps within ±1 LSB over the whole run: "
              f"{(torch.cat(record[i]).abs() <= 1).float().mean().item():.4%}")

# --- 3. Sensitivity: the simulation against itself after one 1-LSB nudge ---------------------------
nudged = simulate(make_step(leak, rnd, form, nudge=LSB, strict=strict))
same, first_n, pred = compare(results[best], nudged)
print(f"\nsimulation vs itself with ONE neuron nudged by 1 LSB at frame 0: identical spikes {same:.2%}, "
      f"first diff frame {first_n}, same prediction {pred}/{N_CLIPS}")
same, first_f, pred = compare(F, results[best])
print(f"FPGA vs best simulation:                                         identical spikes {same:.2%}, "
      f"first diff frame {first_f}, same prediction {pred}/{N_CLIPS}")
