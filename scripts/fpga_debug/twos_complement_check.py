"""Does spleat-server read negative weights written as two's complement the same as signed decimals?

weights.txt normally holds signed decimals (-110 ... 127). This writes a second file in which every negative
8-bit weight and bias is replaced by its two's complement (v + 256, so -110 → 146; thresholds and leaks are
16-bit and already positive), runs the same clips on the board with each file (service restart = fresh
potentials before every clip), and compares the raw output spikes frame by frame, the predictions, and the
fixed-point simulation. The signed weights.txt is put back at the end.

Usage: uv run python scripts/fpga_debug/twos_complement_check.py [--run runs/motion-buddy-80-v4] [--clips 10]
"""
import argparse, json, random, subprocess, sys
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts, NAME_RE
from training.snn.model import layer_shapes
from training.snn.train import load_qat_model

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run", type=Path, default=Path("runs/motion-buddy-80-v4"))
parser.add_argument("--clips", type=int, default=10, help="clips per split (val and test)")
parser.add_argument("--host", default="durance-kria.local")
args = parser.parse_args()
KRIA = f"root@{args.host}"

run, cfg = args.run, load_config(args.run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
shapes = layer_shapes(cfg)
last = shapes[-1]
bits = 8                                          # weights and biases share the 8-bit weight format
n_8bit = sum(s.weights for s in shapes) + sum(s.out_c for s in shapes)   # weights, then biases

signed = [int(v) for v in (run / "deploy" / "weights.txt").read_text().split()]
twos = [v + (1 << bits) if (i < n_8bit and v < 0) else v for i, v in enumerate(signed)]
assert all(0 <= v < 1 << bits for v in twos[:n_8bit]) and all(v >= 0 for v in twos[n_8bit:])
out = Path("runs/fpga_debug"); out.mkdir(parents=True, exist_ok=True)
files = {"signed": out / "weights_signed.txt", "twos": out / "weights_twos_complement.txt"}
files["signed"].write_text("\n".join(map(str, signed)))
files["twos"].write_text("\n".join(map(str, twos)))
print(f"{sum(v < 0 for v in signed[:n_8bit])} negative weights/biases rewritten, e.g. "
      f"{[v for v in signed[:n_8bit] if v < 0][:3]} → {[t for v, t in zip(signed, twos) if v < 0][:3]}")

def clips(people):
    c = [(f, cfg.data.classes.index(NAME_RE.match(f.name).group(2)))
         for p in people for f in sorted(Path(cfg.data.root, p).glob("gesture_*.bin"))
         if NAME_RE.match(f.name) and NAME_RE.match(f.name).group(2) in cfg.data.classes]
    random.Random(1).shuffle(c)
    return c[:args.clips]
chosen = [("val", *c) for c in clips(cfg.data.val)] + [("test", *c) for c in clips(cfg.data.test)]

def ssh(cmd):
    subprocess.run(["ssh", "-o", "BatchMode=yes", KRIA, cmd], check=True)

def run_clip(sums):
    ssh("systemctl reset-failed spleat-server.service; systemctl restart spleat-server.service")
    for _ in range(100):
        try:
            c = FPGAClient(args.host, 3333, (2, S, S), (last.out_c, last.out_h, last.out_w)); c.connect(); break
        except OSError:
            import time; time.sleep(0.1)
    else:
        raise RuntimeError("spleat-server did not come back")
    frames = []
    for frame in sums:
        c.send_tensor(frame); frames.append(c.receive_tensor()[0].ravel())
    c.close()
    return np.array(frames)

qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()
data = []
for split, f, label in chosen:
    sums = events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32)
    with torch.no_grad():
        sim = int(emu(torch.from_numpy(sums / ds ** 2)[None]).argmax())
    data.append((split, label, sums, sim))

outputs = {}
try:
    for name, path in files.items():
        subprocess.run(["scp", "-q", str(path), f"{KRIA}:/root/weights.txt"], check=True)
        outputs[name] = [run_clip(sums) for _, _, sums, _ in data]
        print(f"  ran {len(data)} clips with {name} weights", flush=True)
finally:
    subprocess.run(["scp", "-q", str(files["signed"]), f"{KRIA}:/root/weights.txt"], check=True)
    ssh("systemctl reset-failed spleat-server.service; systemctl restart spleat-server.service")

pred = lambda o: int(o.reshape(T, last.out_c, -1).sum((0, 2)).argmax())
rows = []
for k, (split, label, _, sim) in enumerate(data):
    a, b = outputs["signed"][k], outputs["twos"][k]
    rows.append({"split": split, "label": label, "sim": sim, "signed": pred(a), "twos": pred(b),
                 "identical_outputs": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(a - b).max()),
                 "spikes_signed": float(a.sum()), "spikes_twos": float(b.sum())})
n = len(rows)
same = sum(r["identical_outputs"] for r in rows)
print(f"\n{run.name}: {n} clips ({args.clips} val + {args.clips} test)")
print(f"  raw outputs identical (every frame, every class): {same}/{n}")
print(f"  same prediction signed vs two's complement: {sum(r['signed'] == r['twos'] for r in rows)}/{n}")
for name in ("signed", "twos"):
    print(f"  {name:7s}: correct {sum(r[name] == r['label'] for r in rows)}/{n}, "
          f"agrees with simulation {sum(r[name] == r['sim'] for r in rows)}/{n}, "
          f"mean output spikes per clip {np.mean([r['spikes_' + name] for r in rows]):.1f}")
print(f"  simulation: correct {sum(r['sim'] == r['label'] for r in rows)}/{n}")
(out / f"twos_complement_{run.name}.json").write_text(json.dumps(rows, indent=1))
