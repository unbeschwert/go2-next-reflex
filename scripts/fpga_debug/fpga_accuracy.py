"""FPGA accuracy on validation and test clips, with a reset (service restart) before every clip.

A restart per clip costs ~1-2 s, so this samples clips (alternating val/test, shuffled) until a time budget
runs out, and runs the fixed-point simulation on the same clips for comparison.
Usage: uv run python scripts/fpga_debug/fpga_accuracy.py [budget_seconds=200]
"""
import json, math, random, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [".", "scripts"]
from fpga import FPGAClient
from riverops.nn import prepare_ptq
from training.snn.config import load_config
from training.snn.events import read_events, events_to_counts, NAME_RE
from training.snn.train import load_qat_model

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 200
KRIA = "root@durance-kria.local"
start = time.time()
run = Path("runs/motion-buddy-40-v1"); cfg = load_config(run / "config.yaml")
T, S, ds = cfg.data.time_steps, cfg.data.size, cfg.data.downsampling
qat, formats = load_qat_model(run / "qat_best.pt", cfg)
emu = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).eval()

def clips(people):
    out = [(f, cfg.data.classes.index(NAME_RE.match(f.name).group(2)))
           for p in people for f in sorted(Path(cfg.data.root, p).glob("gesture_*.bin"))
           if NAME_RE.match(f.name) and NAME_RE.match(f.name).group(2) in cfg.data.classes]
    random.Random(0).shuffle(out); return out
queues = {"val": clips(cfg.data.val), "test": clips(cfg.data.test)}
totals = {k: len(v) for k, v in queues.items()}

def restart():
    subprocess.run(["ssh", "-o", "BatchMode=yes", KRIA,
                    "systemctl reset-failed spleat-server.service; systemctl restart spleat-server.service"], check=True)
    for _ in range(100):
        try:
            c = FPGAClient("durance-kria.local", 3333, (2, S, S), (cfg.n_classes, 1, 1)); c.connect(); return c
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("spleat-server did not come back")

results = {"val": [], "test": []}
i = 0
while time.time() - start < BUDGET:
    split = ("val", "test")[i % 2]; i += 1
    if not queues[split]:
        if not any(queues.values()): break
        continue
    f, label = queues[split].pop()
    x = events_to_counts(read_events(f), T, cfg.data.time_window_us, ds).astype(np.float32) / ds**2
    c = restart()                                        # reset: fresh potentials
    out = []
    for frame in x:
        c.send_tensor(frame); out.append(c.receive_tensor()[0].ravel())
    c.close()
    fpga_pred = int(np.array(out).sum(0).argmax())
    with torch.no_grad():
        emu_pred = int(emu.net(torch.from_numpy(x)[None]).sum(1).flatten().argmax())
    results[split].append((label, fpga_pred, emu_pred))
    if i % 10 == 0:
        print(f"  {time.time() - start:5.0f}s  clips done: val {len(results['val'])}, test {len(results['test'])}", flush=True)

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

summary = {}
print(f"\nFPGA with reset per clip ({time.time() - start:.0f}s):")
for split, r in results.items():
    n = len(r)
    if not n: continue
    fpga = sum(l == p for l, p, _ in r); emu_ok = sum(l == e for l, _, e in r); agree = sum(p == e for _, p, e in r)
    lo, hi = wilson(fpga, n)
    summary[split] = {"clips": n, "of": totals[split], "fpga_acc": fpga / n, "ci95": [lo, hi], "sim_acc": emu_ok / n, "agree": agree / n}
    print(f"  {split:4s}: FPGA {fpga}/{n} = {fpga / n:.1%} (95% CI {lo:.0%}–{hi:.0%})   "
          f"simulation on same clips {emu_ok / n:.1%}   same prediction {agree}/{n}   [{n} of {totals[split]} clips]")
out_dir = Path("runs/fpga_debug"); out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "fpga_accuracy.json").write_text(json.dumps({"summary": summary, "results": {
    k: [(int(a), int(b), int(c)) for a, b, c in v] for k, v in results.items()}}, indent=1))
