"""Accuracy per person and per class for a trained run (fixed-point model, as on the FPGA).

The val/test split is by person, so the test number depends a lot on WHICH people are in it.
Usage: uv run --group train python scripts/eval_per_person.py runs/<run> [--float]
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["."]
from riverops.nn import prepare_ptq
from training.snn import data as data_mod
from training.snn.config import load_config
from training.snn.model import build_model
from training.snn.train import SplitData, load_qat_model, pick_device

run = Path(sys.argv[1]); use_float = "--float" in sys.argv
cfg = load_config(run / "config.yaml"); dev = pick_device("auto")
splits = data_mod.discover(cfg)
cache = data_mod.build_cache(cfg, splits, workers=8)
if use_float:
    model = build_model(cfg); model.load_state_dict(torch.load(run / "float_best.pt", map_location="cpu", weights_only=True)["state_dict"])
else:
    qat, formats = load_qat_model(run / "qat_best.pt", cfg)
    model = prepare_ptq(qat, formats.riverops(qat=False), fuse=True)
model = model.to(dev).eval()
C = cfg.data.classes
print(f"{run.name} ({'float' if use_float else 'fixed point'})")
for split in ("val", "test"):
    data = SplitData(*cache[split], cfg, dev)
    files = (Path(str(cache[split][0]).replace("-x.npy", "-x-files.txt"))).read_text().split("\n")
    people = [Path(f).parent.name for f in files if f]
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), 64):
            x, _, _ = data.batch(torch.arange(i, min(i + 64, len(data)), device=dev), augment=False)
            preds.append(model(x).argmax(1).cpu())
    pred = torch.cat(preds).numpy(); y = data.y.cpu().numpy()
    print(f"\n{split}: {np.mean(pred == y):.1%} over {len(y)} clips")
    for person in sorted(set(people)):
        m = np.array([p == person for p in people])
        per_class = "  ".join(f"{c[:8]} {np.mean(pred[m & (y == k)] == k):4.0%}" for k, c in enumerate(C) if (m & (y == k)).any())
        print(f"  person {person}: {np.mean(pred[m] == y[m]):5.1%} ({m.sum()} clips)   {per_class}")
    conf = np.zeros((len(C), len(C)), int)
    for t, p in zip(y, pred): conf[t, p] += 1
    print("  confusion (rows = true, cols = predicted):", *C, sep="  ")
    for k, c in enumerate(C): print(f"    {c:10s} {conf[k]}")
