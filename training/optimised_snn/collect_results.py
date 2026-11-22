#!/usr/bin/env python3
"""Table of all training runs:  python3 collect_results.py [runs_dir] [--sort val|test|synops] [--only PREFIX]

Reads runs/*/metrics.json written by train_snn_v2/v3/v4.py. One row per run, using the best
deployable (fixed-point) result, selected by validation accuracy.
  0.5s    accuracy using only the first 0.5 s of each clip
  ok/wr/mi/fa   robot trigger rule (rate >= 0.5) on the test set: gesture clips triggered correctly /
          wrongly / missed, and No-action clips that falsely trigger (v4+ runs)
  tuned   the same with the rate threshold chosen on validation for <= 1 % false triggers (v5 runs)
"""
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
root = Path(argv[0]) if argv and not argv[0].startswith("--") else Path("runs")
sort = argv[argv.index("--sort") + 1] if "--sort" in argv else "val"
only = argv[argv.index("--only") + 1] if "--only" in argv else ""
rows = []
for f in sorted(root.glob("*/metrics.json")):
    if only and not f.parent.name.startswith(only):
        continue
    try:
        m = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        print("skip", f, e)
        continue
    res = m.get("results")
    if not res:
        continue
    a = m.get("args", m.get("config", {}))
    deploy = {k: v for k, v in res.items() if k.startswith("qat_q")} or {k: v for k, v in res.items() if "ptq" in k}
    if not deploy:
        continue
    key = max(deploy, key=lambda k: deploy[k]["val"])
    v = deploy[key]
    fl = res.get("float", {})
    notes = []
    for name, default in (("aug", "basic"), ("flip", 1), ("learnable_leak", 0), ("spike_reg", 0.0),
                          ("noaction_weight", 1.0), ("event_drop", 0.0), ("noise", 0.0), ("epochs", 40),
                          ("lr", 3e-3), ("weight_decay", 1e-4), ("ce_weight", 0.5), ("batch_size", 16),
                          ("seed", 42), ("val", "04"), ("test", "05"), ("train", "auto")):
        if name in a and a[name] != default:
            notes.append(f"{name}={a[name]}")
    if a.get("init"):
        notes.append("init=" + Path(a["init"]).parent.name)
    rows.append({
        "run": f.parent.name, "arch": a.get("arch", "?"), "weights": m.get("arch", {}).get("weights"),
        "float_val": fl.get("val"), "float_test": fl.get("test", fl.get("test_acc")), "pick": key,
        "val": v["val"], "test": v.get("test", v.get("test_acc")), "half": v.get("test_half"),
        "trig": v.get("trig"), "tuned": v.get("trig_tuned"), "thr": v.get("tuned_thr"),
        "synops": v.get("synops"), "notes": " ".join(notes)})

keyf = {"val": lambda r: -r["val"], "test": lambda r: -(r["test"] or 0), "synops": lambda r: r["synops"] or 1e18}[sort]
rows.sort(key=keyf)
pct = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
print(f"{'run':24s} {'arch':10s} {'weights':>7s} | float {'val':>5s} {'test':>5s} | fixed {'val':>5s} {'test':>5s} {'0.5s':>5s} |"
      f" {'ok':>5s} {'wr':>5s} {'mi':>5s} {'fa':>5s} | tuned {'thr':>4s} {'ok':>5s} {'fa':>5s} | {'SynOps':>7s}  notes")
for r in rows:
    syn = "    -  " if r["synops"] is None else f"{r['synops'] / 1e3:6.0f}k"
    w = "-" if r["weights"] is None else str(r["weights"])
    t = r["trig"] or {}
    u = r["tuned"] or {}
    thr = "  - " if r["thr"] is None else f"{r['thr']:4.2f}"
    print(f"{r['run'][:24]:24s} {r['arch']:10s} {w:>7s} |       {pct(r['float_val'])} {pct(r['float_test'])} |"
          f"       {pct(r['val'])} {pct(r['test'])} {pct(r['half'])} | {pct(t.get('correct'))} {pct(t.get('wrong'))} "
          f"{pct(t.get('missed'))} {pct(t.get('false'))} |       {thr} {pct(u.get('correct'))} {pct(u.get('false'))} |"
          f" {syn:>7s}  {r['notes']}")
