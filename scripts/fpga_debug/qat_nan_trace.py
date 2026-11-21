"""Find where QAT produces its first non-finite gradient (server; replays `qat` on a copy of a run).

Usage: uv run --group train python scripts/fpga_debug/qat_nan_trace.py runs/motion-buddy-40-e300 [--set k=v ...]
"""
import shutil, sys, time
from pathlib import Path
import torch
sys.path[:0] = ["."]
from training.snn import data as data_mod, train as T
from training.snn.__main__ import apply_overrides
from training.snn.config import load_config

src = Path(sys.argv[1]); dst = Path("runs/debug-nan-" + src.name); dst.mkdir(parents=True, exist_ok=True)
for f in ("float_best.pt", "config.yaml", "history.csv"):
    shutil.copy(src / f, dst / f)
cfg = load_config(dst / "config.yaml")
sets = [a.split("=", 1) and a for a in sys.argv[2:] if a != "--set"]
if sets:
    cfg = apply_overrides(cfg, sets)
state = {"step": 0, "model": None}
original_clip = torch.nn.utils.clip_grad_norm_

def bn_scales(model):
    out = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            s = (m.weight / torch.sqrt(m.running_var + m.eps)).detach().abs()
            nz = s[s > 0]
            out.append(f"{name}: min|scale| {nz.min().item() if len(nz) else 0:.3g}, min|gamma|>0 "
                       f"{m.weight.detach().abs()[m.weight.detach().abs() > 0].min().item():.3g}, min var {m.running_var.min().item():.3g}")
    return out

def checked_clip(params, max_norm, *a, **k):
    params = list(params)
    state["step"] += 1
    model = state["model"]
    names = {id(p): n for n, p in model.named_parameters()} if model is not None else {}
    bad = [(names.get(id(p), "?"), p.grad) for p in params if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad:
        print(f"\n!!! step {state['step']}: non-finite gradients in {[n for n, _ in bad]}")
        for n, g in bad:
            print(f"    {n}: shape {tuple(g.shape)}, non-finite {(~torch.isfinite(g)).sum().item()}, "
                  f"where (flat idx) {torch.nonzero(~torch.isfinite(g.flatten()))[:8].flatten().tolist()}")
        big = sorted(((n, p.grad.abs().max().item()) for n, p in model.named_parameters() if p.grad is not None
                      and torch.isfinite(p.grad).all()), key=lambda t: -t[1])[:5]
        print("    largest finite grads:", [(n, f"{v:.3g}") for n, v in big])
        print("    BN:", *bn_scales(model), sep="\n      ")
        raise SystemExit(1)
    if state["step"] % 50 == 0:
        total = original_clip(params, float("inf"))
        print(f"  step {state['step']}: grad norm {total.item():.3g}", flush=True)
    return original_clip(params, max_norm, *a, **k)

torch.nn.utils.clip_grad_norm_ = checked_clip
orig_fit = T.fit
def fit(model, *a, **k):
    state["model"] = model
    return orig_fit(model, *a, **k)
T.fit = fit
cache = data_mod.build_cache(cfg, data_mod.discover(cfg), workers=8)
T.qat_from_float(cfg, {s: cache[s] for s in data_mod.SPLITS}, dst)
