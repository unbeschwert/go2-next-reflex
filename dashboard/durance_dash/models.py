"""Deployable weight files on the laptop: where they are, which network they fit, how they scored.

Two kinds: weights files of the team's train_snn_v8/v9 networks under models/ (recognised by their value
count, ARCHS below; they replace only /root/weights.txt), and runs of our training pipeline under runs/
(snnruns.py: network, formats and firmware come from the run's own files; they install a full board setup)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import snnruns

# input size, layers (in, out, kernel, stride, padding): same table as training/train_snn_v9.py
ARCHS = {
    "best80": (80, [(2, 16, 5, 2, 1), (16, 32, 3, 1, 1), (32, 32, 6, 2, 2), (32, 256, 6, 2, 0), (256, 128, 3, 2, 0), (128, 5, 3, 1, 0)]),
    "nb40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 9, 1, 0), (256, 5, 1, 1, 0)]),
    "fast80": (80, [(2, 16, 6, 4, 0), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 9, 1, 0), (256, 5, 1, 1, 0)]),
    "v3": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 16, 6, 2, 2), (16, 128, 6, 2, 0), (128, 5, 2, 1, 0)]),
    "c40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 2, 1), (16, 64, 6, 2, 0), (64, 5, 3, 1, 0)]),
    "slim80": (80, [(2, 8, 5, 2, 1), (8, 16, 3, 1, 1), (16, 16, 6, 2, 2), (16, 128, 6, 2, 0), (128, 64, 3, 2, 0), (64, 5, 3, 1, 0)]),
}


def value_count(layers) -> int:
    return sum(ci * co * k * k + co + 2 for ci, co, k, _, _ in layers)


COUNTS = {value_count(layers): (name, size) for name, (size, layers) in ARCHS.items()}

_cache: dict[str, tuple[int, dict]] = {}


def describe(path: Path) -> dict | None:
    """md5, value count, architecture (by count) of a weights.txt; cached by mtime. None if not weights."""
    try:
        st = path.stat()
    except OSError:
        return None
    hit = _cache.get(str(path))
    if hit and hit[0] == st.st_mtime_ns:
        return hit[1]
    if st.st_size > 50_000_000:
        return None
    data = path.read_bytes()
    try:
        vals = data.split()
        int(vals[0]), int(vals[-1])
    except (ValueError, IndexError):
        return None
    arch, size = COUNTS.get(len(vals), (None, None))
    info = {"md5": hashlib.md5(data).hexdigest(), "values": len(vals), "arch": arch, "input": size,
            "bytes": st.st_size, "mtime": st.st_mtime}
    _cache[str(path)] = (st.st_mtime_ns, info)
    return info


def _metrics(run_dir: Path, bits: str, name: str = "metrics.json") -> tuple[dict | None, str | None]:
    m = run_dir / name
    if not m.exists():
        return None, None
    try:
        j = json.loads(m.read_text())
    except ValueError:
        return None, None
    res = j.get("results", {})
    key = f"qat_q{bits}" if f"qat_q{bits}" in res else f"float_ptq_q{bits}"
    r = res.get(key)
    names = j.get("classes") or []
    classes = "own" if any("heart" in str(c).lower() for c in names) else ("dataset" if names else None)
    if not r:
        return None, classes
    return {"val": r.get("val"), "test": r.get("test"), "synops": r.get("synops"),
            "trig_correct": (r.get("trig") or {}).get("correct"), "trig_false": (r.get("trig") or {}).get("false")}, classes


# weight format of the team's weights files: "8h" = the 8-bit firmware (/64), "16" = firmware (3) (/4096)
LEGACY_FORMATS = {"8h": "Q2.6", "16": "Q4.12"}


def scan(models_dir: Path, runs_dir: Path | None = None) -> list[dict]:
    """All weights files under models/ (board_ready/, fetched/<run>/, r*/<run>/), then the runs under runs/."""
    out = _scan_models(models_dir)
    out += snnruns.scan(runs_dir)
    return out


def _scan_models(models_dir: Path) -> list[dict]:
    out = []
    seen = set()
    patterns = ["board_ready/*.txt", "optimised_snn/*/weights_*.txt",          # named, deployable files
                "fetched/*/weights_q*.txt", "r*/*/weights_q*.txt", "*/weights_q*.txt"]
    for pat in patterns:
        for p in sorted(models_dir.glob(pat)):
            if p in seen or p.name.endswith("_parameters.txt") and "board_ready" not in str(p):
                continue
            seen.add(p)
            info = describe(p)
            if info is None or info["arch"] is None:
                continue
            bits = "16" if "q16" in p.name else "8h"
            rel = p.relative_to(models_dir)
            named = rel.parts[0] in ("board_ready", "optimised_snn")
            if rel.parts[0] == "optimised_snn":                 # metrics_<run>.json next to weights_<run>_<md5>.txt
                run = p.stem[len("weights_"):].rsplit("_", 1)[0] if p.stem.startswith("weights_") else p.stem
                metrics, classes = _metrics(p.parent, bits, f"metrics_{run}.json")
            else:
                metrics, classes = _metrics(p.parent, bits) if not named else (None, None)
            if classes is None:
                n = str(rel).lower()
                classes = "own" if "own" in n else ("dataset" if ("dataset" in n or "organisers" in n) else None)
            out.append({"id": hashlib.sha1(str(rel).encode()).hexdigest()[:12],
                        "name": (f"{p.parent.name}/{p.stem}" if rel.parts[0] == "optimised_snn" else p.stem) if named
                        else f"{p.parent.name}/{p.stem}", "path": str(rel), "source": rel.parts[0], "classes": classes,
                        "metrics": metrics, "bits": bits, "kind": "weights",
                        "formats": {"weights": LEGACY_FORMATS[bits]}, **info})
    return out


def find(models_dir: Path, model_id: str, runs_dir: Path | None = None) -> dict | None:
    return next((m for m in scan(models_dir, runs_dir) if m["id"] == model_id), None)


def by_md5(models_dir: Path, md5: str, runs_dir: Path | None = None) -> dict | None:
    """The model whose weights.txt has this md5 (runs first: their entry knows the network)."""
    ms = scan(models_dir, runs_dir)
    return next((m for m in ms if m["md5"] == md5 and m["kind"] == "setup"), None) or \
        next((m for m in ms if m["md5"] == md5), None)


def name_for_md5(models_dir: Path, md5: str, runs_dir: Path | None = None) -> str | None:
    m = by_md5(models_dir, md5, runs_dir)
    return m["name"] if m else None


def public(m: dict) -> dict:
    """A model entry for the browser: no local paths of runs."""
    return snnruns.public(m) if m.get("kind") == "setup" else m
