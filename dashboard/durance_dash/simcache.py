"""Build the simulator's replay cache: per virtual-person gesture, a few recorded clips (event counts at
the model's input size) and the per-frame output spikes the FPGA integer model (go2_viz/spleat_sim.py)
gives for them. Built once (about a minute), then reused."""
from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

EV = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])
PERSON_CLASS = {"heart": 1, "left": 2, "right": 3, "none": 5}


def clip_counts(data: bytes, size: int, steps: int = 100) -> np.ndarray:
    n = int.from_bytes(data[:4], "big")
    if n != len(data) - 4 or n % EV.itemsize:
        raise ValueError("invalid recording")
    ev = np.frombuffer(data, EV, offset=4)
    sc = 320 // size
    counts = np.zeros(steps * 2 * size * size, np.int64)
    if len(ev):
        t = (ev["timestamp"].astype(np.int64) - int(ev["timestamp"][0])) // 10_000
        ok = (t >= 0) & (t < steps) & (ev["x"] < 320) & (ev["y"] < 320) & (ev["polarity"] < 2)
        idx = ((t[ok] * 2 + ev["polarity"][ok].astype(np.int64)) * size + ev["y"][ok].astype(np.int64) // sc) * size \
            + ev["x"][ok].astype(np.int64) // sc
        counts = np.bincount(idx, minlength=counts.size)
    return counts.reshape(steps, 2, size, size)


def cache_path(state_dir: Path, weights: Path, zip_path: Path, session: str) -> Path:
    key = hashlib.sha1(f"{weights.stat().st_mtime_ns}{weights}{zip_path}{session}".encode()).hexdigest()[:10]
    return state_dir / f"sim_cache_{key}.npz"


def build(go2_viz: Path, weights: Path, zip_path: Path, session: str, out: Path, per_class: int = 6,
          log=print) -> Path:
    sys.path.insert(0, str(go2_viz))
    from spleat_sim import ARCHS, Spleat  # noqa: E402  (go2_viz is not a package)
    arch = "nb40"
    size = ARCHS[arch][0]
    z = zipfile.ZipFile(zip_path)
    names: dict[int, list[str]] = {}
    for n in sorted(z.namelist()):
        m = re.match(rf"^motion_buddy_dataset/{re.escape(session)}/gesture_(\d+)-[^/]+\.bin$", n)
        if m:
            names.setdefault(int(m.group(1)), []).append(n)
    arrays = {"size": np.array(size)}
    blank = np.zeros((2, size, size), np.float32)
    for person, cls in PERSON_CLASS.items():
        files = names.get(cls, [])[:per_class]
        if not files:
            log(f"no clips of class {cls} ({person}) in session {session}")
            continue
        outs, frames = [], []
        for i, n in enumerate(files):
            c = clip_counts(z.read(n), size)
            sim = Spleat(str(weights), arch, w_shift=6, b_shift=6, in_shift=6, leak_mode="retain")
            sim.run([blank] * 60)                      # settled start, as on the continuously running FPGA
            o = sim.run(c.astype(np.float32) / ((320 // size) ** 2))
            outs.append((np.asarray(o) > 0).astype(np.uint8))
            frames.append(np.minimum(c, 255).astype(np.uint8))
            log(f"{person}: clip {i + 1}/{len(files)}  output spikes/class {np.asarray(o).sum(0).astype(int).tolist()}")
        arrays[f"out_{person}"] = np.stack(outs)
        arrays[f"frames_{person}"] = np.stack(frames)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out)
    return out
