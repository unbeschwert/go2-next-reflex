"""Finding clips, splitting them by person, and caching them as event-count tensors."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import Config
from .events import NAME_RE, events_to_counts, read_events

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Clip:
    path: str
    label: int
    person: str


def discover(cfg: Config) -> dict[str, list[Clip]]:
    """Clips per split. Labels come from the class name in the file name, in cfg.data.classes order."""
    d = cfg.data
    root = Path(d.root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset folder {root} not found")
    label_of = {name: i for i, name in enumerate(d.classes)}
    result = {}
    for split in SPLITS:
        clips = []
        for person in getattr(d, split):
            folder = root / person
            if not folder.is_dir():
                raise FileNotFoundError(f"{split}: person folder {folder} not found")
            for path in sorted(folder.glob("gesture_*.bin")):
                match = NAME_RE.match(path.name)
                if match and match.group(2) in label_of:
                    clips.append(Clip(str(path), label_of[match.group(2)], person))
        if d.limit_per_split and len(clips) > d.limit_per_split:
            clips = sorted(random.Random(cfg.seed).sample(clips, d.limit_per_split), key=lambda c: c.path)
        if not clips:
            raise ValueError(f"{split}: no clips of classes {d.classes} in {getattr(d, split)}")
        result[split] = clips
    return result


def describe(splits: dict[str, list[Clip]], classes: list[str]) -> str:
    lines = []
    for split, clips in splits.items():
        per_class = Counter(c.label for c in clips)
        people = sorted({c.person for c in clips})
        counts = ", ".join(f"{classes[i]} {per_class.get(i, 0)}" for i in range(len(classes)))
        lines.append(f"  {split:5s} {len(clips):5d} clips from {len(people)} people ({counts})")
    return "\n".join(lines)


def _cache_paths(cfg: Config, split: str, clips: list[Clip]) -> tuple[Path, Path]:
    d = cfg.data
    key = json.dumps([d.time_steps, d.time_window_us, d.downsampling, [(c.path, c.label) for c in clips]])
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    folder = Path(d.cache_dir) / f"T{d.time_steps}_w{d.time_window_us}_ds{d.downsampling}"
    return folder / f"{split}-{digest}-x.npy", folder / f"{split}-{digest}-y.npy"


def _load_counts(args: tuple[str, int, int, int]) -> np.ndarray:
    path, time_steps, window_us, downsampling = args
    return events_to_counts(read_events(path), time_steps, window_us, downsampling)


def build_cache(cfg: Config, splits: dict[str, list[Clip]], workers: int = 8) -> dict[str, tuple[Path, Path]]:
    """Write (N, T, 2, H, W) uint16 event counts and (N,) labels per split; reuse existing caches."""
    d = cfg.data
    out = {}
    for split, clips in splits.items():
        x_path, y_path = _cache_paths(cfg, split, clips)
        out[split] = (x_path, y_path)
        if x_path.exists() and y_path.exists():
            print(f"  {split}: using cache {x_path}", flush=True)
            continue
        x_path.parent.mkdir(parents=True, exist_ok=True)
        # Per process: runs started together (remote/queue.sh) may build the same cache at once.
        tmp = x_path.with_suffix(f".partial-{os.getpid()}.npy")
        shape = (len(clips), d.time_steps, 2, d.size, d.size)
        x = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=shape)
        jobs = [(c.path, d.time_steps, d.time_window_us, d.downsampling) for c in clips]
        with Pool(workers) as pool:
            for i, counts in enumerate(pool.imap(_load_counts, jobs, chunksize=4)):
                x[i] = counts
                if (i + 1) % 250 == 0 or i + 1 == len(jobs):
                    print(f"  {split}: {i + 1}/{len(jobs)} clips", flush=True)
        x.flush()
        del x
        if x_path.exists() and y_path.exists():   # another run finished the same cache first
            tmp.unlink()
            print(f"  {split}: using the cache another run just finished", flush=True)
            continue
        np.save(y_path, np.array([c.label for c in clips], dtype=np.int64))
        tmp.replace(x_path)
        (x_path.parent / f"{x_path.stem}-files.txt").write_text("\n".join(c.path for c in clips) + "\n")
    return out
