"""Show what each training augmentation does to one clip, as the network sees it (downsampled frames).

Rows: the original, then each augmentation alone (at full probability), then everything in the config
together. Columns: groups of frames summed over time (red = ON, blue = OFF events).

Usage (from the repo root; needs the train group: `uv sync --group train`):
    uv run --group train scripts/view_augment.py data/motion_buddy_dataset/01/gesture_01-hello-normal-01.bin
    uv run --group train scripts/view_augment.py <file.bin> --config training/configs/motion_buddy_40_aug.yaml \
        --save doc/img/augmentations.png
"""
import argparse
import copy
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.snn import augment as aug  # noqa: E402
from training.snn.config import AugmentConfig, load_config  # noqa: E402
from training.snn.events import events_to_counts, read_events  # noqa: E402


def to_rgb(frames: np.ndarray, vmax: float) -> np.ndarray:
    """frames [n, 2, H, W] summed → RGB image (sqrt scale); red = ON (channel 1), blue = OFF (channel 0)."""
    off, on = np.sqrt(np.clip(frames.sum(0), 0, None) / vmax)
    return np.clip(np.stack([on, np.zeros_like(on), off], -1), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--config", default="training/configs/motion_buddy_40_aug.yaml")
    parser.add_argument("--columns", type=int, default=8, help="time groups to show")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", help="write a PNG instead of opening a window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    T = cfg.data.time_steps
    counts = torch.from_numpy(events_to_counts(read_events(Path(args.file)), T, cfg.data.time_window_us,
                                               cfg.data.downsampling).astype(np.float32))[None]

    rows = [("original", None)]
    for f in dataclasses.fields(AugmentConfig):
        if f.name not in ("qat", "mixup"):     # mixup blends clips of a batch (train.py), not shown here
            single = AugmentConfig()
            getattr(single, f.name).__dict__.update(dataclasses.asdict(getattr(cfg.augment, f.name)) | {"p": 1.0})
            rows.append((f.name.replace("_", " "), single))
    rows.append(("integer shift", "shift"))
    rows.append(("all (config)", cfg.augment))

    images = []
    for name, a in rows:
        torch.manual_seed(args.seed)
        c = copy.deepcopy(cfg)
        if a is None:
            x, warmup = counts, 0
        elif a == "shift":
            x, warmup = aug.random_shift(counts, max(cfg.data.max_shift, 1)), 0
        else:
            c.augment = a
            if name != "all (config)":
                c.data.max_shift = 0
            x, warmup = aug.augment(counts.clone(), c)
        frames = x[0, warmup:].numpy()          # the clip itself; the warm-start prefix is only counted
        label = name + (f"\n(+{warmup} frames before)" if warmup else "")
        groups = np.array_split(np.arange(len(frames)), args.columns)
        images.append((label, [frames[g] for g in groups], warmup))

    original = np.stack([g.sum(0) for g in images[0][1]])
    vmax = float(np.percentile(original[original > 0], 99)) or 1.0      # same scale for every row
    fig, axes = plt.subplots(len(images), args.columns, figsize=(1.3 * args.columns, 1.35 * len(images)),
                             squeeze=False)
    for r, (label, groups, warmup) in enumerate(images):
        for col, g in enumerate(groups):
            ax = axes[r, col]
            ax.imshow(to_rgb(g, vmax), interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    for col in range(args.columns):
        axes[0, col].set_title(f"part {col + 1}", fontsize=7)
    fig.suptitle(f"{Path(args.file).name}: augmentations ({cfg.data.size}x{cfg.data.size}, {T} frames)", fontsize=9)
    fig.tight_layout()
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=110)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
