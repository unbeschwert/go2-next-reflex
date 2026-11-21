"""Show an event-camera recording (motion_buddy_dataset .bin) as a grid of frames.

The clip is split into equal time slices. Each slice becomes one 320x320 image:
red = pixels that got brighter (p=1), blue = pixels that got darker (p=0).

File format: 4-byte big-endian length, then 13-byte big-endian records
(x uint16, y uint16, p uint8, t int64 microseconds).

Usage (from the repo root; `uv sync` once to create .venv):
    uv run scripts/view_events.py data/motion_buddy_dataset/01/gesture_01-sit-normal-01.bin
    uv run scripts/view_events.py <file.bin> --slices 16 --save sit.png

Without --save the figure opens in a window, or, if no GUI toolkit is available
and you're in the kitty terminal, is drawn directly in the terminal.
"""
import argparse
import io
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Same layout as river_tools.capture.events.EVENT_DTYPE
EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("p", "u1"), ("t", ">i8")])
SENSOR_SIZE = 320


def load_events(path):
    """Read a .bin file and return events with t in microseconds from the clip start."""
    with open(path, "rb") as f:
        length = int.from_bytes(f.read(4), "big")
        events = np.fromfile(f, dtype=EVENT_DTYPE)
    if length != events.nbytes:
        raise ValueError(f"{path}: header says {length} bytes, file has {events.nbytes}")
    t = events["t"].astype(np.int64)
    return events["x"].astype(np.int64), events["y"].astype(np.int64), events["p"], t - t[0]


def to_frames(x, y, p, t, slices):
    """Count events per pixel and polarity in each time slice -> (slices, 2, H, W)."""
    duration = max(int(t[-1]), 1)
    slice_index = np.minimum(t * slices // (duration + 1), slices - 1)
    frames = np.zeros((slices, 2, SENSOR_SIZE, SENSOR_SIZE), dtype=np.int32)
    np.add.at(frames, (slice_index, p, y, x), 1)
    return frames, duration


def to_rgb(frame, scale):
    """Red channel = ON events, blue channel = OFF events, on a black background."""
    off, on = frame[0], frame[1]
    rgb = np.zeros((SENSOR_SIZE, SENSOR_SIZE, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(on / scale, 0, 1)
    rgb[..., 2] = np.clip(off / scale, 0, 1)
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, help="path to a .bin recording")
    parser.add_argument("--slices", type=int, default=12, help="number of time slices (default 12)")
    parser.add_argument("--save", type=Path, help="write the figure to this image file instead of showing it")
    args = parser.parse_args()

    x, y, p, t = load_events(args.file)
    frames, duration = to_frames(x, y, p, t, args.slices)
    # Use the same brightness for every slice so they can be compared; the 99th
    # percentile of non-zero pixels keeps a few very busy pixels from washing it out.
    nonzero = frames[frames > 0]
    scale = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0

    cols = int(np.ceil(np.sqrt(args.slices)))
    rows = int(np.ceil(args.slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows + 0.6), squeeze=False)
    slice_ms = duration / args.slices / 1000
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= args.slices:
            continue
        ax.imshow(to_rgb(frames[i], scale), interpolation="nearest")
        ax.set_title(f"{i * slice_ms:.0f}–{(i + 1) * slice_ms:.0f} ms\n{frames[i].sum():,} ev", fontsize=8)

    fig.suptitle(f"{args.file.parent.name}/{args.file.name}  ·  {len(t):,} events  ·  "
                 f"{duration / 1000:.0f} ms  ·  red = ON, blue = OFF", fontsize=10)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"saved {args.save}")
    elif plt.get_backend().lower() == "agg" and shutil.which("kitten") and os.environ.get("KITTY_WINDOW_ID"):
        # No GUI toolkit available, but we're in kitty: draw the image in the terminal.
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=100)
        shown = subprocess.run(["kitten", "icat", "--align", "left"], input=png.getvalue(),
                               stderr=subprocess.DEVNULL).returncode == 0
        if not shown:
            fallback = Path(f"{args.file.stem}.png")
            fallback.write_bytes(png.getvalue())
            print(f"could not draw in the terminal, saved {fallback} instead")
    else:
        plt.show()


if __name__ == "__main__":
    main()
