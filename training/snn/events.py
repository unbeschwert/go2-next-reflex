"""Reading event recordings (.bin, as written by river-tools and in the Motion Buddy dataset)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .config import SENSOR_SIZE

# 4-byte big-endian payload size, then 13-byte big-endian records (river_tools.capture.events).
EVENT_DTYPE = np.dtype([("x", ">u2"), ("y", ">u2"), ("polarity", "u1"), ("timestamp", ">i8")])

# gesture_<NN>-<class name>-<style>-<take>.bin
NAME_RE = re.compile(r"^gesture_(\d+)-(.+)-([a-z]+)-(\d+)\.bin$")


def read_events(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    size = int.from_bytes(data[:4], "big")
    if size != len(data) - 4 or size % EVENT_DTYPE.itemsize:
        raise ValueError(f"{path}: invalid recording (header {size}, file {len(data)} bytes)")
    return np.frombuffer(data, dtype=EVENT_DTYPE, offset=4)


def events_to_counts(events: np.ndarray, time_steps: int, window_us: int, downsampling: int) -> np.ndarray:
    """Count events per (time step, polarity, block row, block column) -> uint16 (T, 2, H, W).

    Time starts at the first event. Events after time_steps * window_us are dropped.
    Each output pixel is the sum over a downsampling x downsampling block of sensor pixels;
    the encoding (mean / sum / binary) is applied later, at training time.
    """
    size = SENSOR_SIZE // downsampling
    counts = np.zeros(time_steps * 2 * size * size, dtype=np.int64)
    if len(events):
        t = events["timestamp"].astype(np.int64)
        step = (t - t[0]) // window_us
        x = events["x"].astype(np.int64)
        y = events["y"].astype(np.int64)
        p = events["polarity"].astype(np.int64)
        ok = (step < time_steps) & (x < SENSOR_SIZE) & (y < SENSOR_SIZE) & (p < 2)
        index = ((step[ok] * 2 + p[ok]) * size + y[ok] // downsampling) * size + x[ok] // downsampling
        counts = np.bincount(index, minlength=counts.size)
    return np.minimum(counts, np.iinfo(np.uint16).max).astype(np.uint16).reshape(time_steps, 2, size, size)
