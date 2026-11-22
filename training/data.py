"""Build a compact event-tensor cache for the five gesture classes."""

import argparse
from hashlib import sha256
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

from event_data import SIZE, decode_recording


CLASSES = ("sit", "lay-down", "hello", "dance", "no-action")
CLASSES_WITHOUT_DANCE = tuple(name for name in CLASSES if name != "dance")
COMMAND_CLASSES = ("turn_left", "turn_right", "stop", "no_action")
MODEL_CLASS_ORDERS = (CLASSES, CLASSES_WITHOUT_DANCE, COMMAND_CLASSES)
SPLIT_NAMES = ("train", "validation", "test")
GRID = 32
CLIP_TICKS = 1_000_000
TICKS_PER_STEP = 100_000
STEPS = CLIP_TICKS // TICKS_PER_STEP
MAX_COUNT = 63
INPUT_SHAPE = (STEPS, 2, GRID, GRID)
RECORDING = re.compile(r"motion_buddy_dataset/(\d{2})/gesture_\d+-(.+)-(?:fast|large|normal)-\d+\.bin")


def input_shape(ticks_per_step, grid=GRID):
    if ticks_per_step <= 0 or ticks_per_step > CLIP_TICKS // 2 or CLIP_TICKS % ticks_per_step:
        raise ValueError("The bin size must divide a one-second clip into at least two equal bins.")
    if grid <= 0 or SIZE % grid:
        raise ValueError("The event grid must divide the camera resolution evenly.")
    return (CLIP_TICKS // ticks_per_step, 2, grid, grid)


def bin_events(events, timestamps, ticks_per_step=TICKS_PER_STEP, *, grid=GRID, max_count=MAX_COUNT):
    shape = input_shape(ticks_per_step, grid)
    limit = np.searchsorted(timestamps, int(timestamps[0]) + CLIP_TICKS)
    steps = ((timestamps[:limit] - timestamps[0]) // ticks_per_step).astype(np.int64)
    events = events[:limit]
    factor = SIZE // grid
    flat_index = ((steps * 2 + events["p"]) * grid + events["y"] // factor) * grid + events["x"] // factor
    counts = np.bincount(flat_index, minlength=int(np.prod(shape))).reshape(shape)
    return counts if max_count is None else np.minimum(counts, max_count)


def write_cache(output, *, ticks_per_step=TICKS_PER_STEP, **arrays):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as cache:
        np.savez(cache, **arrays, ticks_per_step=ticks_per_step)
    temporary.replace(output)
    print(f"Saved {output}")


def prepare(archive, output, ticks_per_step=TICKS_PER_STEP):
    shape = input_shape(ticks_per_step)
    with ZipFile(archive) as source:
        records = []
        for info in source.infolist():
            match = RECORDING.fullmatch(info.filename)
            if match:
                group, gesture = match.groups()
                if gesture in CLASSES:
                    records.append((info, int(group), CLASSES.index(gesture)))
        if not records:
            raise ValueError(f"No labeled gesture recordings in {archive}")
        records.sort(key=lambda record: record[0].header_offset)
        frames = np.empty((len(records), *shape), dtype=np.uint8)
        groups = np.empty(len(records), dtype=np.int64)
        labels = np.empty(len(records), dtype=np.int64)
        paths = []
        fingerprint = sha256()
        for i, (info, group, label) in enumerate(records):
            events, timestamps = decode_recording(source.read(info), info.filename)
            frames[i] = bin_events(events, timestamps, ticks_per_step)
            groups[i], labels[i] = group, label
            paths.append(info.filename)
            fingerprint.update(f"{info.filename}\t{info.CRC:08x}\t{info.file_size}\n".encode())
            if (i + 1) % 250 == 0 or i + 1 == len(records):
                print(f"Prepared {i + 1}/{len(records)} clips", flush=True)

    write_cache(output, frames=frames, groups=groups, labels=labels, paths=paths,
                classes=CLASSES, archive_fingerprint=fingerprint.hexdigest(), ticks_per_step=ticks_per_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("durance-ai-handgesture-recognition.zip"))
    parser.add_argument("--output", type=Path, default=Path(".local/training/gestures.npz"))
    parser.add_argument("--bin-us", type=int, default=TICKS_PER_STEP, help="Event bin duration in microseconds")
    args = parser.parse_args()
    prepare(args.archive, args.output, args.bin_us)
