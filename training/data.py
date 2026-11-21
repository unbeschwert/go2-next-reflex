"""Build a compact event-tensor cache for the five gesture classes."""

import argparse
from hashlib import sha256
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

from event_data import SIZE, decode_recording


CLASSES = ("sit", "lay-down", "hello", "dance", "no-action")
COMMAND_CLASSES = ("turn_left", "turn_right", "stop", "no_action")
SPLIT_NAMES = ("train", "validation", "test")
GRID = 32
STEPS = 10
TICKS_PER_STEP = 100_000
MAX_COUNT = 63
INPUT_SHAPE = (STEPS, 2, GRID, GRID)
RECORDING = re.compile(r"motion_buddy_dataset/(\d{2})/gesture_\d+-(.+)-(?:fast|large|normal)-\d+\.bin")


def bin_events(events, timestamps):
    limit = np.searchsorted(timestamps, int(timestamps[0]) + STEPS * TICKS_PER_STEP)
    steps = ((timestamps[:limit] - timestamps[0]) // TICKS_PER_STEP).astype(np.int64)
    events = events[:limit]
    factor = SIZE // GRID
    flat_index = ((steps * 2 + events["p"]) * GRID + events["y"] // factor) * GRID + events["x"] // factor
    counts = np.bincount(flat_index, minlength=STEPS * 2 * GRID * GRID).reshape(INPUT_SHAPE)
    return np.minimum(counts, MAX_COUNT)


def write_cache(output, **arrays):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as cache:
        np.savez(cache, **arrays, ticks_per_step=TICKS_PER_STEP)
    temporary.replace(output)
    print(f"Saved {output}")


def prepare(archive, output):
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
        frames = np.empty((len(records), *INPUT_SHAPE), dtype=np.uint8)
        groups = np.empty(len(records), dtype=np.int64)
        labels = np.empty(len(records), dtype=np.int64)
        paths = []
        fingerprint = sha256()
        for i, (info, group, label) in enumerate(records):
            events, timestamps = decode_recording(source.read(info), info.filename)
            frames[i] = bin_events(events, timestamps)
            groups[i], labels[i] = group, label
            paths.append(info.filename)
            fingerprint.update(f"{info.filename}\t{info.CRC:08x}\t{info.file_size}\n".encode())
            if (i + 1) % 250 == 0 or i + 1 == len(records):
                print(f"Prepared {i + 1}/{len(records)} clips", flush=True)

    write_cache(output, frames=frames, groups=groups, labels=labels, paths=paths,
                classes=CLASSES, archive_fingerprint=fingerprint.hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("durance-ai-handgesture-recognition.zip"))
    parser.add_argument("--output", type=Path, default=Path(".local/training/gestures.npz"))
    args = parser.parse_args()
    prepare(args.archive, args.output)
