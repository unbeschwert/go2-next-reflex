"""Prepare labeled movement recordings or a synthetic pipeline smoke test."""

import argparse
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from event_data import decode_recording
from training.data import COMMAND_CLASSES, GRID, INPUT_SHAPE, MAX_COUNT, SPLIT_NAMES, STEPS, bin_events, write_cache


def prepare_recorded(archive, output):
    with ZipFile(archive) as source:
        records = []
        group_splits = {}
        names = set()
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if not path.parts or path.parts[0] != "movement_commands" or path.suffix != ".bin":
                continue
            if len(path.parts) != 5:
                raise ValueError(f"Expected movement_commands/<split>/<group>/<command>/<file>.bin: {path}")
            _, split, group, command, _ = path.parts
            if split not in SPLIT_NAMES or command not in COMMAND_CLASSES:
                raise ValueError(f"Unknown split or command in {path}")
            if group_splits.setdefault(group, split) != split:
                raise ValueError(f"Recording group {group} appears in multiple splits")
            if info.filename in names:
                raise ValueError(f"Duplicate recording {path}")
            names.add(info.filename)
            records.append((info, split, group, command))

        present = {(split, command) for _, split, _, command in records}
        missing = [(split, command) for split in SPLIT_NAMES for command in COMMAND_CLASSES
                   if (split, command) not in present]
        if missing:
            raise ValueError(f"Missing recorded split/command pairs: {missing}")

        frames = np.empty((len(records), *INPUT_SHAPE), dtype=np.uint8)
        fingerprint = sha256()
        for index, (info, _, _, _) in enumerate(records):
            events, timestamps = decode_recording(source.read(info), info.filename)
            frames[index] = bin_events(events, timestamps)
            fingerprint.update(f"{info.filename}\t{info.CRC:08x}\t{info.file_size}\n".encode())

    write_cache(output, frames=frames, groups=[group for _, _, group, _ in records],
                labels=[COMMAND_CLASSES.index(command) for _, _, _, command in records],
                splits=[split for _, split, _, _ in records], paths=[info.filename for info, _, _, _ in records],
                classes=COMMAND_CLASSES, archive_fingerprint=fingerprint.hexdigest(), source="recorded")


def synthetic_clip(command, rng):
    y, x = np.ogrid[:GRID, :GRID]
    cx, cy = rng.uniform(12, 19), rng.uniform(9, 23)
    time = np.clip((np.arange(STEPS) - rng.uniform(0, 1.6)) / rng.uniform(5.2, 7.5), 0, 1)
    if command == "turn_left" or command == "turn_right":
        direction = -1 if command == "turn_left" else 1
        center_x = cx + direction * rng.uniform(9, 17) * (time - 0.5)
        radius_x = radius_y = np.full(STEPS, rng.uniform(2.2, 3.7))
    elif command == "stop":
        center_x = np.full(STEPS, cx)
        radius_x = 1.6 + time * rng.uniform(3.5, 5.5)
        radius_y = 1.9 + time * rng.uniform(3.9, 6.0)
    else:
        center_x = cx + 0.9 * np.sin(2 * np.pi * time)
        radius_x = radius_y = np.full(STEPS, rng.uniform(1.8, 3.7))
        cy += rng.uniform(-3, 3)

    frames = rng.poisson(rng.uniform(0.006, 0.04), size=INPUT_SHAPE)
    for t in range(STEPS):
        shape = ((x - center_x[t]) / radius_x[t]) ** 2 + ((y - cy) / radius_y[t]) ** 2 <= 1
        if t:
            frames[t, 0, previous & ~shape] += rng.integers(10, 28)
            frames[t, 1, shape & ~previous] += rng.integers(10, 28)
        previous = shape
    return frames.clip(0, MAX_COUNT).astype(np.uint8)


def prepare_synthetic(output, seed):
    rng = np.random.default_rng(seed)
    frames, labels, splits = [], [], []
    for split, clips_per_class in zip(SPLIT_NAMES, (80, 20, 20), strict=True):
        for label, command in enumerate(COMMAND_CLASSES):
            for _ in range(clips_per_class):
                frames.append(synthetic_clip(command, rng))
                labels.append(label)
                splits.append(split)
    inputs = np.stack(frames)
    write_cache(output, frames=inputs, labels=labels, splits=splits, classes=COMMAND_CLASSES,
                archive_fingerprint=sha256(inputs).hexdigest(), source="synthetic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="ZIP of labeled camera recordings")
    source.add_argument("--synthetic", action="store_true", help="Generate a local workflow fixture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    if args.synthetic:
        prepare_synthetic(args.output, args.seed)
    else:
        prepare_recorded(args.archive, args.output)
