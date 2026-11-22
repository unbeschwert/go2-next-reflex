"""Evaluate 50 and 100 ms left-palm forecasts on six session 20 clips.

Run `python -m labeling.benchmark_left_palm` from the repository root with
NumPy, SciPy, and the session 20 archive. Each CSV row represents one 50 ms
event window. Blank coordinates indicate an indistinct palm. The seeded
tracker receives the first visible point per clip, then sees only events.
"""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from event_data import SIZE, decode_recording


WINDOW_US = 50_000
PATCH_RADIUS = 10
SEARCH_RADIUS = 28
MIN_CORRELATION = 0.25


def read_labels(path):
    clips = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            clips[row["recording"]].append(row)
    return clips


def event_images(raw, name, rows):
    events, timestamps = decode_recording(raw, name)
    starts = [int(row["window_start_us"]) for row in rows]
    expected = list(range(int(timestamps[0]), int(timestamps[0]) + len(rows) * WINDOW_US, WINDOW_US))
    if starts != expected or [int(row["frame"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"Annotation timing differs from recording: {name}")
    boundaries = np.searchsorted(timestamps, [*starts, starts[-1] + WINDOW_US])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        frame = events[start:stop]
        image = np.zeros((SIZE, SIZE), dtype=np.float32)
        image[frame["y"], frame["x"]] = 1
        yield gaussian_filter(image, sigma=1.5)


def match_patch(image, center, template):
    """Find the best normalized patch match near the previous position."""
    x, y = center
    radius = PATCH_RADIUS + SEARCH_RADIUS
    padded = np.pad(image, radius)
    region = padded[y:y + 2 * radius + 1, x:x + 2 * radius + 1]
    centered = template - template.mean()
    kernel = np.ones_like(template)
    patch_sum = fftconvolve(region, kernel, mode="valid")
    patch_energy = fftconvolve(region ** 2, kernel, mode="valid")
    variance = np.maximum(patch_energy - patch_sum ** 2 / template.size, 0)
    scale = np.sqrt(variance * np.sum(centered ** 2))
    scores = np.full(scale.shape, -1.0)
    correlation = fftconvolve(region, centered[::-1, ::-1], mode="valid")
    np.divide(correlation, scale, out=scores, where=scale > 1e-6)
    dy, dx = np.unravel_index(scores.argmax(), scores.shape)
    position = center + np.array([dx, dy]) - SEARCH_RADIUS
    if scores[dy, dx] < MIN_CORRELATION or np.any((position < 0) | (position >= SIZE)):
        return None
    return position


def track(images, first_frame, seed):
    position = np.array(seed, dtype=int)
    template = None
    points = {}
    for frame, image in enumerate(images):
        if frame < first_frame:
            continue
        if template is not None:
            match = match_patch(image, position, template)
            if match is None:
                continue
            position = match
        x, y = position
        patch = np.pad(image, PATCH_RADIUS)[y:y + 2 * PATCH_RADIUS + 1, x:x + 2 * PATCH_RADIUS + 1]
        template = patch
        points[frame] = position
    return points


def forecasts(labels, points, steps):
    errors = []
    for frame, target in labels.items():
        current = frame - steps
        if current not in points or current - 1 not in points:
            continue
        latest = points[current]
        velocity = latest - points[current - 1]
        hold = np.linalg.norm(latest - target)
        extrapolate = np.linalg.norm(latest + steps * velocity - target)
        errors.append((hold, extrapolate))
    return errors


def summarize(label, errors):
    values = np.array(errors)
    print(f"{label} ({len(values)} forecasts)")
    if not len(values):
        return
    for method, column in zip(("Hold last position", "Constant velocity"), values.T):
        print(f"  {method:22} median {np.median(column):5.1f} px, "
              f"p90 {np.percentile(column, 90):5.1f} px, "
              f"mean {column.mean():5.1f} px, within 10 px {(column <= 10).mean():5.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("go2_gestures_s20.zip"))
    parser.add_argument("--labels", type=Path, default=Path(__file__).with_name("session20_left_palm.csv"))
    args = parser.parse_args()
    errors = defaultdict(list)
    localization_errors = []
    localization_targets = 0
    with ZipFile(args.archive) as archive:
        for name, rows in read_labels(args.labels).items():
            labels = {int(row["frame"]): np.array([float(row["left_palm_x"]), float(row["left_palm_y"])])
                      for row in rows if row["left_palm_x"] and row["left_palm_y"]}
            if not labels:
                continue
            first_frame = min(labels)
            points = track(event_images(archive.read(name), name, rows), first_frame, labels[first_frame])
            localization_targets += len(labels) - 1
            localization_errors.extend(np.linalg.norm(points[frame] - target) for frame, target in labels.items()
                                       if frame != first_frame and frame in points)
            for steps in (1, 2):
                # Evaluate both input sources on the same annotated target and input frames.
                comparable = {frame: target for frame, target in labels.items()
                              if frame - steps in labels and frame - steps - 1 in labels}
                for method, observations in (("Manual observations", labels), ("Seeded event tracker", points)):
                    scores = forecasts(comparable, observations, steps)
                    errors[steps, method, "all"].extend(scores)
                    errors[steps, method, rows[0]["style"]].extend(scores)

    print(f"Seeded tracker localization: {len(localization_errors)}/{localization_targets} labeled frames "
          "after the seed")
    if localization_errors:
        print(f"  Median {np.median(localization_errors):.1f} px, "
              f"p90 {np.percentile(localization_errors, 90):.1f} px")
    for steps in (1, 2):
        print(f"\n{steps * WINDOW_US // 1000} ms ahead")
        for method in ("Manual observations", "Seeded event tracker"):
            summarize(method, errors[steps, method, "all"])
        for style in ("normal", "fast", "large"):
            summarize(f"Manual observations / {style}", errors[steps, "Manual observations", style])


if __name__ == "__main__":
    main()
