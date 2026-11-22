"""Train the Conv-LIF SNN to forecast the left palm 50 ms ahead."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from event_data import SIZE, decode_recording
from labeling.benchmark_left_palm import WINDOW_US, event_images, read_labels, track
from training.data import GRID, bin_events
from training.model import PARAMETER_FORMATS, GestureSNN
from training.train import resample_event_counts


BIN_US = 25_000
CONTEXT_STEPS = 2 * WINDOW_US // BIN_US
COORDINATE_OFFSET = SIZE / 2
COORDINATE_SCALE = SIZE / 4
MODEL_ARGS = {"num_classes": 2, "precision": "fp32", "ticks_per_step": BIN_US}
METHODS = ("snn", "manual_hold", "manual_velocity", "seeded_tracker_velocity")


@dataclass
class Recording:
    name: str
    frames: torch.Tensor
    targets: torch.Tensor
    comparable: torch.Tensor
    baselines: dict


def load_recordings(archive_path, label_path, grid=GRID):
    styles = defaultdict(list)
    with ZipFile(archive_path) as archive:
        for name, rows in read_labels(label_path).items():
            raw = archive.read(name)
            events, timestamps = decode_recording(raw, name)
            counts = bin_events(events, timestamps, BIN_US, grid=grid).astype(np.uint8)
            labels = {int(row["frame"]): np.array([float(row["left_palm_x"]), float(row["left_palm_y"])])
                      for row in rows if row["left_palm_x"] and row["left_palm_y"]}
            if not labels:
                continue
            first = min(labels)
            tracked = track(event_images(raw, name, rows), first, labels[first])
            frames, targets, comparable = [], [], []
            baselines = {method: [] for method in METHODS[1:]}
            for frame, target in labels.items():
                end = frame * WINDOW_US // BIN_US
                if end < CONTEXT_STEPS:
                    continue
                # The final input bin ends at the start of the labeled target window.
                frames.append(counts[end - CONTEXT_STEPS:end])
                targets.append(target)
                prior, latest = frame - 2, frame - 1
                compare = prior in labels and latest in labels
                comparable.append(compare)
                if compare:
                    baselines["manual_hold"].append(np.linalg.norm(labels[latest] - target))
                    baselines["manual_velocity"].append(np.linalg.norm(2 * labels[latest] - labels[prior] - target))
                    if prior in tracked and latest in tracked:
                        error = np.linalg.norm(2 * tracked[latest] - tracked[prior] - target)
                        baselines["seeded_tracker_velocity"].append(error)
            if frames:
                styles[rows[0]["style"]].append(Recording(
                    name, torch.from_numpy(np.stack(frames)), torch.tensor(np.stack(targets), dtype=torch.float32),
                    torch.tensor(comparable, dtype=torch.bool), baselines,
                ))
    if not styles or any(len(recordings) != 2 for recordings in styles.values()):
        raise ValueError("The two-fold benchmark requires two labeled recordings per style.")
    return styles


def translate(frames, targets):
    grid = frames.shape[-1]
    offset = torch.randint(-2, 3, (len(frames), 2)).float()
    transform = torch.eye(2, 3).repeat(len(frames), 1, 1)
    transform[:, :, 2] = -2 * offset / grid
    return resample_event_counts(frames, transform), targets + offset * (SIZE / grid)


def fit(recordings, epochs, seed, model_args=None):
    torch.manual_seed(seed)
    model = GestureSNN(**(MODEL_ARGS if model_args is None else model_args))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    frames = torch.cat([recording.frames for recording in recordings])
    targets = torch.cat([recording.targets for recording in recordings])
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for indices in torch.randperm(len(frames)).split(16):
            inputs, coordinates = translate(frames[indices], targets[indices])
            expected = (coordinates - COORDINATE_OFFSET) / COORDINATE_SCALE
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inputs), expected)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            print(f"  Epoch {epoch + 1:3}/{epochs}: train MSE {total_loss / len(frames):.4f}", flush=True)
    return model.eval()


@torch.inference_mode()
def predict(model, frames):
    return model(frames) * COORDINATE_SCALE + COORDINATE_OFFSET


def measure(errors):
    if not len(errors):
        return {"forecasts": 0}
    values = np.asarray(errors)
    return {
        "forecasts": len(values),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "mean_px": float(values.mean()),
        "max_px": float(values.max()),
        "within_10_px": float((values <= 10).mean()),
        "within_30_px": float((values <= 30).mean()),
    }


def run(args):
    torch.set_num_threads(args.threads)
    model_args = {**MODEL_ARGS, "grid": args.grid, "precision": args.precision,
                  "parameter_format": args.parameter_format}
    styles = load_recordings(args.archive, args.labels, args.grid)
    all_errors = {method: [] for method in METHODS}
    all_snn_errors = []
    folds = []
    for held_out in range(2):
        train = [recordings[1 - held_out] for recordings in styles.values()]
        test = [recordings[held_out] for recordings in styles.values()]
        print(f"Fold {held_out + 1}: training on {sum(len(clip.targets) for clip in train)} targets", flush=True)
        model = fit(train, args.epochs, args.seed + held_out, model_args)
        fold_errors = {method: [] for method in METHODS}
        for clip in test:
            errors = torch.linalg.vector_norm(predict(model, clip.frames) - clip.targets, dim=1)
            all_snn_errors.extend(errors.tolist())
            fold_errors["snn"].extend(errors[clip.comparable].tolist())
            for method, values in clip.baselines.items():
                fold_errors[method].extend(values)
        folds.append({
            "train": [clip.name for clip in train],
            "test": [clip.name for clip in test],
            "metrics": {method: measure(values) for method, values in fold_errors.items()},
        })
        for method, values in fold_errors.items():
            all_errors[method].extend(values)
        print(f"  Held-out SNN, baseline-comparable targets: {folds[-1]['metrics']['snn']}", flush=True)

    metrics = {
        "horizon_ms": WINDOW_US // 1000,
        "input_ms": CONTEXT_STEPS * BIN_US // 1000,
        "grid": args.grid,
        "precision": args.precision,
        "parameter_format": args.parameter_format if args.precision == "qat" else "fp32",
        "epochs": args.epochs,
        "seed": args.seed,
        "folds": folds,
        "baseline_comparison": {method: measure(values) for method, values in all_errors.items()},
        "snn_all_labeled_targets": measure(all_snn_errors),
    }
    print(json.dumps({key: value for key, value in metrics.items() if key != "folds"}, indent=2), flush=True)

    all_recordings = [clip for recordings in styles.values() for clip in recordings]
    print(f"Final fit on all {sum(len(clip.targets) for clip in all_recordings)} targets", flush=True)
    model = fit(all_recordings, args.epochs, args.seed, model_args)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "left-palm-snn.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model_args": model_args,
        "context_steps": CONTEXT_STEPS,
        "horizon_us": WINDOW_US,
        "coordinate_offset": COORDINATE_OFFSET,
        "coordinate_scale": COORDINATE_SCALE,
        "target": ["left_palm_x", "left_palm_y"],
    }, checkpoint)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {checkpoint}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("go2_gestures_s20.zip"))
    parser.add_argument("--labels", type=Path, default=Path("labeling/session20_left_palm.csv"))
    parser.add_argument("--output", type=Path, default=Path(".local/training/left-palm-snn-40-q26"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grid", type=int, choices=(32, 40), default=40)
    parser.add_argument("--precision", choices=("fp32", "qat"), default="qat")
    parser.add_argument("--parameter-format", choices=PARAMETER_FORMATS, default="q2.6",
                        help="Weight and bias encoding in QAT builds")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
