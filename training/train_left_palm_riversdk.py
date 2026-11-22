"""Train and export a Q2.6 palm forecaster for main's five-output nb40 graph."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from event_data import SIZE, decode_recording
from labeling.benchmark_left_palm import read_labels
from training.data import bin_events
from training.left_palm_riversdk import BIN_US, FORECAST_WINDOW_US, READOUT_STEPS, coordinates
from training.model import Q26_SCALE, quantized_q26
from training.train import resample_event_counts
from training.train_left_palm import measure


MODEL_PATH = Path("models/left-palm-40-riversdk/model.json")
WARMUP_STEPS = 40
POTENTIAL_SCALE = 1 / 4096
LEAK_CODE = 287
THRESHOLDS = (512, 2048, 2048, 2048, 2048)
REFERENCE_STAGES = ((16, 5, 2, 1), (16, 3, 1, 1), (8, 6, 2, 2), (256, 9, 1, 0), (5, 1, 1, 0))


@dataclass
class Clip:
    name: str
    counts: torch.Tensor
    targets: torch.Tensor
    indices: torch.Tensor


def validate_graph(graph):
    """Require the known nb40 graph before exporting weights for its bitstream."""
    if set(graph) != {"input", "nodes", "edges", "outputs"}:
        raise ValueError("Expected a RiverSDK architecture JSON.")
    if graph["input"] != {"channels": 2, "height": 40, "width": 40}:
        raise ValueError("Expected the nb40 input shape.")
    names = ["conv_lif", *(f"conv_lif_{index}" for index in range(1, len(REFERENCE_STAGES)))]
    expected_nodes = [
        {"node_index": name, "node_type": "ConvLIF2d", "upscaling": 1, "channels": channels,
         "kernel_size": kernel, "stride": stride, "padding": [padding] * 4}
        for name, (channels, kernel, stride, padding) in zip(names, REFERENCE_STAGES, strict=True)
    ]
    expected_edges = [{"source": source, "target": target}
                      for source, target in zip(["input", *names[:-1]], names, strict=True)]
    if graph["nodes"] != expected_nodes or graph["edges"] != expected_edges or graph["outputs"] != names[-1:]:
        raise ValueError("The architecture differs from the known nb40 bitstream graph.")


def load_clips(archive_path, label_path):
    styles = defaultdict(list)
    with ZipFile(archive_path) as archive:
        for name, rows in read_labels(label_path).items():
            events, timestamps = decode_recording(archive.read(name), name)
            counts = bin_events(events, timestamps, BIN_US, grid=40, max_count=None)
            targets, indices = [], []
            for row in rows:
                end = int(row["frame"]) * FORECAST_WINDOW_US // BIN_US
                if int(row["window_start_us"]) != int(timestamps[0]) + end * BIN_US:
                    raise ValueError(f"Annotation timing differs for {name}.")
                if end >= READOUT_STEPS and row["left_palm_x"] and row["left_palm_y"]:
                    targets.append((float(row["left_palm_x"]), float(row["left_palm_y"])))
                    indices.append(WARMUP_STEPS + end - READOUT_STEPS)
            if targets:
                styles[rows[0]["style"]].append(Clip(name, torch.from_numpy(counts).float(),
                                                     torch.tensor(targets), torch.tensor(indices)))
    if not styles or any(len(clips) != 2 for clips in styles.values()):
        raise ValueError("The two-fold benchmark requires two labeled recordings per style.")
    return styles


def membrane_decay(membrane):
    retain = 4096 - LEAK_CODE
    # Match the retain-and-floor mode used by main's integer SPLEAT simulator.
    exact = (torch.floor(membrane.double() * retain) * POTENTIAL_SCALE).float()
    continuous = membrane * (retain * POTENTIAL_SCALE)
    return continuous + (exact - continuous).detach()


class NativePalmSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Conv2d(previous, channels, kernel, stride=stride, padding=padding)
            for previous, (channels, kernel, stride, padding)
            in zip((2, 16, 16, 8, 256), REFERENCE_STAGES, strict=True)
        )
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.layers[-1].bias[:4] = 3 * Q26_SCALE
            self.layers[-1].weight[4].zero_()

    def forward(self, counts):
        activity = counts / 64
        batch, steps = activity.shape[:2]
        for layer, threshold in zip(self.layers, THRESHOLDS, strict=True):
            drive = F.conv2d(activity.flatten(0, 1), quantized_q26(layer.weight), quantized_q26(layer.bias),
                             stride=layer.stride, padding=layer.padding)
            drive = drive.view(batch, steps, *drive.shape[1:])
            membrane = torch.zeros_like(drive[:, 0])
            outputs = []
            for current in drive.unbind(dim=1):
                voltage = (membrane_decay(membrane) + current).clamp(-8, 32767 * POTENTIAL_SCALE)
                relative = voltage - threshold * POTENTIAL_SCALE
                spike = (relative >= 0).to(voltage.dtype)
                membrane = voltage * (1 - spike)
                if self.training:
                    surrogate = (relative + 0.5).clamp(0, 1)
                    spike = spike + (surrogate - surrogate.detach())
                outputs.append(spike)
            activity = torch.stack(outputs, dim=1)
        return activity.flatten(start_dim=2)


def clip_forecasts(model, counts, clips):
    context = F.pad(counts, (0, 0, 0, 0, 0, 0, WARMUP_STEPS, 0))
    rates = model(context).unfold(1, READOUT_STEPS, 1).mean(dim=-1)
    return torch.cat([coordinates(rates[batch, clip.indices]) for batch, clip in enumerate(clips)])


def train(clips, epochs, seed):
    torch.manual_seed(seed)
    model = NativePalmSNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    counts = torch.stack([clip.counts for clip in clips])
    for epoch in range(epochs):
        offset = torch.randint(-2, 3, (len(clips), 2)).float()
        transform = torch.eye(2, 3).repeat(len(clips), 1, 1)
        transform[:, :, 2] = -2 * offset / 40
        inputs = resample_event_counts(counts, transform)
        targets = torch.cat([clip.targets + shift * 8 for clip, shift in zip(clips, offset, strict=True)])
        optimizer.zero_grad(set_to_none=True)
        predictions = clip_forecasts(model, inputs, clips)
        loss = F.mse_loss(predictions / (SIZE / 2), targets / (SIZE / 2))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"  Epoch {epoch + 1:3}/{epochs}: MSE {loss.item():.4f}", flush=True)
    return model.eval()


def export(model, output):
    with torch.no_grad():
        weights = {f"weights_{index}": (quantized_q26(layer.weight).flatten() / Q26_SCALE).int().tolist()
                   for index, layer in enumerate(model.layers)}
        biases = {f"bias_{index}": (quantized_q26(layer.bias).flatten() / Q26_SCALE).int().tolist()
                  for index, layer in enumerate(model.layers)}
    parameters = {**weights, **biases}
    for index, threshold in enumerate(THRESHOLDS):
        parameters[f"threshold_{index}"] = threshold
        parameters[f"leak_{index}"] = LEAK_CODE
    values = [value for item in parameters.values() for value in (item if isinstance(item, list) else [item])]
    (output / "weights.txt").write_bytes("\n".join(map(str, values)).encode("ascii"))


def run(args):
    torch.set_num_threads(args.threads)
    validate_graph(json.loads(args.model.read_text(encoding="utf-8")))
    styles = load_clips(args.archive, args.labels)
    errors, folds = [], []
    for index in range(2):
        train_clips = [clips[1 - index] for clips in styles.values()]
        test_clips = [clips[index] for clips in styles.values()]
        print(f"Fold {index + 1}", flush=True)
        model = train(train_clips, args.epochs, args.seed + index)
        with torch.inference_mode():
            predicted = clip_forecasts(model, torch.stack([clip.counts for clip in test_clips]), test_clips)
            targets = torch.cat([clip.targets for clip in test_clips])
            fold_errors = torch.linalg.vector_norm(predicted - targets, dim=1).tolist()
        errors.extend(fold_errors)
        folds.append({"train": [clip.name for clip in train_clips], "test": [clip.name for clip in test_clips],
                      "metrics": measure(fold_errors)})
        print(folds[-1]["metrics"], flush=True)
    metrics = {"parameter_format": "Q2.6", "potential_format": "Q4.12", "reset": "hard", "bin_us": BIN_US,
               "readout_steps": READOUT_STEPS, "warmup_steps": WARMUP_STEPS, "epochs": args.epochs,
               "seed": args.seed, "folds": folds, "software_metrics": measure(errors)}
    print(metrics["software_metrics"], flush=True)
    print("Refitting on all labeled clips", flush=True)
    final = train([clip for clips in styles.values() for clip in clips], args.epochs, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "model.json").write_bytes(args.model.read_bytes())
    export(final, args.output)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved RiverSDK architecture and SPLEAT weights to {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("go2_gestures_s20.zip"))
    parser.add_argument("--labels", type=Path, default=Path("labeling/session20_left_palm.csv"))
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path(".local/training/left-palm-40-riversdk"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")
    run(args)
