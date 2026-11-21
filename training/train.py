"""Train on the existing gesture clips and evaluate held-out recording groups."""

import argparse
import json
from math import radians
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from training.data import CLASSES, CLASSES_WITHOUT_DANCE, COMMAND_CLASSES, MAX_COUNT, SPLIT_NAMES, TICKS_PER_STEP, input_shape
from training.model import EVENT_ENCODINGS, GestureSNN, MODEL_WIDTHS


def resample_event_counts(frames, transform):
    # Treat time and polarity as channels so every plane uses the same sampling grid.
    planes = frames.flatten(1, 2).float()
    grid = F.affine_grid(transform, planes.shape, align_corners=False)
    return F.grid_sample(planes, grid, align_corners=False).round().to(frames.dtype).view_as(frames)


def center_event_counts(frames):
    """Align the main activity region while preserving the event grid's aspect ratio."""
    centered = torch.empty_like(frames)
    resolution = frames.shape[-1]
    for start in range(0, len(frames), 128):
        block = frames[start:start + 128]
        activity = block.sum(dim=(1, 2)).float()
        histograms = torch.stack((activity.sum(dim=1), activity.sum(dim=2)), dim=1)
        cdf = histograms.cumsum(dim=-1)
        mass = cdf[..., -1:]
        low = (cdf < 0.06 * mass).sum(dim=-1)
        high = (cdf < 0.94 * mass).sum(dim=-1)
        scale = (high - low + 5).amax(dim=1).clamp(14, 40).float() / resolution
        offset = (low + high + 1).float() / resolution - 1
        offset[mass[:, 0, 0] == 0] = 0
        transform = torch.zeros(len(block), 2, 3)
        transform[:, 0, 0] = transform[:, 1, 1] = scale
        transform[:, :, 2] = offset
        centered[start:start + len(block)] = resample_event_counts(block, transform)
    return centered


def load_splits(cache, center_events, dance_policy="keep"):
    with np.load(cache, allow_pickle=False) as data:
        classes = tuple(data["classes"].tolist())
        frames, labels = data["frames"], data["labels"]
        ticks_per_step = int(data["ticks_per_step"])
        if classes not in (CLASSES, COMMAND_CLASSES) or frames.shape[1:] != input_shape(ticks_per_step):
            raise ValueError("The event cache does not match this model's classes or input encoding.")
        if "splits" in data:
            assignments = data["splits"]
            masks = tuple(assignments == name for name in SPLIT_NAMES)
        else:
            groups = data["groups"]
            masks = (groups <= 10, (groups >= 11) & (groups <= 12), groups >= 13)
        fingerprint = str(data["archive_fingerprint"])
        source = str(data["source"]) if "source" in data else "recorded"
    if dance_policy != "keep":
        if classes != CLASSES:
            raise ValueError("The dance ablation requires the five-class gesture archive.")
        non_dance = labels != classes.index("dance")
        # Both four-class runs select and score their checkpoints on the same clips.
        train_mask, validation_mask, test_mask = masks
        if dance_policy == "drop":
            train_mask = train_mask & non_dance
        masks = (train_mask, validation_mask & non_dance, test_mask & non_dance)
        remap = [CLASSES_WITHOUT_DANCE.index("no-action" if name == "dance" else name) for name in classes]
        labels = np.array(remap, dtype=np.int64)[labels]
        classes = CLASSES_WITHOUT_DANCE
    splits = []
    for mask in masks:
        inputs = torch.from_numpy(frames[mask])
        if center_events:
            inputs = center_event_counts(inputs)
        splits.append(TensorDataset(inputs, torch.from_numpy(labels[mask])))
    if not all(len(split) for split in splits):
        raise ValueError("Training, validation, and test must all contain recordings.")
    return splits, fingerprint, classes, source, ticks_per_step


def augment(frames, *, flip=True):
    dx, dy = torch.randint(-2, 3, (2,)).tolist()
    shifted = torch.roll(frames, (dy, dx), dims=(-2, -1))
    if dx > 0:
        shifted[..., :dx] = 0
    elif dx < 0:
        shifted[..., dx:] = 0
    if dy > 0:
        shifted[..., :dy, :] = 0
    elif dy < 0:
        shifted[..., dy:, :] = 0
    if flip and torch.rand(()) < 0.5:
        shifted = shifted.flip(-1)
    return shifted


def vary_timing_and_density(frames):
    """Simulate different gesture speeds and camera event densities during training."""
    batch, steps = frames.shape[:2]
    center = (steps - 1) / 2
    speed = torch.empty(batch, 1).uniform_(0.85, 1.15)
    offset = torch.empty(batch, 1).uniform_(-0.5, 0.5)
    position = ((torch.arange(steps) - center) * speed + center + offset).clamp(0, steps - 1)
    start = position.long()
    end = (start + 1).clamp_max(steps - 1)
    rows = torch.arange(batch).unsqueeze(1)
    fraction = (position - start)[..., None, None, None]
    resampled = torch.lerp(frames[rows, start].float(), frames[rows, end].float(), fraction)
    gain = torch.empty(batch, 1, 1, 1, 1).uniform_(0.6, 1.4)
    return (resampled * gain).round().clamp(0, MAX_COUNT).to(frames.dtype)


def augment_spatially(frames, mode, generator):
    """Crop and pad by a small amount, optionally rotating and shearing the clip."""
    batch = len(frames)
    zoom = torch.empty(batch).uniform_(0.88, 1.10, generator=generator)
    offset = torch.empty(batch, 2).uniform_(-0.06, 0.06, generator=generator)
    transform = torch.zeros(batch, 2, 3)
    transform[:, 0, 0] = transform[:, 1, 1] = zoom
    transform[:, :, 2] = offset
    if mode == "affine":
        angle = torch.empty(batch).uniform_(-radians(8), radians(8), generator=generator)
        shear = torch.empty(batch).uniform_(-0.07, 0.07, generator=generator)
        transform[:, 0, 0] = transform[:, 1, 1] = zoom * angle.cos()
        transform[:, 0, 1] = zoom * (-angle.sin() + shear)
        transform[:, 1, 0] = zoom * angle.sin()
    return resample_event_counts(frames, transform)


@torch.inference_mode()
def evaluate(model, loader, device, classes):
    model.eval()
    n = len(classes)
    confusion = torch.zeros(n * n, dtype=torch.int64)
    for frames, labels in loader:
        predicted = model(frames.to(device)).argmax(dim=1).cpu()
        confusion += torch.bincount(n * labels + predicted, minlength=n * n)
    confusion = confusion.view(n, n)
    correct = confusion.diag()
    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    return {
        "accuracy": (correct.sum() / confusion.sum()).item(),
        "macro_f1": (2 * correct / (actual + predicted).clamp_min(1)).mean().item(),
        "recall": dict(zip(classes, (correct / actual.clamp_min(1)).tolist())),
        "confusion_matrix": confusion.tolist(),
    }


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    spatial_generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    splits, fingerprint, classes, source, ticks_per_step = load_splits(args.cache, args.center_events, args.dance_policy)
    datasets = dict(zip(SPLIT_NAMES, splits))
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train")
               for name, dataset in datasets.items()}
    split_sizes = {name: len(dataset) for name, dataset in datasets.items()}
    print(f"Backend {args.backend} | precision {args.precision} | width {args.model_width} | encoding {args.event_encoding} | "
          f"device {device} | data {source} | centered event counts {args.center_events} | "
          f"event augmentation {args.event_augmentation} | spatial augmentation {args.spatial_augmentation} | "
          f"dance policy {args.dance_policy} | bin duration {ticks_per_step} us | "
          f"split sizes {split_sizes}", flush=True)

    if args.backend == "riverops":
        if classes != COMMAND_CLASSES or ticks_per_step != TICKS_PER_STEP:
            raise ValueError("The RiverOps command model expects the four command classes and 100 ms bins.")
        from training.riverops_commands import build_model

        model = build_model().to(device)
    else:
        model = GestureSNN(len(classes), width=args.model_width, precision=args.precision,
                           event_encoding=args.event_encoding, ticks_per_step=ticks_per_step).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {parameters:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0003)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "gesture-snn.pt"
    temporary = checkpoint_path.with_suffix(".tmp")
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        for frames, labels in loaders["train"]:
            frames = augment(frames, flip=classes != COMMAND_CLASSES)
            if args.spatial_augmentation != "none":
                frames = augment_spatially(frames, args.spatial_augmentation, spatial_generator)
            if args.event_augmentation:
                frames = vary_timing_and_density(frames)
            frames = frames.to(device)
            targets = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(frames), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += loss.detach().item() * len(labels)
        schedule.step()
        validation = evaluate(model, loaders["validation"], device, classes)
        loss = epoch_loss / split_sizes["train"]
        history.append({"epoch": epoch, "training_loss": loss, "validation": validation})
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "classes": classes, "input_shape": input_shape(ticks_per_step),
                        "ticks_per_step": ticks_per_step, "epoch": epoch, "archive_fingerprint": fingerprint,
                        "center_events": args.center_events, "source": source, "backend": args.backend,
                        "spatial_augmentation": args.spatial_augmentation, "dance_policy": args.dance_policy,
                        "model_width": args.model_width, "precision": args.precision,
                        "event_encoding": args.event_encoding}, temporary)
            temporary.replace(checkpoint_path)
        print(f"Epoch {epoch:02d} | loss {loss:.3f} | validation accuracy {validation['accuracy']:.1%} | "
              f"macro-F1 {validation['macro_f1']:.3f} | {time.perf_counter() - started:.1f}s", flush=True)

    best = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    test = evaluate(model, loaders["test"], device, classes)
    result = {"classes": classes, "source": source, "backend": args.backend, "device": str(device),
              "model_width": args.model_width, "precision": args.precision, "parameters": parameters,
              "event_encoding": args.event_encoding, "ticks_per_step": ticks_per_step,
              "seed": args.seed, "split_sizes": split_sizes, "event_augmentation": args.event_augmentation,
              "spatial_augmentation": args.spatial_augmentation, "dance_policy": args.dance_policy,
              "archive_fingerprint": fingerprint, "center_events": args.center_events,
              "best_epoch": best["epoch"], "history": history, "test": test}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Held-out test | accuracy {test['accuracy']:.1%} | macro-F1 {test['macro_f1']:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".local/training/gestures.npz"))
    parser.add_argument("--output", type=Path, default=Path(".local/training/run"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("pytorch", "riverops"), default="pytorch")
    parser.add_argument("--precision", choices=("qat", "fp32"), default="qat")
    parser.add_argument("--model-width", choices=MODEL_WIDTHS, default="compact")
    parser.add_argument("--event-encoding", choices=EVENT_ENCODINGS, default="polarity-counts")
    parser.add_argument("--center-events", action="store_true", help="Center and scale each clip by event activity")
    parser.add_argument("--event-augmentation", action="store_true", help="Vary clip timing and event density in training")
    parser.add_argument("--spatial-augmentation", choices=("none", "crop", "affine"), default="none",
                        help="Apply small crops or affine warps consistently across a training clip")
    parser.add_argument("--dance-policy", choices=("keep", "drop", "background"), default="keep",
                        help="Keep dance as a class, drop it, or train dance clips as no-action")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch size, and threads must be positive")
    if args.backend == "riverops" and (args.precision, args.model_width, args.event_encoding) != (
        "qat", "compact", "polarity-counts",
    ):
        parser.error("The precision, width, and encoding options apply to the PyTorch model.")
    train(args)
