"""Training: float phase → choose formats → QAT phase → fixed-point check → export.

Run through `python -m training.snn train --config ... --run-dir ...` (see __main__.py), or
`python -m training.snn qat --run-dir ...` to redo everything after the float phase from float_best.pt.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from riverops.nn import BatchNorm2d, Conv2d, prepare_ptq, prepare_qat

from .augment import augment as augment_counts
from .config import Config
from .model import build_model
from .quant import Formats, apply_layer_scales, choose_formats, describe_layers, layer_scales


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SplitData:
    """One split kept on the device as float16 event counts; encoded per batch."""

    def __init__(self, x_path: Path, y_path: Path, cfg: Config, device: torch.device):
        counts = np.load(x_path, mmap_mode="r")
        self.x = torch.empty(counts.shape, dtype=torch.float16, device=device)
        for i in range(0, len(counts), 256):     # convert in chunks to keep host memory low
            chunk = np.minimum(counts[i:i + 256], 65504).astype(np.float16)
            self.x[i:i + 256] = torch.from_numpy(chunk).to(device)
        self.y = torch.from_numpy(np.load(y_path)).to(device)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.y)

    def batch(self, index: torch.Tensor, augment: bool):
        """Encoded frames [B, W+T, 2, H, W], labels, W = warm-start prefix frames (0 unless augmenting),
        and mix = (permutation, lambda) if mixup blended this batch with itself, else None."""
        counts, warmup, mix = self.x[index].float(), 0, None
        if augment:
            counts, warmup = augment_counts(counts, self.cfg)
            m = self.cfg.augment.mixup
            if m.p and torch.rand(()) < m.p:
                lam = float(torch.distributions.Beta(m.alpha, m.alpha).sample())
                perm = torch.randperm(len(counts), device=counts.device)
                counts = lam * counts + (1 - lam) * counts[perm]
                mix = (perm, lam)
        return encode(counts, self.cfg), self.y[index], warmup, mix


def encode(counts: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.data.encoding == "mean":
        return counts / cfg.data.downsampling ** 2
    if cfg.data.encoding == "binary":
        return (counts >= 0.5).float()      # == counts > 0 on real (integer) counts; augmentation can make fractions
    return counts


def spike_targets(labels: torch.Tensor, time_steps: int, n_classes: int, cfg: Config) -> torch.Tensor:
    targets = torch.full((len(labels), n_classes), cfg.train.target_low * time_steps, device=labels.device)
    targets[torch.arange(len(labels), device=labels.device), labels] = cfg.train.target_high * time_steps
    return targets


def spike_count_loss(scores: torch.Tensor, targets: torch.Tensor, time_steps: int) -> torch.Tensor:
    return (scores - targets).square().mean(dim=1).mean() / time_steps


def clamp_parameters(model: nn.Module, limit: float) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (Conv2d, BatchNorm2d)):
                for p in (module.weight, getattr(module, "bias", None)):
                    if p is not None:
                        p.clamp_(-limit, limit)


def check_finite(model: nn.Module, where: str) -> None:
    """NaN weights don't show in the loss (a NaN potential never crosses the threshold → silent network)."""
    bad = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise FloatingPointError(f"non-finite parameters after {where}: {', '.join(bad)}; "
                                 "see 'NaN during QAT' in training/README.md")


def run_epoch(model, data: SplitData, cfg: Config, optimizer=None, clamp: float | None = None,
              augment: bool = True, after_step=None) -> dict:
    training = optimizer is not None
    model.train(training)
    T, n = cfg.data.time_steps, cfg.n_classes
    order = torch.randperm(len(data), device=data.y.device) if training else torch.arange(len(data), device=data.y.device)
    total_loss, correct, spikes = 0.0, 0, 0.0
    confusion = torch.zeros(n, n, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for index in order.split(cfg.train.batch_size):
            x, y, warmup, mix = data.batch(index, augment=training and augment)
            scores = model(x, warmup=warmup)
            targets = spike_targets(y, T, n, cfg)
            if mix is not None:                     # mixup: blend the spike-count targets like the inputs
                perm, lam = mix
                targets = lam * targets + (1 - lam) * targets[perm]
            loss = spike_count_loss(scores, targets, T)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss is {loss.item()} ({'training' if training else 'evaluation'}); "
                                         "see 'NaN during QAT' in training/README.md")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                if clamp:
                    clamp_parameters(model, clamp)
                if after_step:
                    after_step()
            predicted = scores.argmax(dim=1)
            total_loss += loss.item() * len(y)
            spikes += scores.detach().sum().item()
            correct += (predicted == y).sum().item()
            for t, p in zip(y.tolist(), predicted.tolist()):
                confusion[t, p] += 1
    if training:
        check_finite(model, "a training epoch")
    # spikes: mean output spikes per clip and class; 0 means the network is silent
    return {"loss": total_loss / len(data), "acc": correct / len(data), "confusion": confusion.tolist(),
            "spikes": spikes / len(data) / n}


def lr_schedule(cfg: Config):
    """epoch (1-based) → learning rate: cosine from train.lr to train.lr_min over train.cosine_epochs
    (default: epochs - average_last), then constant lr_min. None if train.lr_min is not set."""
    t = cfg.train
    if t.lr_min is None:
        return None
    n = t.cosine_epochs or max(1, t.epochs - t.average_last)
    return lambda epoch: t.lr_min + (t.lr - t.lr_min) * 0.5 * (1 + math.cos(math.pi * min(epoch - 1, n) / n))


def fit(model, train: SplitData, val: SplitData, cfg: Config, *, epochs: int, lr: float, phase: str,
        clamp: float | None, history: list, run_dir: Path, augment: bool = True, after_step=None,
        schedule=None, average_last: int = 0):
    """Train; return the model with the best validation accuracy, or with average_last > 0 the average
    of the weights of the last average_last epochs (no selection on validation)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.train.weight_decay)
    best_acc, best_state = -1.0, copy.deepcopy(model.state_dict())
    average, averaged = None, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        if schedule:
            for group in optimizer.param_groups:
                group["lr"] = schedule(epoch)
        tr = run_epoch(model, train, cfg, optimizer, clamp, augment=augment, after_step=after_step)
        va = run_epoch(model, val, cfg)
        improved = va["acc"] > best_acc
        if improved:
            best_acc, best_state = va["acc"], copy.deepcopy(model.state_dict())
        if average_last and epoch > epochs - average_last:
            state = model.state_dict()
            if average is None:
                average = {k: v.detach().clone().float() if v.is_floating_point() else v.clone() for k, v in state.items()}
            else:
                for k, v in state.items():
                    average[k] = average[k] + v.float() if v.is_floating_point() else v.clone()
            averaged += 1
        history.append({"phase": phase, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": tr["loss"], "train_acc": tr["acc"],
                        "val_loss": va["loss"], "val_acc": va["acc"], "val_spikes": va["spikes"],
                        "seconds": time.time() - start})
        print(f"[{phase}] epoch {epoch:3d}/{epochs}  lr {optimizer.param_groups[0]['lr']:.2e}  "
              f"train loss {tr['loss']:.4f} acc {tr['acc']:.1%}  "
              f"val loss {va['loss']:.4f} acc {va['acc']:.1%} spikes {va['spikes']:.1f}  ({time.time() - start:.0f}s)"
              f"{'  *best' if improved else ''}", flush=True)
        write_history(history, run_dir)
    if average is not None:
        reference = model.state_dict()
        model.load_state_dict({k: (v / averaged).to(reference[k].dtype) if reference[k].is_floating_point() else v
                               for k, v in average.items()})
        va = run_epoch(model, val, cfg)
        print(f"[{phase}] average of the last {averaged} epochs: val acc {va['acc']:.1%} "
              f"(best single epoch was {best_acc:.1%}; the average is used)", flush=True)
        return model, va["acc"]
    model.load_state_dict(best_state)
    return model, best_acc


def write_history(history: list, run_dir: Path) -> None:
    fields = list(dict.fromkeys(k for row in history for k in row))   # older rows may lack newer columns
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(path: Path, model, cfg: Config, formats: Formats | None, phase: str) -> None:
    # Thresholds are plain attributes, not in the state_dict; per-layer scaling changes them.
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(), "phase": phase,
                "formats": formats.to_dict() if formats else None,
                "thresholds": [float(layer.lif.v_threshold) for layer in model.net]}, path)


def load_qat_model(path: Path, cfg: Config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    formats = Formats(**checkpoint["formats"])
    model = prepare_qat(build_model(cfg), formats.riverops(), fused_bn=True)
    model.load_state_dict(checkpoint["state_dict"])
    for layer, threshold in zip(model.net, checkpoint.get("thresholds") or []):
        layer.lif.v_threshold = threshold
    return model.eval(), formats


def zero_tiny_gammas(model: nn.Module, limit: float = 1e-6) -> int:
    """Set every BatchNorm scale gamma with |gamma| < limit to exactly 0; returns how many.

    Fusion-aware QAT divides by the folded scale gamma / sqrt(var + eps), and its gradient by the scale
    squared, which overflows (inf → NaN in all weights after gradient clipping) once gamma is ~1e-13.
    It happens to channels whose input never changes (running var ~0): the loss gives their gamma no
    gradient, so weight decay shrinks it smoothly towards 0, during float training (300 epochs: 32 of
    256 layer-3 gammas at ~5e-41) and during QAT. Their output is the BN bias whatever gamma is, so 0
    changes nothing, and riverops handles an exact 0 scale as a special case. Called before QAT and
    after every QAT step (a gamma may only get small during QAT).
    """
    zeroed = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) and module.weight is not None:
                tiny = (module.weight.abs() < limit) & (module.weight != 0)
                if tiny.any():
                    module.weight[tiny] = 0.0
                    zeroed += int(tiny.sum())
    return zeroed


def train(cfg: Config, cache: dict, run_dir: Path) -> dict:
    """Float training, then quantize() (formats → QAT → fixed-point check)."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.train.device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
          flush=True)
    if cfg.augment.active() or cfg.data.max_shift:
        print(f"augmentation: {', '.join(cfg.augment.active() + [f'shift +-{cfg.data.max_shift}px'])}"
              f"{'' if cfg.augment.qat else ' (float phase only)'}", flush=True)
    splits = {name: SplitData(*paths, cfg, device) for name, paths in cache.items()}
    history: list[dict] = []

    # 1. Float training (notebook recipe).
    model = build_model(cfg, dropout=True).to(device)
    model, _ = fit(model, splits["train"], splits["val"], cfg, epochs=cfg.train.epochs, lr=cfg.train.lr,
                   phase="float", clamp=cfg.train.weight_clamp, history=history, run_dir=run_dir,
                   schedule=lr_schedule(cfg), average_last=cfg.train.average_last)
    save_checkpoint(run_dir / "float_best.pt", model, cfg, None, "float")
    return quantize(model, cfg, splits, run_dir, history)


def qat_from_float(cfg: Config, cache: dict, run_dir: Path) -> dict:
    """Redo formats → QAT → fixed-point check from run_dir/float_best.pt, keeping the float history."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.train.device)
    splits = {name: SplitData(*paths, cfg, device) for name, paths in cache.items()}
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "float_best.pt", map_location="cpu", weights_only=True)["state_dict"])
    history = []
    if (run_dir / "history.csv").exists():
        with open(run_dir / "history.csv", newline="") as f:
            history = [row for row in csv.DictReader(f) if row["phase"] == "float"]
    return quantize(model.to(device), cfg, splits, run_dir, history)


def quantize(model, cfg: Config, splits: dict, run_dir: Path, history: list) -> dict:
    device = next(model.parameters()).device
    plain = build_model(cfg).to(device)            # same weights, without the training-only dropout layers
    plain.load_state_dict(model.state_dict())
    model = plain.eval()
    float_val, float_test = run_epoch(model, splits["val"], cfg), run_epoch(model, splits["test"], cfg)

    # 2. Choose the number formats from the BN-folded parameters.
    zeroed = zero_tiny_gammas(model)
    if zeroed:
        print(f"zeroed {zeroed} BatchNorm gamma(s) with |gamma| < 1e-6 (dead channels) before QAT", flush=True)
    formats, format_info = choose_formats(model.cpu(), cfg)
    scales, scale_info = layer_scales(model, formats, cfg)
    apply_layer_scales(model, scales, cfg.model.neuron.v_threshold)
    model.to(device)
    print(f"formats: {formats.describe()}  ({ {k: v for k, v in format_info.items() if k != 'layers'} })", flush=True)
    print(describe_layers(format_info["layers"]), flush=True)
    print(f"per-layer scale k (weights, bias, threshold / k): {scales}  "
          f"{[(s['need'], s['fits']) for s in scale_info]}", flush=True)

    # 3. Quantization-aware fine-tuning with BN folded inside the fake quantization,
    #    so training sees exactly the clipped, rounded weights the FPGA will get.
    qat = prepare_qat(model, formats.riverops(), fused_bn=True).to(device)
    if cfg.quant.qat_epochs:
        qat, _ = fit(qat, splits["train"], splits["val"], cfg, epochs=cfg.quant.qat_epochs, lr=cfg.quant.qat_lr,
                     phase="qat", clamp=None, history=history, run_dir=run_dir, augment=cfg.augment.qat,
                     after_step=lambda: zero_tiny_gammas(qat))
    save_checkpoint(run_dir / "qat_best.pt", qat, cfg, formats, "qat")
    qat_val, qat_test = run_epoch(qat, splits["val"], cfg), run_epoch(qat, splits["test"], cfg)

    # 4. Fixed-point emulation of the deployed model (folded BN, integer-grid arithmetic).
    fixed = prepare_ptq(qat, formats.riverops(qat=False), fuse=True).to(device)
    fixed_val, fixed_test = run_epoch(fixed, splits["val"], cfg), run_epoch(fixed, splits["test"], cfg)

    metrics = {
        "classes": cfg.data.classes,
        "augment": cfg.augment.active(),
        "formats": formats.to_dict() | {"description": formats.describe()},
        "format_choice": format_info,
        "layer_scales": scale_info,
        "thresholds": [float(layer.lif.v_threshold) for layer in qat.net],
        "dead_batchnorm_channels": zeroed,
        "float": {"val_acc": float_val["acc"], "test_acc": float_test["acc"]},
        "qat": {"val_acc": qat_val["acc"], "test_acc": qat_test["acc"]},
        "fixed_point": {"val_acc": fixed_val["acc"], "test_acc": fixed_test["acc"],
                        "test_mean_output_spikes": fixed_test["spikes"],
                        "test_confusion": fixed_test["confusion"]},
        "clips": {name: len(s) for name, s in splits.items()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("accuracy (val / test):\n"
          f"  float        {float_val['acc']:.1%} / {float_test['acc']:.1%}\n"
          f"  QAT          {qat_val['acc']:.1%} / {qat_test['acc']:.1%}\n"
          f"  fixed point  {fixed_val['acc']:.1%} / {fixed_test['acc']:.1%}   ← closest to the FPGA", flush=True)
    return metrics
