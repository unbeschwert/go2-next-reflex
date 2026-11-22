"""Train and evaluate the Motion Buddy C-SiLIF classifier."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.snn.config import load_config
from training.snn.data import describe, discover

from .data import EventDataset, collate_events
from .model import ReactClassifier


def summarize_confusion(confusion):
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    per_class_f1 = 2 * true_positive / (confusion.sum(dim=0) + support).clamp_min(1)
    return {
        "accuracy": true_positive.sum().item() / support.sum().item(),
        "macro_f1": per_class_f1.mean().item(),
        "recall": [tp / n if n else None for tp, n in zip(true_positive.tolist(), support.tolist())],
        "confusion": confusion.tolist(),
        "clips": support.sum().item(),
    }


def validation_fold(data_config, subjects):
    eligible = sorted(set(data_config.train) | set(data_config.val))
    if len(set(subjects)) != len(subjects) or not set(subjects) < set(eligible):
        raise ValueError("validation subjects must be unique and leave some training subjects, "
                         "using only people in the configured train and validation splits")
    train = [person for person in eligible if person not in subjects]
    return replace(data_config, train=train, val=subjects)


def run_epoch(model, loader, device, loss_fn, classes, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        people = None
        matrices = torch.zeros((1, classes, classes), dtype=torch.long, device=device)
    else:
        # Evaluation loaders follow the order of their clip list.
        people = sorted({clip.person for clip in loader.dataset.clips})
        person_to_index = {person: index for index, person in enumerate(people)}
        person_indices = torch.tensor([person_to_index[clip.person] for clip in loader.dataset.clips],
                                      device=device)
        matrices = torch.zeros((len(people), classes, classes), dtype=torch.long, device=device)
    loss_total = torch.zeros((), device=device)
    count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            tokens, elapsed, valid, labels = (item.to(device, non_blocking=True) for item in batch)
            scores = model(tokens, elapsed, valid)
            loss = loss_fn(scores, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()

            pairs = classes * labels + scores.detach().argmax(dim=1)
            if people is not None:
                pairs += classes * classes * person_indices[count:count + labels.numel()]
            matrices += torch.bincount(pairs, minlength=matrices.numel()).reshape_as(matrices)
            loss_total += loss.detach() * labels.numel()
            count += labels.numel()

    matrices = matrices.cpu()
    result = summarize_confusion(matrices.sum(dim=0))
    result["loss"] = loss_total.item() / count
    if people is not None:
        result["per_person"] = {
            person: summarize_confusion(matrix) for person, matrix in zip(people, matrices)
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/motion_buddy_40.yaml",
                        help="read the dataset path, subject splits, class order, and downsampling")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=2048)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--spatial-augment", action="store_true",
                        help="apply a random translation and zoom to training clips")
    parser.add_argument("--val-subjects", nargs="+",
                        help="hold out these development subjects and train on the others")
    parser.add_argument("--skip-test", action="store_true",
                        help="leave the test recordings for the final evaluation")
    parser.add_argument("--check-position-shift", action="store_true",
                        help="replay the best checkpoint on validation clips shifted left and right")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width, args.blocks) < 1 or args.max_events < 2:
        parser.error("epochs, batch size, width, and blocks must be positive and max-events must be at least 2")
    if args.workers < 0 or (args.limit_per_split is not None and args.limit_per_split < 1):
        parser.error("workers cannot be negative and limit-per-split must be positive")
    if not 0 < args.lr < float("inf"):
        parser.error("learning rate must be positive and finite")

    config = load_config(args.config)
    if args.val_subjects:
        try:
            config = replace(config, data=validation_fold(config.data, args.val_subjects))
        except ValueError as exc:
            parser.error(str(exc))
    if args.limit_per_split is not None:
        config = replace(config, data=replace(config.data, limit_per_split=args.limit_per_split))
    splits = discover(config)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but PyTorch cannot see a CUDA device")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    torch.manual_seed(config.seed)
    model = ReactClassifier(config.data.size, config.n_classes, args.width, args.blocks).to(device)
    dynamics, weights = [], []
    for name, parameter in model.named_parameters():
        group = dynamics if ".cell." in name else weights
        group.append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": weights}, {"params": dynamics, "weight_decay": 0.0}],
        lr=args.lr, weight_decay=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    def loader_for(split, shift_x=0):
        dataset = EventDataset(splits[split], config.data, args.max_events,
                               spatial_augment=split == "train" and args.spatial_augment,
                               shift_x=shift_x)
        return DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=split == "train", collate_fn=collate_events, num_workers=args.workers,
            persistent_workers=args.workers > 0, pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config.seed),
        )

    train_loader, val_loader = loader_for("train"), loader_for("val")

    args.run_dir.mkdir(parents=True, exist_ok=False)
    options = vars(args).copy()
    del options["run_dir"]
    options["device"] = str(device)
    (args.run_dir / "config.json").write_text(json.dumps({
        "options": options,
        "seed": config.seed,
        "classes": config.data.classes,
        "size": config.data.size,
        "clip_us": config.data.clip_us,
        "subjects": {split: sorted({clip.person for clip in clips}) for split, clips in splits.items()},
    }, indent=2) + "\n")
    print(describe(splits, config.data.classes), flush=True)
    print(f"{device}, {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    history = []
    checkpoint = args.run_dir / "best.pt"
    best = None
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, device, loss_fn, config.n_classes, optimizer)
        val = run_epoch(model, val_loader, device, loss_fn, config.n_classes)
        history.append({"epoch": epoch, "train": train, "val": val})
        if best is None or val["macro_f1"] > best["val"]["macro_f1"]:
            best = history[-1]
            torch.save(model.state_dict(), checkpoint)
        print(f"epoch {epoch:3d}  train acc {train['accuracy']:.1%}  "
              f"val acc {val['accuracy']:.1%}  val macro F1 {val['macro_f1']:.3f}", flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    results = {
        "classes": config.data.classes, "best_epoch": best["epoch"], "val": best["val"],
        "history": history,
    }
    if args.check_position_shift:
        results["position_shift"] = {
            shift: run_epoch(model, loader_for("val", shift), device, loss_fn, config.n_classes)
            for shift in (-4, 4)
        }
        for shift, metrics in results["position_shift"].items():
            print(f"val shift {shift:+d} cells  acc {metrics['accuracy']:.1%}  "
                  f"macro F1 {metrics['macro_f1']:.3f}", flush=True)
    if not args.skip_test:
        test = run_epoch(model, loader_for("test"), device, loss_fn, config.n_classes)
        results["test"] = test
        print(f"test acc {test['accuracy']:.1%}  test macro F1 {test['macro_f1']:.3f}", flush=True)
    (args.run_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"best epoch {best['epoch']}  val acc {best['val']['accuracy']:.1%}  "
          f"val macro F1 {best['val']['macro_f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
