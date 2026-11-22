"""Recompute the BatchNorm running statistics of a finished run's float model, in place.

For runs trained with weight averaging (train.average_last) before train.recompute_bn existed: their
float_best.pt holds the averaged weights with the *averaged* BatchNorm statistics. This replaces those
statistics with ones computed for the averaged weights (one pass over the training data, exactly as
training now does), and prints val accuracy before and after. Afterwards run `remote/qat.sh <run>` to
redo quantization, the fixed-point check and the export from the updated float model.

Work on a copy of the run directory if you want to keep the original for comparison.

    python scripts/recompute_bn.py runs/<run>
"""
import argparse
from pathlib import Path

import torch

from training.snn import data as data_mod
from training.snn.config import load_config
from training.snn.model import build_model
from training.snn.train import SplitData, pick_device, recompute_bn, run_epoch, save_checkpoint, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.run_dir / "config.yaml")
    seed_everything(cfg.seed)
    device = pick_device(cfg.train.device)
    cache = data_mod.build_cache(cfg, data_mod.discover(cfg))
    train, val = (SplitData(*cache[name], cfg, device) for name in ("train", "val"))

    # The model as it was trained (with its training-only dropout layers), so the statistics are
    # computed exactly as in fit(); the parameter names are the same as the plain model's.
    model = build_model(cfg, dropout=True).to(device)
    path = args.run_dir / "float_best.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])

    before = run_epoch(model, val, cfg)["acc"]
    layers = recompute_bn(model, train, cfg)
    after = run_epoch(model, val, cfg)["acc"]
    save_checkpoint(path, model, cfg, None, "float")
    print(f"{args.run_dir.name}: BatchNorm statistics recomputed for {layers} layers; "
          f"float val {before:.1%} (averaged statistics) → {after:.1%} (recomputed); saved {path}")


if __name__ == "__main__":
    main()
