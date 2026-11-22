"""Command line for the SNN training pipeline. Run from the repository root:

    python -m training.snn check   --config training/configs/motion_buddy_40.yaml
    python -m training.snn prepare --config ...            # build the event-count cache
    python -m training.snn train   --config ... --run-dir runs/NAME   # prepare + check + train + export
                                   [--set train.epochs=20 --set seed=43 ...]   # (check/prepare too)
    python -m training.snn qat     --run-dir runs/NAME [--set quant.layer_scaling=true ...]
                                                           # redo formats + QAT + export from float_best.pt
    python -m training.snn export  --run-dir runs/NAME     # re-export from runs/NAME/qat_best.pt

`check` and `prepare` need only numpy/pyyaml; `train` and `export` need the `train` dependency group
(torch + riverops), i.e. `uv run --group train ...`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import data as data_mod
from . import fpga
from .config import ConfigError, load_config, save_config
from .model import describe as describe_model
from .model import layer_shapes


def check(cfg) -> dict:
    print(f"config '{cfg.name}': input {cfg.data.size}x{cfg.data.size}, {cfg.data.time_steps} steps of "
          f"{cfg.data.time_window_us / 1000:g} ms, encoding {cfg.data.encoding}, {cfg.n_classes} classes", flush=True)
    print(describe_model(layer_shapes(cfg)), flush=True)
    report = fpga.estimate(cfg)
    print(f"FPGA memory (weights {cfg.quant.weight_bits} bit, potentials {cfg.quant.potential_bits} bit):", flush=True)
    print(fpga.report_text(report), flush=True)
    widths = ", ".join(f"{w}-bit weights → {b:.1f}" for w, b in fpga.compare_weight_widths(cfg))
    print(f"  BRAM-only cost by weight width: {widths}", flush=True)
    return report


def apply_overrides(cfg, overrides: list[str]):
    """'quant.layer_scaling=true' → set cfg.quant.layer_scaling = True (value parsed as YAML), validated."""
    import yaml
    from .config import Config, _build, validate
    raw = cfg.to_dict()
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        node, parts = raw, key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ConfigError(f"--set: no config section {key!r}")
            node = node[part]
        node[parts[-1]] = yaml.safe_load(value)
    new = _build(Config, raw, "config")
    validate(new)
    return new


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.snn", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "train"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                       help="override a config value (parsed as YAML); a run saves the result in its config.yaml")
        if name in ("prepare", "train"):
            p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
        if name == "train":
            p.add_argument("--run-dir", required=True)
    p = sub.add_parser("export")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("qat")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override the run's config (YAML value), saved back to the run's config.yaml")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = parser.parse_args(argv)

    try:
        if args.command == "export":
            run_dir = Path(args.run_dir)
            cfg = load_config(run_dir / "config.yaml")
            from .export import export_deployment
            from .train import load_qat_model
            model, formats = load_qat_model(run_dir / "qat_best.pt", cfg)
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
            export_deployment(model, formats, cfg, run_dir, metrics)
            return 0

        if args.command == "qat":
            run_dir = Path(args.run_dir)
            cfg = load_config(run_dir / "config.yaml")
            if args.set:
                cfg = apply_overrides(cfg, args.set)
                save_config(cfg, run_dir / "config.yaml")
                print(f"config overrides: {', '.join(args.set)}", flush=True)
            start = time.time()
            cache = data_mod.build_cache(cfg, data_mod.discover(cfg), workers=args.workers)
            from .export import export_deployment
            from .train import load_qat_model, qat_from_float
            metrics = qat_from_float(cfg, {k: cache[k] for k in data_mod.SPLITS}, run_dir)
            model, formats = load_qat_model(run_dir / "qat_best.pt", cfg)
            export_deployment(model, formats, cfg, run_dir, metrics)
            print(f"done in {(time.time() - start) / 60:.1f} min → {run_dir}/deploy/DEPLOY.md", flush=True)
            return 0

        cfg = load_config(args.config)
        if args.set:
            cfg = apply_overrides(cfg, args.set)
            print(f"config overrides: {', '.join(args.set)}", flush=True)
        report = check(cfg)
        if args.command == "check":
            return 0 if report["verdict"] != "too_big" else 1
        if report["verdict"] == "too_big" and cfg.fpga.fail_on_overflow:
            print("model does not fit the FPGA memory budget; not training (fpga.fail_on_overflow)", file=sys.stderr)
            return 1

        splits = data_mod.discover(cfg)
        print("data:\n" + data_mod.describe(splits, cfg.data.classes), flush=True)
        start = time.time()
        cache = data_mod.build_cache(cfg, splits, workers=args.workers)
        print(f"cache ready ({time.time() - start:.0f}s)", flush=True)
        if args.command == "prepare":
            return 0

        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.yaml")
        shutil.copy(args.config, run_dir / "config.original.yaml")
        (run_dir / "run_info.json").write_text(json.dumps(
            {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": args.config, "overrides": args.set,
             "git": git_info(), "host": os.uname().nodename}, indent=2) + "\n")

        from .export import export_deployment
        from .train import load_qat_model, train
        metrics = train(cfg, {k: cache[k] for k in data_mod.SPLITS}, run_dir)
        model, formats = load_qat_model(run_dir / "qat_best.pt", cfg)
        export_deployment(model, formats, cfg, run_dir, metrics)
        print(f"done in {(time.time() - start) / 60:.1f} min → {run_dir}/deploy/DEPLOY.md", flush=True)
        return 0
    except (ConfigError, FileNotFoundError, ValueError, FloatingPointError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
