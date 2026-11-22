"""Compare left-palm weight encodings using the published 40x40 folds."""

import argparse
import copy
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from training import model as model_module
from training import train_left_palm as trainer

REFERENCE = Path("models/left-palm-40-metrics.json")
SCALE = 1.0 / 64


def fixed_q26(values):
    return torch.fake_quantize_per_tensor_affine(values, SCALE, 0, -128, 127)


class Q26Bias(nn.Module):
    def forward(self, bias):
        return fixed_q26(bias)


def use_q26_bias(model):
    for layer in (*model.convolutions, model.classifier):
        parametrize.register_parametrization(layer, "bias", Q26Bias())
    return model


class Q26SNN(model_module.GestureSNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        use_q26_bias(self)


def score(model, clips, fixed):
    context = patch.object(model_module, "quantized_weight", fixed_q26) if fixed else nullcontext()
    with context:
        errors = [torch.linalg.vector_norm(trainer.predict(model, clip.frames) - clip.targets, dim=1).numpy()
                  for clip in clips]
    return np.concatenate(errors), np.concatenate([clip.comparable.numpy() for clip in clips])


def summary(errors):
    return {**trainer.measure(errors), "count_within_30_px": int((errors <= 30).sum())}


def compare(archive, labels, output_path):
    torch.set_num_threads(4)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    styles = trainer.load_recordings(archive, labels, grid=40)
    by_name = {clip.name: clip for recordings in styles.values() for clip in recordings}
    cases = ("per_channel_int8_qat", "q2_6_post_training_conversion", "q2_6_qat")
    all_errors = {case: [] for case in cases}
    masks = []
    folds = []
    model_args = {**trainer.MODEL_ARGS, "grid": 40, "precision": "qat"}

    for index, fold in enumerate(reference["folds"]):
        train = [by_name[name] for name in fold["train"]]
        test = [by_name[name] for name in fold["test"]]
        seed = reference["seed"] + index
        print(f"Fold {index + 1} / per-channel baseline", flush=True)
        baseline = trainer.fit(train, reference["epochs"], seed, model_args)
        original, comparable = score(baseline, test, fixed=False)
        converted, _ = score(use_q26_bias(copy.deepcopy(baseline)), test, fixed=True)
        print(f"Fold {index + 1} / Q2.6-aware training", flush=True)
        with patch.object(trainer, "GestureSNN", Q26SNN), patch.object(model_module, "quantized_weight", fixed_q26):
            fixed = trainer.fit(train, reference["epochs"], seed, model_args)
        retrained, _ = score(fixed, test, fixed=True)
        masks.append(comparable)
        errors = dict(zip(cases, (original, converted, retrained)))
        for case, values in errors.items():
            all_errors[case].append(values)
        folds.append({"train": fold["train"], "test": fold["test"],
                      "metrics": {case: summary(values) for case, values in errors.items()}})
        print(f"Fold {index + 1} results: {folds[-1]['metrics']}", flush=True)

    comparable = np.concatenate(masks)
    all_errors = {case: np.concatenate(values) for case, values in all_errors.items()}
    signed_range = [-128 * SCALE, 127 * SCALE]
    output = {
        "grid": 40, "horizon_ms": 50, "q2_6_scale": SCALE, "q2_6_range": signed_range,
        "q2_6_parameters": ["convolution weights", "convolution biases", "readout weights", "readout biases"],
        "input_and_membrane": "unchanged software simulation",
        "all_labeled_targets": {case: summary(values) for case, values in all_errors.items()},
        "baseline_comparable_targets": {case: summary(values[comparable]) for case, values in all_errors.items()},
        "folds": folds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "folds"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=Path("go2_gestures_s20.zip"), type=Path)
    parser.add_argument("--labels", default=Path("labeling/session20_left_palm.csv"), type=Path)
    parser.add_argument("--output", default=Path(".local/training/left-palm-snn-40-qat/q26-comparison.json"), type=Path)
    args = parser.parse_args()
    compare(args.archive, args.labels, args.output)
