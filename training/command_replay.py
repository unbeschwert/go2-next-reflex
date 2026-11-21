"""Decode four SNN scores into a movement command without driving a robot."""

import argparse
import json
from math import isfinite
from pathlib import Path

from training.data import COMMAND_CLASSES


def select_command(scores, minimum_margin):
    if len(scores) != len(COMMAND_CLASSES) or not all(isfinite(score) for score in scores):
        raise ValueError("Expected four finite SNN scores in command order.")
    if not isfinite(minimum_margin) or minimum_margin < 0:
        raise ValueError("The minimum score margin must be finite and nonnegative.")
    ordered = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    winner, runner_up = ordered[:2]
    margin = scores[winner] - scores[runner_up]
    selected = winner if margin > minimum_margin else COMMAND_CLASSES.index("no_action")
    return {"command_id": selected, "command": COMMAND_CLASSES[selected], "score_margin": margin}


def replay(cache, weights_path, minimum_margin):
    import numpy as np
    import torch

    from training.export import load_export
    from training.train import center_event_counts

    model, weights, metadata = load_export(weights_path)
    with np.load(cache, allow_pickle=False) as data:
        if (tuple(metadata["classes"]) != COMMAND_CLASSES or tuple(data["classes"]) != COMMAND_CLASSES
                or metadata["ticks_per_step"] != int(data["ticks_per_step"])
                or metadata["archive_fingerprint"] != str(data["archive_fingerprint"])):
            raise ValueError("Expected matching movement command weights and event cache.")
        selected = data["splits"] == "test"
        inputs, labels = torch.from_numpy(data["frames"][selected]), data["labels"][selected]
    if metadata["center_events"]:
        inputs = center_event_counts(inputs)
    with torch.inference_mode():
        for i, (clip, expected) in enumerate(zip(inputs, labels, strict=True)):
            scores = model(clip.unsqueeze(0), weights=weights)[0].tolist()
            result = select_command(scores, minimum_margin)
            yield {"sample": i, "source": metadata["source"], "expected": COMMAND_CLASSES[expected],
                   "scores": scores, **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-margin", type=float, default=0.0)
    commands = parser.add_subparsers(dest="mode", required=True)
    decode = commands.add_parser("decode", help="Decode four scores in command order")
    decode.add_argument("scores", type=float, nargs=len(COMMAND_CLASSES))
    replay_parser = commands.add_parser("replay", help="Run the exported INT8 model on the held-out clips")
    replay_parser.add_argument("--cache", type=Path, required=True)
    replay_parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "decode":
        print(json.dumps(select_command(args.scores, args.minimum_margin)))
    else:
        import torch

        torch.set_num_threads(4)
        for result in replay(args.cache, args.weights, args.minimum_margin):
            print(json.dumps(result))
