"""RiverOps QAT model and native RiverSDK architecture export for movement commands."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from event_data import SIZE
from training.data import COMMAND_CLASSES, GRID, INPUT_SHAPE, TICKS_PER_STEP


def river_input(counts):
    return counts.float() / (SIZE // GRID) ** 2


class RiverCommandSNN(nn.Module):
    def __init__(self, conv):
        super().__init__()
        lif = dict(initial_tau=1.0 / (1.0 - 0.93), learnable_leak=False, v_threshold=1.0, reset_mode="HARD")
        self.net = nn.Sequential(
            conv(2, 16, kernel_size=5, stride=2, padding=1, bias=False, first_layer=True, **lif),
            conv(16, 16, kernel_size=3, padding=1, bias=False, **lif),
            conv(16, 8, kernel_size=6, stride=2, padding=2, bias=False, **lif),
            conv(8, 256, kernel_size=6, stride=2, bias=False, **lif),
            conv(256, len(COMMAND_CLASSES), kernel_size=1, bias=False, **lif),
        )

    def forward(self, counts):
        return self.net(river_input(counts)).sum(dim=1).flatten(start_dim=1)


def build_model():
    try:
        from riverops.nn import FixedPointFormat, QuantConfig, SpikingConvBN2d, prepare_qat
    except ModuleNotFoundError as error:
        if error.name == "riverops":
            raise RuntimeError("Install the Durance RiverOps SDK in this Python environment to train or export the FPGA model.") from error
        raise
    config = QuantConfig(
        weight_format=FixedPointFormat(total_bits=8, frac_bits=7),
        potential_format=FixedPointFormat(total_bits=16, frac_bits=12),
        enable_weight_quant=True,
        enable_potential_quant=True,
    )
    return prepare_qat(RiverCommandSNN(SpikingConvBN2d), config)


def export(checkpoint_path, cache, output):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("backend") != "riverops" or tuple(checkpoint["classes"]) != COMMAND_CLASSES
            or tuple(checkpoint["input_shape"]) != INPUT_SHAPE or checkpoint["ticks_per_step"] != TICKS_PER_STEP):
        raise ValueError("Expected a RiverOps checkpoint with the movement command output order.")
    model = build_model().eval()
    model.load_state_dict(checkpoint["state_dict"])
    from riverops.export import export_for_river

    with np.load(cache, allow_pickle=False) as data:
        if str(data["archive_fingerprint"]) != checkpoint["archive_fingerprint"]:
            raise ValueError("The example cache must match the training checkpoint.")
        sample = torch.from_numpy(data["frames"][:1])
    if checkpoint.get("center_events"):
        from training.train import center_event_counts

        sample = center_event_counts(sample)
    with torch.no_grad():
        if model(sample).shape != (1, len(COMMAND_CLASSES)):
            raise ValueError("Expected one spike count per command.")
        architecture = export_for_river(model.net, river_input(sample))
        parameters = [layer.export_params() for layer in model.net]
    json.loads(architecture)

    values = []
    for name in ("weights", "bias"):
        for layer in parameters:
            values.extend(np.asarray(layer[name]).reshape(-1).tolist())
    for layer in parameters:
        for name in ("threshold", "leak"):
            values.extend(np.asarray(layer[name]).reshape(-1).tolist())
    if any(not isinstance(value, int) for value in values):
        raise ValueError("RiverOps returned a non-integer parameter for the board weights file.")

    output.mkdir(parents=True, exist_ok=True)
    (output / "movement_model.json").write_text(architecture, encoding="utf-8")
    (output / "weights.txt").write_text("\n".join(map(str, values)), encoding="utf-8")
    print(f"Exported movement_model.json and weights.txt to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.cache, args.output)
