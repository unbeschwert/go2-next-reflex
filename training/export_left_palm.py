"""Pack the quantization-aware left-palm SNN for PyTorch replay and bridging."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from event_data import SIZE
from training.data import MAX_COUNT
from training.model import GestureSNN, STATE_SCALE, THRESHOLD, quantized_weight, weight_scales


FORMAT = "left-palm-snn-int8-v1"
TARGET = ["left_palm_x", "left_palm_y"]


def model_layers(model):
    yield from ((f"convolutions.{index}", conv) for index, conv in enumerate(model.convolutions))
    yield "classifier", model.classifier


def export(checkpoint, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_args = payload["model_args"]
    if payload["target"] != TARGET or model_args["precision"] != "qat":
        raise ValueError("Expected a left-palm quantization-aware checkpoint.")
    model = GestureSNN(**model_args).eval()
    model.load_state_dict(payload["state_dict"])
    with torch.no_grad():
        layers = {name: pack_layer(layer) for name, layer in model_layers(model)}
    data = {
        "format": FORMAT,
        "runtime": "pytorch",
        "target": TARGET,
        "model_args": model_args,
        "input_shape": [payload["context_steps"], 2, model_args["grid"], model_args["grid"]],
        "polarity_order": ["off", "on"],
        "sensor_size": [SIZE, SIZE],
        "max_event_count": MAX_COUNT,
        "horizon_us": payload["horizon_us"],
        "coordinate_offset": payload["coordinate_offset"],
        "coordinate_scale": payload["coordinate_scale"],
        "quantization": {"weight_dtype": "int8", "weight_scale_axis": 0, "input_scale": 1 / 255,
                         "state_scale": STATE_SCALE, "state_range": [-32768, 32767],
                         "threshold": THRESHOLD, "leak": model.leak},
        "layers": layers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pack_layer(layer):
    scales = weight_scales(layer.weight)
    shape = (-1,) + (1,) * (layer.weight.ndim - 1)
    effective = quantized_weight(layer.weight)
    codes = (effective / scales.view(shape)).round().clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(codes.float() * scales.view(shape), effective, atol=0, rtol=0)
    return {"shape": list(codes.shape), "int8_base64": base64.b64encode(codes.numpy().tobytes()).decode("ascii"),
            "scales": scales.tolist(), "bias": layer.bias.tolist()}


def load_model(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["format"] != FORMAT or data["target"] != TARGET or data["model_args"]["precision"] != "qat":
        raise ValueError("Expected a portable INT8 left-palm model.")
    model = GestureSNN(**data["model_args"]).eval()
    weights = []
    with torch.no_grad():
        for name, layer in model_layers(model):
            source = data["layers"][name]
            if tuple(source["shape"]) != layer.weight.shape:
                raise ValueError(f"Weight dimensions differ for {name}.")
            raw = base64.b64decode(source["int8_base64"], validate=True)
            codes = np.frombuffer(raw, dtype=np.int8).reshape(source["shape"]).astype(np.float32)
            scales = torch.tensor(source["scales"], dtype=torch.float32)
            if scales.shape != layer.bias.shape:
                raise ValueError(f"Weight scale dimensions differ for {name}.")
            scale_shape = (-1,) + (1,) * (codes.ndim - 1)
            weights.append(torch.from_numpy(codes) * scales.view(scale_shape))
            bias = torch.tensor(source["bias"], dtype=torch.float32)
            if bias.shape != layer.bias.shape or not torch.isfinite(bias).all() or not torch.isfinite(scales).all():
                raise ValueError(f"Invalid bias or scale for {name}.")
            layer.bias.copy_(bias)
    return model, weights, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
