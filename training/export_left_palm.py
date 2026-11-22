"""Pack the quantization-aware left-palm SNN for PyTorch replay and bridging."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from event_data import SIZE
from training.data import MAX_COUNT
from training.model import Q26_SCALE, STATE_SCALE, THRESHOLD, GestureSNN, quantized_q26, quantized_weight, weight_scales


FORMAT = "left-palm-snn-int8-v1"
Q26_FORMAT = "left-palm-snn-q2.6-v1"
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
    use_q26 = model.parameter_format == "q2.6"
    with torch.no_grad():
        layers = {name: pack_q26_layer(layer) if use_q26 else pack_layer(layer)
                  for name, layer in model_layers(model)}
    parameter_encoding = ({"weight_scale": Q26_SCALE, "bias_dtype": "int8", "bias_scale": Q26_SCALE}
                          if use_q26 else {"weight_scale_axis": 0})
    data = {
        "format": Q26_FORMAT if use_q26 else FORMAT,
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
        "quantization": {"weight_dtype": "int8", **parameter_encoding, "input_scale": 1 / 255,
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


def encode_q26(values):
    codes = (quantized_q26(values) / Q26_SCALE).to(torch.int8)
    return base64.b64encode(codes.numpy().tobytes()).decode("ascii")


def pack_q26_layer(layer):
    return {"shape": list(layer.weight.shape), "int8_base64": encode_q26(layer.weight),
            "bias_int8_base64": encode_q26(layer.bias)}


def decode_int8(encoded, shape):
    raw = base64.b64decode(encoded, validate=True)
    codes = np.frombuffer(raw, dtype=np.int8).reshape(shape).astype(np.float32)
    return torch.from_numpy(codes)


def load_model(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["format"] not in (FORMAT, Q26_FORMAT) or data["target"] != TARGET or data["model_args"]["precision"] != "qat":
        raise ValueError("Expected a portable quantized left-palm model.")
    model = GestureSNN(**data["model_args"]).eval()
    use_q26 = data["format"] == Q26_FORMAT
    if use_q26 != (model.parameter_format == "q2.6"):
        raise ValueError("The model and packed parameter formats disagree.")
    if use_q26 and any(data["quantization"].get(field) != Q26_SCALE for field in ("weight_scale", "bias_scale")):
        raise ValueError("Q2.6 weights and biases require a fixed scale of 1/64.")
    weights = []
    with torch.no_grad():
        for name, layer in model_layers(model):
            source = data["layers"][name]
            if tuple(source["shape"]) != layer.weight.shape:
                raise ValueError(f"Weight dimensions differ for {name}.")
            codes = decode_int8(source["int8_base64"], layer.weight.shape)
            if use_q26:
                bias = decode_int8(source["bias_int8_base64"], layer.bias.shape)
                weights.append(codes * Q26_SCALE)
                layer.bias.copy_(bias * Q26_SCALE)
                continue
            scales = torch.tensor(source["scales"], dtype=torch.float32)
            if scales.shape != layer.bias.shape:
                raise ValueError(f"Weight scale dimensions differ for {name}.")
            scale_shape = (-1,) + (1,) * (codes.ndim - 1)
            weights.append(codes * scales.view(scale_shape))
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
