"""Export and load the trained SNN's signed INT8 weights as a portable NPZ."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.data import MAX_COUNT, MODEL_CLASS_ORDERS, input_shape
from training.model import GestureSNN, STATE_SCALE, THRESHOLD, quantized_weight, weight_scales


def export_weights(checkpoint_path, output):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("backend", "pytorch") != "pytorch":
        raise ValueError("Use training.riverops_commands to export a RiverOps checkpoint.")
    if checkpoint.get("precision", "qat") != "qat":
        raise ValueError("The portable INT8 export requires a QAT checkpoint for replay parity.")
    classes = tuple(checkpoint["classes"])
    ticks_per_step = checkpoint["ticks_per_step"]
    if classes not in MODEL_CLASS_ORDERS or tuple(checkpoint["input_shape"]) != input_shape(ticks_per_step):
        raise ValueError("The checkpoint uses a different class order or event encoding.")
    model = GestureSNN(len(classes), width=checkpoint.get("model_width", "compact"),
                       event_encoding=checkpoint.get("event_encoding", "polarity-counts"),
                       ticks_per_step=ticks_per_step).eval()
    model.load_state_dict(checkpoint["state_dict"])
    layers = [f"conv{i}" for i in range(1, len(model.convolutions) + 1)] + ["classifier"]
    metadata = {
        "format": "gesture-snn-int8-v1",
        "classes": classes,
        "model_width": checkpoint.get("model_width", "compact"),
        "event_encoding": model.event_encoding,
        "source": checkpoint.get("source", "recorded"),
        "dance_policy": checkpoint.get("dance_policy", "keep"),
        "input_shape": input_shape(ticks_per_step),
        "ticks_per_step": ticks_per_step,
        "max_event_count": MAX_COUNT,
        "center_events": checkpoint.get("center_events", False),
        "layer_order": layers,
        "weight_zero_point": 0,
        "weight_scale_axis": 0,
        "input_scale": 1.0 if model.event_encoding == "ternary" else 1.0 / 255,
        "state_scale": STATE_SCALE,
        "state_range": [-32768, 32767],
        "leak": model.leak,
        "threshold": THRESHOLD,
        "checkpoint_epoch": checkpoint["epoch"],
        "archive_fingerprint": checkpoint["archive_fingerprint"],
    }
    arrays = {"metadata": np.array(json.dumps(metadata))}
    with torch.no_grad():
        for name, layer in zip(layers, (*model.convolutions, model.classifier), strict=True):
            scales = weight_scales(layer.weight)
            scale_shape = (-1,) + (1,) * (layer.weight.ndim - 1)
            effective_weight = quantized_weight(layer.weight)
            codes = (effective_weight / scales.view(scale_shape)).round().clamp(-127, 127).to(torch.int8)
            torch.testing.assert_close(codes.float() * scales.view(scale_shape), effective_weight, rtol=0, atol=0)
            arrays[f"{name}_weight"] = codes.numpy()
            arrays[f"{name}_scale"] = scales.numpy()
            arrays[f"{name}_bias"] = layer.bias.numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output)
    print(f"Exported {sum(arrays[f'{name}_weight'].size for name in layers):,} INT8 weights to {output}")


def load_export(path):
    weights = []
    with np.load(path, allow_pickle=False) as data, torch.no_grad():
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("format") != "gesture-snn-int8-v1":
            raise ValueError("Unrecognized gesture SNN export format.")
        if tuple(metadata["classes"]) not in MODEL_CLASS_ORDERS:
            raise ValueError("Unrecognized gesture SNN class order.")
        model = GestureSNN(len(metadata["classes"]), width=metadata.get("model_width", "compact"),
                           event_encoding=metadata.get("event_encoding", "polarity-counts"),
                           ticks_per_step=metadata["ticks_per_step"]).eval()
        layers = (*model.convolutions, model.classifier)
        for name, layer in zip(metadata["layer_order"], layers, strict=True):
            codes = torch.from_numpy(data[f"{name}_weight"])
            scales = torch.from_numpy(data[f"{name}_scale"])
            layer.bias.copy_(torch.from_numpy(data[f"{name}_bias"]))
            weights.append(codes.float() * scales.view((-1,) + (1,) * (codes.ndim - 1)))
    return model, weights, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(".local/training/run/gesture-snn.pt"))
    parser.add_argument("--output", type=Path, default=Path(".local/training/run/gesture-snn-int8.npz"))
    args = parser.parse_args()
    export_weights(args.checkpoint, args.output)
