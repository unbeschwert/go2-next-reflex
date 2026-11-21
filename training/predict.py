"""Classify one recording from the local gesture ZIP."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from event_data import decode_recording
from training.data import MODEL_CLASS_ORDERS, bin_events, input_shape
from training.export import load_export
from training.model import GestureSNN
from training.train import center_event_counts


@torch.inference_mode()
def predict(archive, recording, checkpoint_path, *, use_export=False):
    if use_export:
        model, weights, checkpoint = load_export(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("backend", "pytorch") == "riverops":
            from training.riverops_commands import build_model

            model = build_model().eval()
        else:
            model = GestureSNN(len(checkpoint["classes"]), width=checkpoint.get("model_width", "compact"),
                               precision=checkpoint.get("precision", "qat"),
                               event_encoding=checkpoint.get("event_encoding", "polarity-counts"),
                               ticks_per_step=checkpoint["ticks_per_step"]).eval()
        model.load_state_dict(checkpoint["state_dict"])
        weights = None
    classes = tuple(checkpoint["classes"])
    ticks_per_step = checkpoint["ticks_per_step"]
    if classes not in MODEL_CLASS_ORDERS or tuple(checkpoint["input_shape"]) != input_shape(ticks_per_step):
        raise ValueError("The checkpoint uses a different class order or event encoding.")
    with ZipFile(archive) as source:
        events, timestamps = decode_recording(source.read(recording), recording)
    counts = torch.from_numpy(bin_events(events, timestamps, ticks_per_step)).to(torch.uint8).unsqueeze(0)
    if checkpoint.get("center_events", False):
        counts = center_event_counts(counts)
    scores = model(counts, weights=weights).softmax(dim=1).squeeze(0)
    return {"recording": recording, "prediction": classes[scores.argmax().item()],
            "scores": dict(zip(classes, scores.tolist()))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Path of the recording inside the ZIP")
    parser.add_argument("--archive", type=Path, default=Path("durance-ai-handgesture-recognition.zip"))
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--checkpoint", type=Path, default=Path(".local/training/run/gesture-snn.pt"))
    sources.add_argument("--weights", type=Path, help="Portable INT8 NPZ produced by training.export")
    args = parser.parse_args()
    result = predict(args.archive, args.recording, args.weights or args.checkpoint, use_export=args.weights is not None)
    print(json.dumps(result, indent=2))
