"""Portable Conv-LIF model with FP32 and fake-quantized execution modes."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F

from training.data import CLASSES, GRID, MAX_COUNT, TICKS_PER_STEP


LEAK = 0.875
THRESHOLD = 0.5
STATE_SCALE = 1.0 / 128
MODEL_WIDTHS = {"compact": (8, 16, 24), "wide": (16, 32, 48)}
EVENT_ENCODINGS = ("polarity-counts", "ternary")


def weight_scales(weight):
    return weight.detach().flatten(1).abs().amax(dim=1).clamp_min(1e-6) / 127


def quantized_weight(weight):
    scales = weight_scales(weight)
    zero_points = torch.zeros_like(scales, dtype=torch.int32)
    return torch.fake_quantize_per_channel_affine(weight, scales, zero_points, 0, -127, 127)


class GestureSNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES), *, width="compact", precision="qat",
                 event_encoding="polarity-counts", ticks_per_step=TICKS_PER_STEP, grid=GRID):
        super().__init__()
        if event_encoding not in EVENT_ENCODINGS:
            raise ValueError(f"Unknown event encoding: {event_encoding}")
        self.use_qat = precision == "qat"
        self.event_encoding = event_encoding
        self.leak = LEAK ** (ticks_per_step / TICKS_PER_STEP)
        channels = (1 if event_encoding == "ternary" else 2, *MODEL_WIDTHS[width])
        self.convolutions = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        )
        self.classifier = nn.Linear(2 * channels[-1] * ((grid + 7) // 8) ** 2, num_classes)
        for layer in self.convolutions:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode_events(self, counts):
        if self.event_encoding == "ternary":
            off, on = counts.float().unbind(dim=2)
            return (on - off).sign().unsqueeze(2)
        events = torch.log1p(counts.float()) / log(1 + MAX_COUNT)
        if self.use_qat:
            events = torch.fake_quantize_per_tensor_affine(events, 1.0 / 255, 0, 0, 255)
        return events

    def forward(self, counts, *, weights=None):
        if weights is None:
            layers = (*self.convolutions, self.classifier)
            weights = [quantized_weight(layer.weight) if self.use_qat else layer.weight for layer in layers]
        events = self.encode_events(counts)
        batch, steps = events.shape[:2]
        for convolution, weight in zip(self.convolutions, weights[:-1], strict=True):
            current = F.conv2d(events.flatten(0, 1), weight, convolution.bias, stride=2, padding=1)
            current = current.view(batch, steps, *current.shape[1:])
            membrane = torch.zeros_like(current[:, 0])
            activity = []
            for drive in current.unbind(dim=1):
                voltage = self.leak * membrane + drive
                if self.use_qat:
                    voltage = torch.fake_quantize_per_tensor_affine(voltage, STATE_SCALE, 0, -32768, 32767)
                relative = voltage - THRESHOLD
                spike = (relative >= 0).to(voltage.dtype)
                membrane = voltage - THRESHOLD * spike
                if self.training:
                    surrogate = (relative + 0.5).clamp(0, 1)
                    spike = spike + (surrogate - surrogate.detach())
                activity.append(spike)
            events = torch.stack(activity, dim=1)

        early, late = events.tensor_split(2, dim=1)
        features = torch.cat((early.mean(dim=1), late.mean(dim=1)), dim=1).flatten(1)
        return F.linear(features, weights[-1], self.classifier.bias)
