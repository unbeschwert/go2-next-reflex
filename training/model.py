"""Portable Conv-LIF model with fake quantization in every forward pass."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F

from training.data import CLASSES, GRID, MAX_COUNT


LEAK = 0.875
THRESHOLD = 0.5
STATE_SCALE = 1.0 / 128


def weight_scales(weight):
    return weight.detach().flatten(1).abs().amax(dim=1).clamp_min(1e-6) / 127


def quantized_weight(weight):
    scales = weight_scales(weight)
    zero_points = torch.zeros_like(scales, dtype=torch.int32)
    return torch.fake_quantize_per_channel_affine(weight, scales, zero_points, 0, -127, 127)


class GestureSNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        channels = (2, 8, 16, 24)
        self.convolutions = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        )
        self.classifier = nn.Linear(2 * channels[-1] * (GRID // 8) ** 2, num_classes)
        for layer in self.convolutions:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, counts, *, weights=None):
        if weights is None:
            weights = [quantized_weight(layer.weight) for layer in (*self.convolutions, self.classifier)]
        events = torch.log1p(counts.float()) / log(1 + MAX_COUNT)
        events = torch.fake_quantize_per_tensor_affine(events, 1.0 / 255, 0, 0, 255)
        batch, steps = events.shape[:2]
        for convolution, weight in zip(self.convolutions, weights[:-1], strict=True):
            current = F.conv2d(events.flatten(0, 1), weight, convolution.bias, stride=2, padding=1)
            current = current.view(batch, steps, *current.shape[1:])
            membrane = torch.zeros_like(current[:, 0])
            activity = []
            for drive in current.unbind(dim=1):
                voltage = torch.fake_quantize_per_tensor_affine(
                    LEAK * membrane + drive, STATE_SCALE, 0, -32768, 32767,
                )
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
