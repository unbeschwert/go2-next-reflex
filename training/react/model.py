"""REACT-style C-SiLIF classifier with batched and streaming inference."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn


def affine_scan(decay: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """Evaluate u[t] = decay[t] * u[t-1] + drive[t] from a zero state.

    Compose adjacent transitions, scan the resulting pairs, then recover the
    states at the even indices. This takes linear work and logarithmic depth.
    """
    length = drive.shape[1]
    if length == 1:
        return drive

    odd_decay = decay[:, 1::2]
    paired = affine_scan(
        odd_decay * decay[:, :length - 1:2],
        odd_decay * drive[:, :length - 1:2] + drive[:, 1::2],
    )
    even = drive[:, 2::2] + decay[:, 2::2] * paired[:, :(length - 1) // 2]
    even = torch.cat((drive[:, :1], even), dim=1)
    interleaved = torch.stack((even[:, :length // 2], paired), dim=2).flatten(1, 2)
    return torch.cat((interleaved, even[:, length // 2:]), dim=1)


class CSiLIF(nn.Module):
    """Complex state with elapsed time in seconds and a binary readout."""

    def __init__(self, width: int):
        super().__init__()
        # The learned time scale is absorbed into the decay and frequency.
        self.log_decay = nn.Parameter(torch.linspace(math.log(3), math.log(30), width))
        self.angular_frequency = nn.Parameter(2 * math.pi * torch.linspace(1, 8, width))
        self.gain_real = nn.Parameter(torch.empty(width).uniform_(0.01, 0.05))
        self.gain_imag = nn.Parameter(torch.zeros(width))
        self.threshold = nn.Parameter(torch.ones(width))

    def transition(self, feature: torch.Tensor, elapsed_s: torch.Tensor):
        elapsed = elapsed_s.unsqueeze(-1)
        magnitude = torch.exp(-elapsed * self.log_decay.exp())
        phase = elapsed * self.angular_frequency
        decay = torch.polar(magnitude, phase)
        drive = feature * torch.complex(self.gain_real, self.gain_imag)
        return decay, drive

    def spike(self, state: torch.Tensor) -> torch.Tensor:
        potential = 2 * state.real - self.threshold
        hard = (potential >= 0).to(potential.dtype)
        if self.training:
            smooth = torch.atan(math.pi * potential) / math.pi
            return hard + (smooth - smooth.detach())
        return hard

    def forward(self, feature: torch.Tensor, elapsed_s: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        elapsed_s = elapsed_s.masked_fill(~valid, 0)
        feature = feature.masked_fill(~valid.unsqueeze(-1), 0)
        decay, drive = self.transition(feature, elapsed_s)
        return self.spike(affine_scan(decay, drive))

    def step(self, feature: torch.Tensor, elapsed_s: torch.Tensor,
             state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        decay, drive = self.transition(feature, elapsed_s)
        state = drive if state is None else decay * state + drive
        return self.spike(state), state


class SpikingBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.cell = CSiLIF(width)
        self.mix = nn.Linear(width, width, bias=False)

    def forward(self, feature: torch.Tensor, elapsed_s: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        return feature + self.mix(self.cell(feature, elapsed_s, valid))

    def step(self, feature: torch.Tensor, elapsed_s: torch.Tensor,
             state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        spikes, state = self.cell.step(feature, elapsed_s, state)
        return feature + self.mix(spikes), state


class StreamState(NamedTuple):
    layers: tuple[torch.Tensor, ...]
    feature_sum: torch.Tensor
    events: int


class ReactClassifier(nn.Module):
    """Clip classifier with a running-mean readout for event-by-event replay."""

    def __init__(self, size: int, classes: int, width: int = 32, blocks: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(2 * size * size, width)
        self.norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(SpikingBlock(width) for _ in range(blocks))
        self.readout = nn.Linear(width, classes)

    def forward(self, tokens: torch.Tensor, elapsed_s: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        feature = self.norm(self.embedding(tokens))
        for block in self.blocks:
            feature = block(feature, elapsed_s, valid)
        total = feature.masked_fill(~valid.unsqueeze(-1), 0).sum(dim=1)
        return self.readout(total / valid.sum(dim=1, keepdim=True).clamp_min(1))

    def step(self, token: torch.Tensor, elapsed_s: torch.Tensor,
             state: StreamState | None = None) -> tuple[torch.Tensor, StreamState]:
        """Consume one event per stream. Inputs have shape [batch]."""
        feature = self.norm(self.embedding(token))
        states = []
        for index, block in enumerate(self.blocks):
            previous = None if state is None else state.layers[index]
            feature, current = block.step(feature, elapsed_s, previous)
            states.append(current)
        total = feature if state is None else state.feature_sum + feature
        count = 1 if state is None else state.events + 1
        return self.readout(total / count), StreamState(tuple(states), total, count)
