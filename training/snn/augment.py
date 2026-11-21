"""Data augmentation on batches of cached event counts, on the GPU (training batches only).

Input: counts [B, T, 2, H, W] (float, events per downsampled pixel per frame, before encoding).
Output: augmented counts, possibly with a warm-start prefix of extra frames in front, and the length
of that prefix (the loss only counts output spikes after it). See doc/data-augmentation.md.

Order: what changes the scene first, what the sensor adds last:
  dropout → speed / time shift → affine (zoom, rotation) / flip → integer shift → cutout → rate
  → polarity → noise / hot pixels → warm-start prefix
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .config import AugmentConfig, Config


def _pick(p: float, b: int, device) -> torch.Tensor:
    """Which clips of the batch get an augmentation: bool [B]."""
    return torch.rand(b, device=device) < p


def _uniform(low: float, high: float, b: int, device) -> torch.Tensor:
    return low + (high - low) * torch.rand(b, device=device)


def _per_clip(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """[B] → [B, 1, 1, ...] to broadcast against x."""
    return v.view(-1, *([1] * (x.dim() - 1)))


def dropout(x: torch.Tensor, a: AugmentConfig) -> torch.Tensor:
    """Remove a random fraction (0..max) of the events of each clip: binomial thinning of the counts."""
    chosen = _pick(a.dropout.p, len(x), x.device)
    if not chosen.any():
        return x
    keep = torch.where(chosen, 1 - a.dropout.max * torch.rand(len(x), device=x.device), torch.ones_like(chosen, dtype=x.dtype))
    sub = x[chosen]
    x = x.clone()
    x[chosen] = torch.binomial(sub.round(), _per_clip(keep[chosen], sub).expand_as(sub))
    return x


def resample_time(x: torch.Tensor, speed: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Play each clip `speed` times faster, delayed by `shift` frames (both [B], may be fractional).

    Output frame t holds the events the clip produced in source time [tau(t), tau(t+1)), with
    tau(t) = c + (t - c) * speed - shift and c the clip centre. This integrates the cumulative event count,
    so no event is invented or lost except those moved outside the clip, and the events per frame
    scale with the speed, as they would for a faster movement. Gaps are empty frames.
    """
    b, t = x.shape[:2]
    cumulative = torch.cat([torch.zeros_like(x[:, :1]), x.cumsum(dim=1)], dim=1)          # [B, T+1, ...]
    steps = torch.arange(t + 1, device=x.device, dtype=x.dtype)
    centre = t / 2
    tau = (centre + (steps[None] - centre) * speed[:, None] - shift[:, None]).clamp(0, t)   # [B, T+1]
    lo = tau.floor().long().clamp(max=t - 1)
    frac = (tau - lo).view(b, t + 1, *([1] * (x.dim() - 2)))
    index = lo.view(b, t + 1, *([1] * (x.dim() - 2))).expand_as(cumulative)
    low = cumulative.gather(1, index)
    high = cumulative.gather(1, index + 1)
    return (low + frac * (high - low)).diff(dim=1).clamp_min(0)


def temporal(x: torch.Tensor, a: AugmentConfig) -> torch.Tensor:
    b, dev = len(x), x.device
    speed_on, shift_on = _pick(a.speed.p, b, dev), _pick(a.time_shift.p, b, dev)
    if not (speed_on.any() or shift_on.any()):
        return x
    speed = torch.where(speed_on, _uniform(*a.speed.range, b, dev), torch.ones(b, device=dev))
    m = a.time_shift.max_frames
    shift = torch.where(shift_on, torch.randint(-m, m + 1, (b,), device=dev).float(), torch.zeros(b, device=dev))
    chosen = speed_on | shift_on
    x = x.clone()
    x[chosen] = resample_time(x[chosen], speed[chosen], shift[chosen])
    return x


def spatial(x: torch.Tensor, a: AugmentConfig) -> torch.Tensor:
    """Zoom + rotation (one bilinear affine transform per clip, same for all frames), then horizontal flip."""
    b, dev = len(x), x.device
    chosen = _pick(a.affine.p, b, dev)
    if chosen.any():
        n = int(chosen.sum())
        zoom = _uniform(*a.affine.scale, n, dev)
        angle = _uniform(-a.affine.rotate_deg, a.affine.rotate_deg, n, dev) * math.pi / 180
        cos, sin, zero = torch.cos(angle) / zoom, torch.sin(angle) / zoom, torch.zeros(n, device=dev)
        # affine_grid maps output → input coordinates: zoom > 1 samples a smaller input area (person bigger)
        theta = torch.stack([torch.stack([cos, -sin, zero], -1), torch.stack([sin, cos, zero], -1)], 1)
        sub = x[chosen]
        _, t, c, h, w = sub.shape
        grid = nn.functional.affine_grid(theta, [n, t * c, h, w], align_corners=False)
        out = nn.functional.grid_sample(sub.reshape(n, t * c, h, w), grid, mode="bilinear",
                                        padding_mode="zeros", align_corners=False)
        x = x.clone()
        x[chosen] = out.view(n, t, c, h, w)
    flip = _pick(a.flip.p, b, dev)
    if flip.any():
        x = torch.where(_per_clip(flip, x), x.flip(dims=[-1]), x)
    return x


def random_shift(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Shift each clip by up to max_shift whole pixels, filling with zeros (as the notebook does)."""
    if not max_shift:
        return x
    b, _, _, h, w = x.shape
    padded = nn.functional.pad(x, (max_shift,) * 4)
    offsets = torch.randint(0, 2 * max_shift + 1, (b, 2)).tolist()
    return torch.stack([padded[i, ..., dy:dy + h, dx:dx + w] for i, (dy, dx) in enumerate(offsets)])


def cutout(x: torch.Tensor, a: AugmentConfig) -> torch.Tensor:
    """Blank one random rectangle per chosen clip, in every frame (something in front of the person)."""
    b, _, _, h, w = x.shape
    chosen = _pick(a.cutout.p, b, x.device)
    if not chosen.any():
        return x
    side = _uniform(*a.cutout.size, b, x.device)
    rh, rw = (side * h).round().long().clamp(min=1), (side * w).round().long().clamp(min=1)
    top = (torch.rand(b, device=x.device) * (h - rh + 1)).floor().long()
    left = (torch.rand(b, device=x.device) * (w - rw + 1)).floor().long()
    rows, cols = torch.arange(h, device=x.device), torch.arange(w, device=x.device)
    inside = (((rows[None] >= top[:, None]) & (rows[None] < (top + rh)[:, None]))[:, :, None]
              & ((cols[None] >= left[:, None]) & (cols[None] < (left + rw)[:, None]))[:, None, :])   # [B, H, W]
    inside &= chosen[:, None, None]
    return x * (~inside)[:, None, None].to(x.dtype)


def rate(x: torch.Tensor, a: AugmentConfig) -> torch.Tensor:
    """Scale the event counts of each clip (lighting, contrast, camera bias settings)."""
    chosen = _pick(a.rate.p, len(x), x.device)
    if not chosen.any():
        return x
    factor = torch.where(chosen, _uniform(*a.rate.range, len(x), x.device), torch.ones(len(x), device=x.device))
    return x * _per_clip(factor, x)


def polarity(x: torch.Tensor, a: AugmentConfig) -> torch.Tensor:
    chosen = _pick(a.polarity.p, len(x), x.device)
    if not chosen.any():
        return x
    return torch.where(_per_clip(chosen, x), x.flip(dims=[2]), x)


def sensor_noise(x: torch.Tensor, a: AugmentConfig) -> torch.Tensor:
    """Poisson background events everywhere, plus a few always-firing (hot) pixels."""
    b, _, c, h, w = x.shape
    dev = x.device
    noise_on, hot_on = _pick(a.noise.p, b, dev), _pick(a.hot_pixels.p, b, dev)
    if not (noise_on.any() or hot_on.any()):
        return x
    rate_map = torch.zeros(b, c, h, w, device=dev)
    background = torch.where(noise_on, a.noise.max_rate * torch.rand(b, device=dev), torch.zeros(b, device=dev))
    rate_map += background.view(b, 1, 1, 1) / c                       # split between OFF and ON
    for i in hot_on.nonzero().flatten().tolist():
        k = int(torch.randint(1, a.hot_pixels.max_count + 1, (1,)))
        rate_map[i].view(-1)[torch.randint(0, c * h * w, (k,), device=dev)] += _uniform(*a.hot_pixels.rate, k, dev)
    return x + torch.poisson(rate_map[:, None].expand_as(x).contiguous())


def warm_start(x: torch.Tensor, a: AugmentConfig) -> tuple[torch.Tensor, int]:
    """Put a prefix in front of the batch, like the FPGA, whose potentials never reset between clips.

    The prefix is the end of another clip of the batch (a previous gesture) or empty frames (only the
    biases drive the neurons). Returns (frames, prefix length); the loss ignores spikes in the prefix.
    """
    if torch.rand(()) >= a.warm_start.p:
        return x, 0
    low, high = a.warm_start.frames
    n = int(torch.randint(low, high + 1, (1,)))
    if n == 0:
        return x, 0
    other = x[torch.randperm(len(x), device=x.device), -n:]
    from_clip = _per_clip(_pick(a.warm_start.from_clip, len(x), x.device), other)
    prefix = torch.where(from_clip, other, torch.zeros_like(other))
    return torch.cat([prefix, x], dim=1), n


def augment(counts: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, int]:
    """All configured augmentations, in order. Returns (counts, warm-start prefix length)."""
    a = cfg.augment
    x = dropout(counts, a)
    x = temporal(x, a)
    x = spatial(x, a)
    x = random_shift(x, cfg.data.max_shift)
    x = cutout(x, a)
    x = rate(x, a)
    x = polarity(x, a)
    x = sensor_noise(x, a)
    return warm_start(x, a)
