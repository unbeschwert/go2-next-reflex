"""Read raw Motion Buddy events and batch variable-length event sequences."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from training.snn.config import SENSOR_SIZE, DataConfig
from training.snn.data import Clip
from training.snn.events import read_events


def transform_positions(events: np.ndarray, dx: int, dy: int, zoom: float) -> np.ndarray:
    """Move whole clips in sensor coordinates, preserving retained event timestamps."""
    center = (SENSOR_SIZE - 1) / 2
    x = np.rint((events["x"] - center) * zoom + center + dx)
    y = np.rint((events["y"] - center) * zoom + center + dy)
    inside = (x >= 0) & (x < SENSOR_SIZE) & (y >= 0) & (y < SENSOR_SIZE)
    if not inside.any():
        return events
    transformed = events[inside]
    transformed["x"] = x[inside]
    transformed["y"] = y[inside]
    return transformed


class EventDataset(Dataset):
    def __init__(self, clips: list[Clip], config: DataConfig, max_events: int,
                 spatial_augment: bool = False, shift_x: int = 0):
        self.clips = clips
        self.config = config
        self.max_events = max_events
        self.spatial_augment = spatial_augment
        self.shift_x = shift_x

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        clip = self.clips[index]
        events = read_events(clip.path)
        if not len(events):
            raise ValueError(f"{clip.path}: recording contains no events")

        timestamps = events["timestamp"]
        end = np.searchsorted(timestamps, int(timestamps[0]) + self.config.clip_us)
        events = events[:end]
        if len(events) > self.max_events:
            # Sample across the whole clip and preserve the first and last events.
            indices = np.arange(self.max_events, dtype=np.int64)
            indices = indices * (len(events) - 1) // (self.max_events - 1)
            events = events[indices]

        if np.any((events["x"] >= SENSOR_SIZE) | (events["y"] >= SENSOR_SIZE) | (events["polarity"] > 1)):
            raise ValueError(f"{clip.path}: event outside sensor bounds")
        downsampling = self.config.downsampling
        if self.spatial_augment:
            dx, dy = (torch.randint(-4, 5, (2,)) * downsampling).tolist()
            zoom = torch.empty(()).uniform_(0.8, 1.2).item()
            events = transform_positions(events, dx, dy, zoom)
        elif self.shift_x:
            events = transform_positions(events, self.shift_x * downsampling, 0, 1.0)

        size = self.config.size
        x = events["x"].astype(np.int64)
        y = events["y"].astype(np.int64)
        p = events["polarity"].astype(np.int64)
        x //= downsampling
        y //= downsampling
        tokens = (p * size + y) * size + x
        timestamps = events["timestamp"].astype(np.int64)
        intervals = np.diff(timestamps, prepend=timestamps[0])
        if np.any(intervals < 0):
            raise ValueError(f"{clip.path}: event timestamps are out of order")
        return torch.from_numpy(tokens), torch.from_numpy(intervals.astype(np.float32) * 1e-6), clip.label


def collate_events(batch):
    tokens, elapsed, labels = zip(*batch)
    lengths = torch.tensor([sequence.numel() for sequence in tokens])
    tokens = pad_sequence(tokens, batch_first=True)
    valid = torch.arange(tokens.size(1)) < lengths.unsqueeze(1)
    return tokens, pad_sequence(elapsed, batch_first=True), valid, torch.tensor(labels)
