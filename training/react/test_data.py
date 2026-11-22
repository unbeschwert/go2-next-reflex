"""Check raw-event transforms and subject isolation."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from training.snn.config import DataConfig
from training.snn.data import Clip
from training.snn.events import EVENT_DTYPE

from .__main__ import validation_fold
from .data import EventDataset


class ReactDataTests(unittest.TestCase):
    def test_position_shift_crops_and_preserves_elapsed_time(self):
        events = np.array([(50, 80, 0, 1000), (310, 80, 1, 1100), (60, 80, 1, 1400)],
                          dtype=EVENT_DTYPE)
        config = DataConfig()
        dataset = EventDataset([Clip("clip.bin", 3, "11")], config, 16, shift_x=4)
        with patch("training.react.data.read_events", return_value=events):
            tokens, intervals, label = dataset[0]
        self.assertEqual(tokens.tolist(), [410, 2011])
        self.assertEqual(label, 3)
        torch.testing.assert_close(intervals, torch.tensor([0.0, 0.0004]))

    def test_validation_fold_leaves_configured_test_subjects_out(self):
        config = DataConfig()
        fold = validation_fold(config, ["09", "10"])
        self.assertEqual(fold.val, ["09", "10"])
        self.assertEqual(fold.train[-2:], ["11", "12"])
        self.assertEqual(fold.test, ["13", "14"])
        with self.assertRaises(ValueError):
            validation_fold(config, ["13"])
