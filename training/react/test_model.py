"""Regression checks for training and streaming state updates."""
from __future__ import annotations

import math
import unittest

import torch

from .model import CSiLIF, ReactClassifier, affine_scan


class ReactModelTests(unittest.TestCase):
    def test_scan_matches_sequential_recurrence_for_odd_and_even_lengths(self):
        generator = torch.Generator().manual_seed(11)
        for length in (1, 2, 3, 9, 16):
            with self.subTest(length=length):
                decay = torch.rand(2, length, 4, dtype=torch.complex64, generator=generator) * 0.5
                drive = torch.randn(2, length, 4, dtype=torch.complex64,
                                    generator=generator, requires_grad=True)
                expected = []
                prior = torch.zeros_like(drive[:, 0])
                for index in range(length):
                    prior = decay[:, index] * prior + drive[:, index]
                    expected.append(prior)
                reference = torch.stack(expected, dim=1)
                actual = affine_scan(decay, drive)
                torch.testing.assert_close(actual, reference)
                actual_grad, = torch.autograd.grad(actual.abs().sum(), drive)
                expected_grad, = torch.autograd.grad(reference.abs().sum(), drive)
                torch.testing.assert_close(actual_grad, expected_grad)

    def test_padded_batch_matches_streaming_replay(self):
        torch.manual_seed(42)
        model = ReactClassifier(size=4, classes=5, width=8).eval()
        tokens = torch.randint(32, (2, 9))
        elapsed = torch.rand(2, 9) * 0.01
        elapsed[:, 0] = 0
        valid = torch.arange(9) < torch.tensor([9, 4]).unsqueeze(1)
        with torch.no_grad():
            for block in model.blocks:
                block.cell.threshold.fill_(0.05)
            batched_scores = model(tokens, elapsed, valid)
            for index, length in enumerate((9, 4)):
                state = None
                for position in range(length):
                    scores, state = model.step(tokens[index:index + 1, position],
                                               elapsed[index:index + 1, position], state)
                self.assertEqual(state.events, length)
                torch.testing.assert_close(batched_scores[index], scores[0])

    def test_elapsed_time_changes_spikes_and_spike_preserves_state(self):
        cell = CSiLIF(width=1).eval()
        with torch.no_grad():
            cell.log_decay.fill_(math.log(2))
            cell.angular_frequency.zero_()
            cell.gain_real.fill_(0.5)
            cell.threshold.fill_(0.7)
            first_spike, state = cell.step(torch.ones(1, 1), torch.zeros(1), None)
            immediate, _ = cell.step(torch.zeros(1, 1), torch.zeros(1), state)
            delayed, _ = cell.step(torch.zeros(1, 1), torch.ones(1), state)
        self.assertEqual(first_spike.item(), 1)
        self.assertEqual(immediate.item(), 1)
        self.assertEqual(delayed.item(), 0)


if __name__ == "__main__":
    unittest.main()
