#!/usr/bin/env python3
"""Integer model of the SPLEAT accelerator, for fitting the hardware's arithmetic (numpy only).

Runs an exported weights file frame by frame with integer potentials, with the number handling
exposed as options, so hypotheses can be compared against outputs recorded from the real FPGA
(go2_viz/fpga_probe.py, fpga_eval.py --save).

Arithmetic (per layer and 10 ms step), all integers, potentials in units of 1/4096:
    current = (sum of weights of active inputs) << w_shift  +  bias << b_shift
    layer 0:  current = (sum of round(x * 4096) * weight) >> in_shift  +  bias << b_shift
    P = P - leak(P) + current ;  spike where P >= threshold ;  P = 0 where it spiked
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ARCHS = {
    "nb40": (40, [(2, 16, 5, 2, 1), (16, 16, 3, 1, 1), (16, 8, 6, 2, 2), (8, 256, 9, 1, 0), (256, 5, 1, 1, 0)]),
    "best80": (80, [(2, 16, 5, 2, 1), (16, 32, 3, 1, 1), (32, 32, 6, 2, 2),
                    (32, 256, 6, 2, 0), (256, 128, 3, 2, 0), (128, 5, 3, 1, 0)]),
}


def load_params(path, arch):
    size, layers = ARCHS[arch]
    vals = np.array([int(v) for v in open(path).read().split()], dtype=np.int64)
    n_w = [ci * co * k * k for (ci, co, k, _, _) in layers]
    n_b = [co for (_, co, _, _, _) in layers]
    assert len(vals) == sum(n_w) + sum(n_b) + 2 * len(layers), "file does not fit the architecture"
    W, B, o = [], [], 0
    for (ci, co, k, _, _), n in zip(layers, n_w):
        W.append(vals[o:o + n].reshape(co, ci, k, k)); o += n
    for n in n_b:
        B.append(vals[o:o + n]); o += n
    neuron = vals[o:].reshape(len(layers), 2)
    return size, layers, W, B, neuron[:, 0], neuron[:, 1]


def conv(x, w, stride, pad):
    """x [C, H, W] int64, w [Co, Ci, k, k] int64 -> [Co, Ho, Wo] int64"""
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    k = w.shape[-1]
    win = sliding_window_view(x, (k, k), axis=(1, 2))[:, ::stride, ::stride]     # [Ci, Ho, Wo, k, k]
    return np.tensordot(w, win, axes=([1, 2, 3], [0, 3, 4]))


class Spleat:
    def __init__(self, path, arch="nb40", w_shift=5, b_shift=5, in_shift=7, leak_mode="floor",
                 saturate=True, delay=False, w0_shift=None, sat_current=False, pot_bits=16,
                 reset="hard", cmp="ge", order="leak_add"):
        self.size, self.layers, self.W, self.B, self.thr, self.leak = load_params(path, arch)
        self.w_shift, self.b_shift, self.in_shift = w_shift, b_shift, in_shift
        self.leak_mode, self.saturate, self.delay, self.sat_current = leak_mode, saturate, delay, sat_current
        self.lim = 1 << (pot_bits - 1)
        self.reset_mode, self.cmp, self.order = reset, cmp, order
        self.reset()

    def reset(self):
        self.P = None
        self.prev = None

    def _leak(self, p, leak):
        if self.leak_mode == "floor":
            return (p * leak) >> 12
        if self.leak_mode == "round":
            return (p * leak + 2048) >> 12
        if self.leak_mode == "trunc":                       # toward zero
            return np.sign(p) * ((np.abs(p) * leak) >> 12)
        if self.leak_mode == "retain":                      # P * (4096 - leak) >> 12
            return p - ((p * (4096 - leak)) >> 12)
        raise ValueError(self.leak_mode)

    def step(self, frame):
        """frame: float [2, S, S] -> output spikes [n_out] (0/1)"""
        x = np.rint(np.asarray(frame, np.float64) * 4096).astype(np.int64)
        if self.P is None:
            self.P, shape = [], x.shape
            for (ci, co, k, s, p) in self.layers:
                h = (shape[1] + 2 * p - k) // s + 1
                shape = (co, h, h)
                self.P.append(np.zeros(shape, np.int64))
            self.prev = [np.zeros_like(p) for p in self.P]
        spikes_in = x
        outs = []
        for i, (ci, co, k, s, p) in enumerate(self.layers):
            inp = spikes_in if (i == 0 or not self.delay) else self.prev[i - 1]
            acc = conv(inp, self.W[i], s, p)
            cur = (acc >> self.in_shift) if i == 0 else (acc << self.w_shift)
            cur = cur + (self.B[i][:, None, None] << self.b_shift)
            if self.sat_current:
                cur = np.clip(cur, -self.lim, self.lim - 1)
            L, thr = int(self.leak[i]), int(self.thr[i])
            if self.order == "leak_add":            # decay the old potential, then integrate
                P = self.P[i] - self._leak(self.P[i], L) + cur
            elif self.order == "add_leak":          # integrate, then decay the sum
                P = self.P[i] + cur
                P = P - self._leak(P, L)
            else:                                   # "add_spike_leak": integrate, spike/reset, decay afterwards
                P = self.P[i] + cur
            if self.saturate:
                P = np.clip(P, -self.lim, self.lim - 1)
            sp = ((P >= thr) if self.cmp == "ge" else (P > thr)).astype(np.int64)
            P = P * (1 - sp) if self.reset_mode == "hard" else P - sp * thr
            if self.order == "add_spike_leak":
                P = P - self._leak(P, L)
            self.P[i] = P
            outs.append(sp)
            spikes_in = sp
        self.prev = outs
        return outs[-1].reshape(-1)

    def run(self, frames):
        return np.stack([self.step(f) for f in frames])
