"""Model shapes (no torch needed) and the riverops model built from them."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config, ConfigError


@dataclass(frozen=True)
class LayerShape:
    index: int
    in_c: int
    in_h: int
    in_w: int
    out_c: int
    out_h: int
    out_w: int
    kernel: int        # as on the FPGA: with pooling, the fused kernel (conv kernel + pool - 1)
    stride: int        # as on the FPGA: with pooling, the pool size
    padding: int
    pool: int = 1      # avg-pool size in training (1 = none); the conv itself then has stride 1

    @property
    def conv_kernel(self) -> int:
        """Kernel of the trained conv (before folding the avg-pool into it)."""
        return self.kernel - self.pool + 1

    @property
    def weights(self) -> int:
        return self.out_c * self.in_c * self.kernel * self.kernel

    @property
    def neurons(self) -> int:
        return self.out_c * self.out_h * self.out_w


def layer_shapes(cfg: Config) -> list[LayerShape]:
    """Propagate the input size through the layers and check what the export / FPGA needs."""
    c, h, w = 2, cfg.data.size, cfg.data.size
    shapes = []
    for i, layer in enumerate(cfg.model.layers):
        out = cfg.n_classes if layer.out == "classes" else layer.out
        if not isinstance(out, int) or out < 1:
            raise ConfigError(f"model.layers[{i}].out must be a positive int or 'classes'")
        k, s, p, pool = layer.kernel, layer.stride, layer.padding, layer.pool
        if not all(isinstance(v, int) for v in (k, s, p, pool)) or k < 1 or s < 1 or p < 0 or pool < 1:
            raise ConfigError(f"model.layers[{i}]: kernel/stride/pool must be >= 1 and padding >= 0 "
                              "(only square kernels/strides and integer padding can be exported)")
        if pool > 1 and s != 1:
            raise ConfigError(f"model.layers[{i}]: pool needs stride 1 (riverops folds conv + avg-pool "
                              "exactly only for a stride-1 conv); the pool does the downsampling")
        # With pooling, the FPGA runs one conv of kernel k + pool - 1 and stride pool; its output size
        # equals the conv's (stride 1) output pooled by `pool`.
        k_fpga, s_fpga = (k + pool - 1, pool) if pool > 1 else (k, s)
        oh, ow = (h + 2 * p - k_fpga) // s_fpga + 1, (w + 2 * p - k_fpga) // s_fpga + 1
        if oh < 1 or ow < 1:
            raise ConfigError(f"model.layers[{i}]: kernel {k_fpga} does not fit a {h}x{w} input")
        shapes.append(LayerShape(i, c, h, w, out, oh, ow, k_fpga, s_fpga, p, pool))
        c, h, w = out, oh, ow
    last = shapes[-1]
    if last.out_c != cfg.n_classes:
        raise ConfigError(f"the last layer must have one channel per class ({cfg.n_classes}, out: classes), "
                          f"got {last.out_c}")
    # The output may be a classes x H x W map: the score of a class is its spike count summed over time
    # and over the H x W positions (done in software, on the board too), which makes the head
    # translation-invariant. H = W = 1 is the classic one-neuron-per-class output.
    return shapes


def has_pooling(cfg: Config) -> bool:
    return any(layer.pool > 1 for layer in cfg.model.layers)


def describe(shapes: list[LayerShape]) -> str:
    rows = ["  layer  input        kernel/stride/pad  output       weights   neurons"]
    for s in shapes:
        geometry = f"{s.kernel:2d}/{s.stride}/{s.padding}"
        if s.pool > 1:
            geometry += f" (k{s.conv_kernel}+pool{s.pool})"
        rows.append(f"  {s.index:5d}  {s.in_c:3d}x{s.in_h:3d}x{s.in_w:<3d}  {geometry:<17s}  "
                    f"{s.out_c:3d}x{s.out_h:3d}x{s.out_w:<3d}  {s.weights:8,d}  {s.neurons:7,d}")
    rows.append(f"  total weights {sum(s.weights for s in shapes):,}, neurons {sum(s.neurons for s in shapes):,}")
    return "\n".join(rows)


def build_model(cfg: Config, dropout: bool = False):
    """riverops model: a stack of SpikingConvBN2d (SpikingConvBNAvgPool2d for layers with `pool`);
    forward returns spike counts per class [B, N], summed over time and over the output map's positions.

    dropout=True (float training only) uses the Dropout variants for model.dropout_layers. Dropout has no
    parameters, so the state_dict is the same either way: quantization and export rebuild the model
    without it (riverops can't export the dropout layer type)."""
    from torch import nn
    from riverops.nn import SpikingConvBN2d, SpikingConvBNAvgPool2d
    from riverops.nn.spiking_conv import SpikingConvBNDropout2d, SpikingConvBNDropoutAvgPool2d

    neuron = cfg.model.neuron
    model_cfg = cfg.model

    def layer(s):
        common = dict(kernel_size=s.conv_kernel, stride=1 if s.pool > 1 else s.stride, padding=s.padding,
                      bias=False, first_layer=s.index == 0, initial_tau=neuron.initial_tau,
                      learnable_leak=neuron.learnable_leak, v_threshold=neuron.v_threshold,
                      reset_mode=neuron.reset_mode)
        drop = dropout and model_cfg.dropout and s.index in model_cfg.dropout_layers
        if s.pool > 1:
            common["pool_kernel_size"] = s.pool
            if drop:
                return SpikingConvBNDropoutAvgPool2d(s.in_c, s.out_c, dropout_p=model_cfg.dropout, **common)
            return SpikingConvBNAvgPool2d(s.in_c, s.out_c, **common)
        if drop:
            return SpikingConvBNDropout2d(s.in_c, s.out_c, dropout_p=model_cfg.dropout, **common)
        return SpikingConvBN2d(s.in_c, s.out_c, **common)

    class SpikingClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*[layer(s) for s in layer_shapes(cfg)])

        def forward(self, x, warmup: int = 0):   # x: [B, warmup + T, 2, H, W]
            # Spikes during a warm-start prefix (training augmentation) are not counted. Summing over the
            # output positions (dims 2, 3 after the time sum) is the same as flatten for a 1x1 output.
            return self.net(x)[:, warmup:].sum(dim=1).sum(dim=(2, 3))

    return SpikingClassifier()
