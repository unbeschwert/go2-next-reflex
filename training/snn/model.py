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
    kernel: int
    stride: int
    padding: int

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
        k, s, p = layer.kernel, layer.stride, layer.padding
        if not all(isinstance(v, int) for v in (k, s, p)) or k < 1 or s < 1 or p < 0:
            raise ConfigError(f"model.layers[{i}]: kernel/stride must be >= 1 and padding >= 0 "
                              "(only square kernels/strides and integer padding can be exported)")
        oh, ow = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
        if oh < 1 or ow < 1:
            raise ConfigError(f"model.layers[{i}]: kernel {k} does not fit a {h}x{w} input")
        shapes.append(LayerShape(i, c, h, w, out, oh, ow, k, s, p))
        c, h, w = out, oh, ow
    last = shapes[-1]
    if (last.out_c, last.out_h, last.out_w) != (cfg.n_classes, 1, 1):
        hint = (f" The last layer sees a {last.in_h}x{last.in_w} input: use kernel={last.in_h}, padding=0, "
                "or shrink earlier.") if last.out_h != 1 else ""
        raise ConfigError(f"the output must be {cfg.n_classes}x1x1 (one neuron per class, as spleat-server "
                          f"expects), but the last layer gives {last.out_c}x{last.out_h}x{last.out_w}.{hint}")
    return shapes


def describe(shapes: list[LayerShape]) -> str:
    rows = ["  layer  input        kernel/stride/pad  output       weights   neurons"]
    for s in shapes:
        rows.append(f"  {s.index:5d}  {s.in_c:3d}x{s.in_h:3d}x{s.in_w:<3d}  {s.kernel:2d}/{s.stride}/{s.padding}"
                    f"             {s.out_c:3d}x{s.out_h:3d}x{s.out_w:<3d}  {s.weights:8,d}  {s.neurons:7,d}")
    rows.append(f"  total weights {sum(s.weights for s in shapes):,}, neurons {sum(s.neurons for s in shapes):,}")
    return "\n".join(rows)


def build_model(cfg: Config):
    """riverops model: a stack of SpikingConvBN2d; forward returns spike counts per class [B, N]."""
    from torch import nn
    from riverops.nn import SpikingConvBN2d

    neuron = cfg.model.neuron

    class SpikingClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*[
                SpikingConvBN2d(s.in_c, s.out_c, kernel_size=s.kernel, stride=s.stride, padding=s.padding,
                                bias=False, first_layer=s.index == 0, initial_tau=neuron.initial_tau,
                                learnable_leak=neuron.learnable_leak, v_threshold=neuron.v_threshold,
                                reset_mode=neuron.reset_mode)
                for s in layer_shapes(cfg)
            ])

        def forward(self, x, warmup: int = 0):   # x: [B, warmup + T, 2, H, W]
            # Spikes during a warm-start prefix (training augmentation) are not counted.
            return self.net(x)[:, warmup:].sum(dim=1).flatten(start_dim=1)

    return SpikingClassifier()
