"""Model shapes (no torch needed) and the riverops model built from them."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config, ConfigError

INPUT = -1   # source index of the network input in LayerShape.sources


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
    sources: tuple[int, ...] = ()   # layers whose spikes are summed into the input (INPUT = the frames)
    recurrent: bool = False         # RecurrentSpikingConv2d: + a feedback conv out_c -> out_c, stride 1

    @property
    def conv_kernel(self) -> int:
        """Kernel of the trained conv (before folding the avg-pool into it)."""
        return self.kernel - self.pool + 1

    @property
    def units(self) -> int:
        """Conv + neuron units in the layer: 2 for a recurrent layer (input and feedback path), else 1."""
        return 2 if self.recurrent else 1

    @property
    def weights(self) -> int:
        feedback = self.out_c * self.out_c * self.kernel * self.kernel if self.recurrent else 0
        return self.out_c * self.in_c * self.kernel * self.kernel + feedback

    @property
    def biases(self) -> int:
        return self.out_c * self.units

    @property
    def neurons(self) -> int:
        return self.out_c * self.out_h * self.out_w * self.units


def layer_shapes(cfg: Config) -> list[LayerShape]:
    """Propagate the input size through the layers and check what the export / FPGA needs."""
    shapes: list[LayerShape] = []
    for i, layer in enumerate(cfg.model.layers):
        if layer.inputs is None:
            sources = (INPUT,) if i == 0 else (i - 1,)
        else:
            sources = tuple(INPUT if s == "input" else s for s in layer.inputs)
        source_shapes = {(2, cfg.data.size, cfg.data.size) if s == INPUT else
                         (shapes[s].out_c, shapes[s].out_h, shapes[s].out_w) for s in sources}
        if len(source_shapes) != 1:
            raise ConfigError(f"model.layers[{i}]: the summed inputs {list(sources)} have different shapes "
                              f"{sorted(source_shapes)} (spikes can only be added element-wise)")
        c, h, w = source_shapes.pop()
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
        shapes.append(LayerShape(i, c, h, w, out, oh, ow, k_fpga, s_fpga, p, pool, sources, layer.recurrent))
    last = shapes[-1]
    if last.out_c != cfg.n_classes:
        raise ConfigError(f"the last layer must have one channel per class ({cfg.n_classes}, out: classes), "
                          f"got {last.out_c}")
    used = {src for s in shapes for src in s.sources}
    unused = [s.index for s in shapes[:-1] if s.index not in used]
    if unused:
        raise ConfigError(f"model.layers {unused} feed no other layer (only the last layer is an output)")
    # The output may be a classes x H x W map: the score of a class is its spike count summed over time
    # and over the H x W positions (done in software, on the board too), which makes the head
    # translation-invariant. H = W = 1 is the classic one-neuron-per-class output.
    return shapes


def has_pooling(cfg: Config) -> bool:
    return any(layer.pool > 1 for layer in cfg.model.layers)


def is_chain(cfg: Config) -> bool:
    """Every layer reads only the previous one and none is recurrent (all networks before 2026-11-24)."""
    return all(s.sources == ((INPUT,) if s.index == 0 else (s.index - 1,)) and not s.recurrent
               for s in layer_shapes(cfg))


def describe(shapes: list[LayerShape]) -> str:
    graph = any(s.sources != ((INPUT,) if s.index == 0 else (s.index - 1,)) or s.recurrent for s in shapes)
    rows = ["  layer  input        kernel/stride/pad  output       weights   neurons" + ("  from" if graph else "")]
    for s in shapes:
        geometry = f"{s.kernel:2d}/{s.stride}/{s.padding}"
        if s.pool > 1:
            geometry += f" (k{s.conv_kernel}+pool{s.pool})"
        if s.recurrent:
            geometry += " rec"
        source = " + ".join("input" if src == INPUT else str(src) for src in s.sources)
        rows.append(f"  {s.index:5d}  {s.in_c:3d}x{s.in_h:3d}x{s.in_w:<3d}  {geometry:<17s}  "
                    f"{s.out_c:3d}x{s.out_h:3d}x{s.out_w:<3d}  {s.weights:8,d}  {s.neurons:7,d}"
                    + (f"  {source}" if graph else ""))
    rows.append(f"  total weights {sum(s.weights for s in shapes):,}, neurons {sum(s.neurons for s in shapes):,}")
    return "\n".join(rows)


def neuron_blocks(layer) -> list:
    """The conv + neuron units of one layer, each with `.conv` and `.lif`: the layer itself, or for a
    riverops RecurrentSpikingConv2d its input and feedback SpikingConv2d."""
    if hasattr(layer, "recurrent_block"):
        return [layer.input_block, layer.recurrent_block]
    return [layer]


def build_model(cfg: Config, dropout: bool = False):
    """riverops model: SpikingConvBN2d layers (SpikingConvBNAvgPool2d for layers with `pool`,
    RecurrentSpikingConv2d for `recurrent` ones); forward returns spike counts per class [B, N], summed over
    time and over the output map's positions.

    `net` is an nn.Sequential for a chain (as before), else a LayerGraph: the same indexed layers, run in
    order, each on the sum of its sources' spikes. Both have the same state_dict keys (net.<i>.…).

    dropout=True (float training only) uses the Dropout variants for model.dropout_layers. Dropout has no
    parameters, so the state_dict is the same either way: quantization and export rebuild the model
    without it (riverops can't export the dropout layer type)."""
    from torch import nn
    from riverops.nn import RecurrentSpikingConv2d, SpikingConvBN2d, SpikingConvBNAvgPool2d
    from riverops.nn.spiking_conv import SpikingConvBNDropout2d, SpikingConvBNDropoutAvgPool2d

    neuron = cfg.model.neuron
    model_cfg = cfg.model
    shapes = layer_shapes(cfg)

    def layer(s):
        neuron_args = dict(initial_tau=neuron.initial_tau, learnable_leak=neuron.learnable_leak,
                           v_threshold=neuron.v_threshold, reset_mode=neuron.reset_mode)
        if s.recurrent:   # no BatchNorm inside, so the convs keep their own bias
            return RecurrentSpikingConv2d(s.in_c, s.out_c, s.kernel, stride=s.stride, padding=s.padding,
                                          bias=True, **neuron_args)
        common = dict(kernel_size=s.conv_kernel, stride=1 if s.pool > 1 else s.stride, padding=s.padding,
                      bias=False, first_layer=s.index == 0, **neuron_args)
        drop = dropout and model_cfg.dropout and s.index in model_cfg.dropout_layers
        if s.pool > 1:
            common["pool_kernel_size"] = s.pool
            if drop:
                return SpikingConvBNDropoutAvgPool2d(s.in_c, s.out_c, dropout_p=model_cfg.dropout, **common)
            return SpikingConvBNAvgPool2d(s.in_c, s.out_c, **common)
        if drop:
            return SpikingConvBNDropout2d(s.in_c, s.out_c, dropout_p=model_cfg.dropout, **common)
        return SpikingConvBN2d(s.in_c, s.out_c, **common)

    class LayerGraph(nn.ModuleList):
        """Layers run in index order; each gets the element-wise sum of its sources' spikes. Iterating and
        indexing work as for nn.Sequential, so per-layer code (quantization, export) is unchanged."""

        def __init__(self, layers, sources):
            super().__init__(layers)
            self.sources = [tuple(src) for src in sources]

        def forward(self, x):
            outputs = []
            for module, sources in zip(self, self.sources):
                total = None
                for src in sources:
                    value = x if src == INPUT else outputs[src]
                    total = value if total is None else total + value
                outputs.append(module(total))
            return outputs[-1]

    layers = [layer(s) for s in shapes]
    chain = is_chain(cfg)

    class SpikingClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers) if chain else LayerGraph(layers, [s.sources for s in shapes])

        def forward(self, x, warmup: int = 0):   # x: [B, warmup + T, 2, H, W]
            # Spikes during a warm-start prefix (training augmentation) are not counted. Summing over the
            # output positions (dims 2, 3 after the time sum) is the same as flatten for a 1x1 output.
            return self.net(x)[:, warmup:].sum(dim=1).sum(dim=(2, 3))

    return SpikingClassifier()
