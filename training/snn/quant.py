"""Choosing fixed-point formats and checking how well the model fits them.

Both formats are global (the firmware form takes one weight and one potential format), and
both are signed Q<int>.<frac> with the sign inside the int bits, as on riversdk.dev:
    weights + biases   Q<weight_int_bits>.<weight_bits - weight_int_bits>   (riverops: biases use the weight format)
    potentials, threshold, leak   Q<potential_int_bits>.<potential_bits - potential_int_bits>
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from riverops.nn import FixedPointFormat, QuantConfig, fuse_model
from riverops.nn.spiking_conv import SpikingConv2d

from .config import Config


@dataclass(frozen=True)
class Formats:
    weight_int: int
    weight_frac: int
    potential_int: int
    potential_frac: int

    @property
    def weight(self) -> FixedPointFormat:
        return FixedPointFormat(total_bits=self.weight_int + self.weight_frac, frac_bits=self.weight_frac)

    @property
    def potential(self) -> FixedPointFormat:
        return FixedPointFormat(total_bits=self.potential_int + self.potential_frac, frac_bits=self.potential_frac)

    def riverops(self, qat: bool = True) -> QuantConfig:
        return QuantConfig(weight_format=self.weight, potential_format=self.potential,
                           enable_weight_quant=True, enable_potential_quant=True, qat_mode=qat)

    def describe(self) -> str:
        return f"weights Q{self.weight_int}.{self.weight_frac}, potentials Q{self.potential_int}.{self.potential_frac}"

    def to_dict(self) -> dict:
        return {"weight_int": self.weight_int, "weight_frac": self.weight_frac,
                "potential_int": self.potential_int, "potential_frac": self.potential_frac}


def folded_params(model) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Weights and biases per layer after folding BatchNorm, as they will be exported."""
    fused = fuse_model(model).eval()
    out = []
    for layer in fused.net:
        if not isinstance(layer, SpikingConv2d):
            raise TypeError(f"unexpected layer after fusion: {type(layer).__name__}")
        bias = layer.conv.bias if layer.conv.bias is not None else torch.zeros(layer.conv.out_channels)
        out.append((layer.conv.weight.detach().cpu(), bias.detach().cpu()))
    return out


def _percentile(values: torch.Tensor, pct: float) -> float:
    return torch.quantile(values.float(), pct / 100).item() if len(values) < 16_000_000 else values.max().item()


def choose_formats(model, cfg: Config) -> tuple[Formats, dict]:
    """Pick the weight int bits: the fewest that hold the folded weights (at the configured
    percentile) and all folded biases. Fewer int bits = more fractional precision for the same
    memory, since the total width (and so the BRAM cost) is fixed by quant.weight_bits.

    With quant.layer_scaling, the format only has to hold the weights of all layers together (in
    practice the big layers); a layer that doesn't fit is scaled instead (layer_scales)."""
    q = cfg.quant
    params = folded_params(model)
    weights = torch.cat([w.flatten().abs() for w, _ in params])
    biases = torch.cat([b.flatten().abs() for _, b in params])
    w_need = _percentile(weights, q.weight_percentile)
    b_need = biases.max().item() if len(biases) else 0.0
    need = max(w_need, b_need, 1e-9)
    if q.weight_int_bits != "auto":
        int_bits = q.weight_int_bits
    elif q.layer_scaling:
        # every layer gets its own int bits through its scale k = 2^(own - shared), so the shared format
        # is the smallest any layer needs (in practice the big layers', not layer 0's)
        int_bits = min(_int_bits(n, q.weight_bits) for n in layer_needs(params, cfg))
    else:
        int_bits = _int_bits(need, q.weight_bits)
    formats = Formats(int_bits, q.weight_bits - int_bits, q.potential_int_bits,
                      q.potential_bits - q.potential_int_bits)
    info = {"folded_weight_abs_max": weights.max().item(), f"folded_weight_abs_p{q.weight_percentile}": w_need,
            "folded_bias_abs_max": b_need, "chosen": formats.describe(),
            "layers": layer_stats(params, formats.weight)}
    return formats, info


def layer_stats(params: list[tuple[torch.Tensor, torch.Tensor]], fmt: FixedPointFormat) -> list[dict]:
    """How each layer's BN-folded weights and biases fare in the weight format `fmt`.

    clipped: share of weights outside the format's range; zeroed: share of non-zero weights that round to 0;
    sqnr_db: signal-to-quantization-noise ratio of the weights (rounding + clipping), higher is better
    (each extra bit of precision adds ~6 dB); bias_clipped: share of biases outside the range.
    """
    lsb = 2.0 ** -fmt.frac_bits
    out = []
    for i, (w, b) in enumerate(params):
        q = torch.round(w / lsb)
        noise = ((w - q.clamp(fmt.min_int, fmt.max_int) * lsb) ** 2).sum().item()
        sqnr = 10 * math.log10((w ** 2).sum().item() / noise) if noise > 0 else float("inf")
        qb = torch.round(b / lsb)
        out.append({
            "layer": i,
            "weights": w.numel(),
            "abs_max": round(w.abs().max().item(), 4),
            "clipped": ((q > fmt.max_int) | (q < fmt.min_int)).float().mean().item(),
            "zeroed": ((q == 0) & (w != 0)).float().mean().item(),
            "sqnr_db": round(sqnr, 2),
            "bias_clipped": ((qb > fmt.max_int) | (qb < fmt.min_int)).float().mean().item() if b.numel() else 0.0,
        })
    return out


def describe_layers(layers: list[dict]) -> str:
    """Text table of layer_stats() (also in choose_formats(...)[1]["layers"])."""
    rows = ["  layer   weights   |w| max   clipped   zeroed   SQNR dB   bias clipped"]
    for s in layers:
        rows.append(f"  {s['layer']:5d}  {s['weights']:8d}  {s['abs_max']:7.3f}  {100 * s['clipped']:6.2f} %"
                    f"  {100 * s['zeroed']:5.1f} %  {s['sqnr_db']:8.1f}  {100 * s['bias_clipped']:9.2f} %")
    return "\n".join(rows)


def _int_bits(need: float, bits: int) -> int:
    """Fewest int bits of a signed Q<I>.<F> (range [-2^(I-1), 2^(I-1))) that hold `need`."""
    return min(max(1, math.ceil(math.log2(max(need, 1e-9))) + 1), bits - 1)


def layer_needs(params: list[tuple[torch.Tensor, torch.Tensor]], cfg: Config) -> list[float]:
    """Per layer, the magnitude its format must hold: weights at the configured percentile, and with
    quant.layer_scale_biases also all its biases."""
    q = cfg.quant
    return [max(_percentile(w.flatten().abs(), q.weight_percentile),
                b.abs().max().item() if (b.numel() and q.layer_scale_biases) else 0.0) for w, b in params]


def layer_scales(model, formats: Formats, cfg: Config) -> tuple[list[int], list[dict]]:
    """Per layer, the smallest power of 2 k such that the layer's folded weights (at the configured
    percentile) and, with quant.layer_scale_biases, all its folded biases, divided by k, fit the weight
    format. 1 = unchanged."""
    q = cfg.quant
    limit = formats.weight.max_int / formats.weight.scale
    scales, info = [], []
    for i, need in enumerate(layer_needs(folded_params(model), cfg)):
        k = 1
        if q.layer_scaling:
            while need / k > limit and k < q.max_layer_scale:
                k *= 2
        scales.append(k)
        info.append({"layer": i, "need": round(need, 4), "k": k, "fits": need / k <= limit})
    return scales, info


def apply_layer_scales(model, scales: list[int], threshold: float) -> None:
    """Divide each layer's folded weights and bias, and its threshold, by k: through the BatchNorm affine
    when the layer still has BatchNorm, else (an already fused SpikingConv2d) the conv itself."""
    with torch.no_grad():
        for layer, k in zip(model.net, scales):
            if k != 1:
                if getattr(layer, "bn", None) is not None:
                    layer.bn.weight /= k
                    layer.bn.bias /= k
                else:
                    layer.conv.weight /= k
                    if layer.conv.bias is not None:
                        layer.conv.bias /= k
            layer.lif.v_threshold = threshold / k


def saturation(values: list[int], fmt: FixedPointFormat) -> float:
    """Fraction of exported integers stuck at the format's min or max (i.e. clipped)."""
    if not values:
        return 0.0
    return sum(v <= fmt.min_int or v >= fmt.max_int for v in values) / len(values)
