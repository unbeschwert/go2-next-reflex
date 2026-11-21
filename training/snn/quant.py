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
    need = max(w_need, 1e-9) if q.layer_scaling else max(w_need, b_need, 1e-9)
    if q.weight_int_bits == "auto":
        # signed Q<I>.<F> holds [-2^(I-1), 2^(I-1) - 2^-F]: need 2^(I-1) >= need
        int_bits = min(max(1, math.ceil(math.log2(need)) + 1), q.weight_bits - 1)
    else:
        int_bits = q.weight_int_bits
    formats = Formats(int_bits, q.weight_bits - int_bits, q.potential_int_bits,
                      q.potential_bits - q.potential_int_bits)
    info = {"folded_weight_abs_max": weights.max().item(), f"folded_weight_abs_p{q.weight_percentile}": w_need,
            "folded_bias_abs_max": b_need, "chosen": formats.describe()}
    return formats, info


def layer_scales(model, formats: Formats, cfg: Config) -> tuple[list[int], list[dict]]:
    """Per layer, the smallest power of 2 k such that the layer's folded weights (at the configured
    percentile) and all its folded biases, divided by k, fit the weight format. 1 = unchanged."""
    q = cfg.quant
    limit = formats.weight.max_int / formats.weight.scale
    scales, info = [], []
    for i, (w, b) in enumerate(folded_params(model)):
        need = max(_percentile(w.flatten().abs(), q.weight_percentile), b.abs().max().item() if b.numel() else 0.0)
        k = 1
        if q.layer_scaling:
            while need / k > limit and k < q.max_layer_scale:
                k *= 2
        scales.append(k)
        info.append({"layer": i, "need": round(need, 4), "k": k, "fits": need / k <= limit})
    return scales, info


def apply_layer_scales(model, scales: list[int], threshold: float) -> None:
    """Divide each layer's BatchNorm affine (so the folded weights and bias) and its threshold by k."""
    with torch.no_grad():
        for layer, k in zip(model.net, scales):
            if k != 1:
                layer.bn.weight /= k
                layer.bn.bias /= k
            layer.lif.v_threshold = threshold / k


def saturation(values: list[int], fmt: FixedPointFormat) -> float:
    """Fraction of exported integers stuck at the format's min or max (i.e. clipped)."""
    if not values:
        return 0.0
    return sum(v <= fmt.min_int or v >= fmt.max_int for v in values) / len(values)
