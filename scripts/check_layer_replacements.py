"""Check which riverops layers without export_params can be rewritten as exportable layers.

The FPGA RTL only implements the layer types riverops exports (ConvLIF2d, ConvLIF2d with upscaling,
RecurrentConvLIF2d), and weights.txt is built from export_params(). This script checks, numerically, the
rewrites listed in doc/architecture-options.md ("Layers without export_params"):

    uv run scripts/check_layer_replacements.py
"""
import json

import torch
import torch.nn.functional as F
from torch import nn

from riverops.export import export_for_river
from riverops.nn import (FixedPointFormat, IFNode, NearestUpsample, QuantConfig, RecurrentSpikingConv2d,
                         SpikingConv2d, SpikingConvAvgPool2d, SpikingConvBN2d, SpikingConvBNAvgPool2d,
                         SpikingConvBNDropout2d, SpikingConvBNDropoutAvgPool2d, fuse_model, prepare_qat)
from riverops.nn.fusion import fuse_conv_bn
from riverops.nn.neuron import PLIFNode

torch.manual_seed(0)
B, T = 4, 6
QUANT = QuantConfig(FixedPointFormat(8, 7), FixedPointFormat(16, 12), True, True)  # Q1.7 / Q4.12


def randomize_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            with torch.no_grad():
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.2, 0.2)


def spikes_equal(a, b, x):
    ya, yb = a(x), b(x)
    return ya.shape == yb.shape and torch.equal(ya, yb), (ya != yb).float().mean().item(), ya.mean().item()


def graph(m, x):
    g = json.loads(export_for_river(m, x))
    return [(n["node_type"], n["kernel_size"], n["stride"], n["padding"][0], n.get("upscaling")) for n in g["nodes"]]


x = (torch.rand(B, T, 2, 20, 20) < 0.3).float() * 2.0
print("== 1. blocks that have fuse() but no export_params ==")
for cls, kw in [(SpikingConvBNDropout2d, dict(stride=2)),
                (SpikingConvAvgPool2d, dict(pool_kernel_size=2)),
                (SpikingConvBNAvgPool2d, dict(pool_kernel_size=4)),
                (SpikingConvBNDropoutAvgPool2d, dict(pool_kernel_size=2)),
                (SpikingConvBNAvgPool2d, dict(stride=2, pool_kernel_size=2))]:
    m = nn.Sequential(cls(2, 8, 3, padding=1, v_threshold=0.5, **kw)).eval()
    randomize_bn(m)
    f = fuse_model(m).eval()
    eq, diff, rate = spikes_equal(m, f, x)
    p = prepare_qat(f, QUANT)[0].export_params()
    print(f"{cls.__name__}{kw}: fused==original {eq} (diff {diff:.3f}, rate {rate:.3f}); "
          f"graph orig {graph(m, x)} fused {graph(f, x)}; export_params keys {sorted(p)}")

print("\n== 2. Dropout variant: same state_dict as the plain block ==")
a, b = SpikingConvBNDropout2d(2, 8, 3), SpikingConvBN2d(2, 8, 3)
print("state_dict keys equal:", list(a.state_dict()) == list(b.state_dict()))
b.load_state_dict(a.state_dict()); randomize_bn(a); b.load_state_dict(a.state_dict())
print("eval spikes equal:", torch.equal(a.eval()(x), b.eval()(x)))

print("\n== 3. IFNode vs PLIFNode with a fixed, huge tau (leak -> 0) ==")
cur = torch.randn(B, T, 8, 5, 5) * 0.6
ifn = IFNode(v_threshold=1.0)
plif = PLIFNode(initial_tau=1e12, learnable_leak=False, v_threshold=1.0)
print("float spikes equal:", torch.equal(ifn(cur), plif(cur)))
s = SpikingConv2d(2, 8, 3, initial_tau=1e12, learnable_leak=False)
s.conv.quant = QUANT
print("export leak integer:", s.export_params()["leak"], "threshold:", s.export_params()["threshold"])
s2 = SpikingConv2d(2, 8, 3, initial_tau=1e12, learnable_leak=False)
# PLIF block with neuron swapped for IFNode gives the same spikes as the huge-tau PLIF block
s2.load_state_dict(s.state_dict()); s2.lif = IFNode(v_threshold=1.0)
s.conv.quant = None
print("SpikingConv2d(IF) == SpikingConv2d(PLIF tau=1e12):", torch.equal(s(x), s2(x)))
try:
    print("graph with IF neuron inside:", graph(nn.Sequential(s2), x))
except Exception as e:
    print("graph with IF neuron inside fails:", type(e).__name__, e)

print("\n== 4. NearestUpsample -> SpikingConv2d ==")
up = nn.Sequential(SpikingConv2d(2, 4, 3, stride=2, padding=1), NearestUpsample(2), SpikingConv2d(4, 3, 3, padding=1)).eval()
print("graph:", graph(up, x))
try:
    graph(nn.Sequential(SpikingConv2d(2, 4, 3, padding=1), NearestUpsample(2)).eval(), x)
except Exception as e:
    print("upsample not followed by a conv fails:", type(e).__name__, e)

print("\n== 5. RecurrentSpikingConv2d ==")
r = RecurrentSpikingConv2d(2, 6, 3, stride=2, padding=1, v_threshold=0.5).eval()
net = nn.Sequential(r, SpikingConv2d(6, 3, 3, padding=1)).eval()
print("graph:", graph(net, x))
y = r(x)
print("output values:", sorted(set(y.unique().tolist())), "(spike sum of input + feedback path)")
rq = prepare_qat(r, QUANT)
r_in, r_rec = rq.input_block, rq.recurrent_block
print("params available per path:", {k: len(v) if isinstance(v, list) else v
                                       for k, v in r_in.export_params().items()},
      {k: len(v) if isinstance(v, list) else v for k, v in r_rec.export_params().items()})
# Can a feed-forward chain reproduce it? Unroll: the feedback reads the *previous* step's output, which a
# chain of [B,T,...] layers that only mix channels/space (never time) cannot provide.
ff = nn.Sequential(r.input_block).eval()
print("input path alone == recurrent layer:", torch.equal(ff(x), y))
# BatchNorm before the input path can be folded into the input conv after training.
bn = nn.BatchNorm2d(6).eval(); randomize_bn(bn)
folded = fuse_conv_bn(r.input_block.conv, bn)
print("BN folded into input conv: weight shape", tuple(folded.weight.shape), "bias", folded.bias is not None)
try:
    RecurrentSpikingConv2d(2, 6, 4, stride=2, padding=1)(x)
except Exception as e:
    print("even kernel (recurrent map changes size) fails:", type(e).__name__, str(e)[:90])

print("\n== 6. AvgPool on spikes, then conv: fold into one bigger conv ==")
P, k, sc = 2, 3, 1
w = torch.randn(4, 2, k, k); bias = torch.randn(4)
xin = x.flatten(0, 1)
ref = F.conv2d(F.avg_pool2d(xin, P), w, bias, stride=sc, padding=1)
wf = torch.kron(w, torch.ones(P, P)) / P ** 2
out = F.conv2d(xin, wf, bias, stride=sc * P, padding=1 * P)
print(f"conv(k{k},s{sc},p1)(avgpool{P}(x)) == conv(k{k*P},s{sc*P},p{P})(x):",
      torch.allclose(ref, out, atol=1e-5), "max err", (ref - out).abs().max().item())

print("\n== 7. Two convs with no neuron between: one conv (exact when the 2nd has padding 0) ==")
w1, b1 = torch.randn(6, 2, 3, 3), torch.randn(6)
w2, b2 = torch.randn(4, 6, 3, 3), torch.randn(4)
ref = F.conv2d(F.conv2d(xin, w1, b1, stride=2, padding=1), w2, b2, stride=1)
# composite kernel: w[o,i] = sum_c w2[o,c] (*) w1[c,i], dilated by stride 2
wc = torch.zeros(4, 2, 3 + 2 * (3 - 1), 3 + 2 * (3 - 1))
for a in range(3):
    for b_ in range(3):
        wc[:, :, 2 * a:2 * a + 3, 2 * b_:2 * b_ + 3] += torch.einsum("oc,cihw->oihw", w2[:, :, a, b_], w1)
bc = b2 + torch.einsum("oc,c->o", w2.sum(dim=(2, 3)), b1)
out = F.conv2d(xin, wc, bc, stride=2, padding=1)
print("conv2(conv1(x)) == conv_c(x):", torch.allclose(ref, out, atol=1e-4))
