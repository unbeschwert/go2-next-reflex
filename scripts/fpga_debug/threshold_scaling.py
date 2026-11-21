"""Per-layer weight range through the threshold: does scaling layer 0 by 1/k help, without retraining?

A LIF layer spikes identically if its weights, bias and threshold are all divided by k (the potential just
scales by 1/k; leak is multiplicative, hard reset goes to 0). weights.txt has one threshold per layer, so
this gives layer 0 a wider weight range (and coarser steps) while all layers keep 8-bit Q2.6 weights.
Starts from the run's float model (float_best.pt) and quantizes it without fine-tuning (post-training
quantization): a QAT checkpoint has already adapted to the clipping, so it is not a fair comparison.
Usage: uv run --group train python scripts/fpga_debug/threshold_scaling.py [run=runs/motion-buddy-40-v1]
"""
import copy, sys
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["."]
from riverops.nn import prepare_ptq, prepare_qat
from training.snn.config import load_config
from training.snn.data import discover
from training.snn.events import read_events, events_to_counts
from training.snn.model import build_model
from training.snn.train import load_qat_model, zero_tiny_gammas

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/motion-buddy-40-v1")
cfg = load_config(run / "config.yaml")
_, formats = load_qat_model(run / "qat_best.pt", cfg)       # only for the formats the run chose
qat = build_model(cfg)
qat.load_state_dict(torch.load(run / "float_best.pt", map_location="cpu", weights_only=True)["state_dict"])
zero_tiny_gammas(qat)
qat.eval()
T, ds = cfg.data.time_steps, cfg.data.downsampling
splits = discover(cfg)
data = {}
for split in ("val", "test"):
    clips = splits[split]
    x = np.stack([events_to_counts(read_events(c.path), T, cfg.data.time_window_us, ds) for c in clips]).astype(np.float32) / ds**2
    data[split] = (torch.from_numpy(x), torch.tensor([c.label for c in clips]))
W = formats.weight
lim = W.max_int / W.scale

def scaled(k):
    m = copy.deepcopy(qat)
    layer = m.net[0]
    with torch.no_grad():
        layer.bn.weight /= k; layer.bn.bias /= k
    layer.lif.v_threshold = cfg.model.neuron.v_threshold / k
    return m

print(f"{run.name} float model, no fine-tuning: weights {formats.describe()}, layer 0 scaled by 1/k (weights, bias, threshold)")
for k in (1, 2, 4, 8):
    m = scaled(k)
    fixed = prepare_ptq(prepare_qat(m, formats.riverops(), fused_bn=True).eval(), formats.riverops(qat=False), fuse=True).eval()
    # clipping of layer 0's folded weights at this scale
    bn, conv = m.net[0].bn, m.net[0].conv
    folded = conv.weight * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1)
    clip = (folded.abs() > lim).float().mean().item()
    acc = {}
    with torch.no_grad():
        for split, (x, y) in data.items():
            pred = torch.cat([fixed(x[i:i + 64]).argmax(1) for i in range(0, len(x), 64)])
            acc[split] = (pred == y).float().mean().item()
    print(f"  k={k}: threshold {1 / k:.4f}, layer-0 weights clipped {clip:6.1%}, effective step {k / W.scale:.4f}"
          f"   fixed point val {acc['val']:.1%}  test {acc['test']:.1%}", flush=True)
