# Training an SNN with `riverops`

> Our pipeline that automates all of this on the GPU server: [training/README.md](../training/README.md).

Condensed from the RIVER SDK notebook "Train a model" (`train_motion_buddy`) and the `riverops` 0.2.0 docs.

## Install

`riverops` is a PyTorch library for spiking networks. It needs Python ≥ 3.10.

```bash
uv add ./data/riverops-0.2.0-py3-none-any.whl torch
```

The `riverops` docs still say `riverops-0.1.0`, but the download is **0.2.0**, and the notebook uses 0.2.0 names such as `SpikingConvBN2d`. See the changelog note at the end.

## The reference pipeline (Motion Buddy notebook)

### Settings

| Setting | Value |
|---|---|
| `TIME_STEPS` | 100 |
| `TIME_WINDOW_US` | 10,000 (10 ms per step, so 1 s per clip) |
| `IMAGE_SIZE` | 32 (from 320, averaged down) |
| `BATCH_SIZE` | 4 |
| `EPOCHS` | 10 |
| `LEARNING_RATE` | 0.003 (Adam) |
| `WEIGHT_DECAY` | 1e-4 |
| `MAX_SHIFT` | 2 (random ±2 px shift for training only) |
| `SEED` | 42 |

### Classes and split

`CLASS_NAMES = ("Sit", "Lay down", "Hello", "Dance", "No action")`. Training uses folders 01–03, validation 04, test 05.

The notebook opens `DuranceAI_motion_buddy_dataset.zip`. Our copy is called `durance-ai-handgesture-recognition.zip` and is already extracted, so point `dataset_dir` at `data/motion_buddy_dataset` and skip the unzip step.

### Events → input tensor

1. Take events from the **first 1 s** of the clip and put each into one of 100 time bins of 10 ms.
2. Count them into `(100, 2, 320, 320)`: time, polarity, y, x.
3. Shrink to 32×32 with `F.interpolate(..., size=(32, 32), mode="area")`, i.e. averaging blocks of pixels.
4. The result per clip is `(T=100, C=2, H=32, W=32)`, and batches are `[B, T, C, H, W]`.

The prepared frames for folders 01–04 take about 1 GB of RAM.

### Model

```python
lif = {"initial_tau": 1.0 / (1.0 - 0.93),   # fixed leak, decay 0.93 per step
       "learnable_leak": False, "v_threshold": 1.0, "reset_mode": "HARD"}
net = nn.Sequential(
    SpikingConvBN2d(2,   16, kernel_size=5, stride=2, padding=1, bias=False, first_layer=True, **lif),
    SpikingConvBN2d(16,  16, kernel_size=3, padding=1,           bias=False, **lif),
    SpikingConvBN2d(16,   8, kernel_size=6, stride=2, padding=2, bias=False, **lif),
    SpikingConvBN2d(8,  256, kernel_size=6, stride=2,            bias=False, **lif),
    SpikingConvBN2d(256,  5, kernel_size=1,                      bias=False, **lif),
)
# forward: scores = net(x).sum(dim=1).flatten(start_dim=1)   → spike count per class, shape [B, 5]
```

The prediction is the class with the most output spikes, i.e. `scores.argmax(1)`.

### Loss and training

- **Target spike counts:** 80% of the time steps (80) for the correct class and 20% (20) for the others.
- **Loss:** `mean((scores − targets)²) / T`.
- **Each step:**
  1. backward
  2. clip the gradient norm to 1.0
  3. clamp weights to [−1, 1], since they must fit 8-bit fixed point later
  4. `optimizer.step()`
- Keep the checkpoint with the best **validation** accuracy (`motion_buddy_best.pt`), then evaluate once on the test folder.

### Export (what the FPGA needs)

```python
from riverops.export import export_for_river
model_json = export_for_river(model.net, example_input)        # layer structure → motion_buddy_model.json

from riverops.nn import FixedPointFormat, QuantConfig, prepare_qat
qc = QuantConfig(weight_format=FixedPointFormat(total_bits=8, frac_bits=7),      # Q1.7
                 potential_format=FixedPointFormat(total_bits=16, frac_bits=12), # Q4.12
                 enable_weight_quant=True, enable_potential_quant=True)
export_model = prepare_qat(model, qc).eval()
for i, layer in enumerate(export_model.net):
    p = layer.export_params()        # BatchNorm folded in, values as integers
    # p["weights"], p["bias"], p["threshold"], p["leak"]
```

- Export `model.net`, the spiking layers only. The sum over time in `forward()` isn't part of the hardware graph; the software reading the FPGA output does it.
- Parameters are written in a fixed order: all weights → all biases → `threshold, leak` per layer. They go to `motion_buddy_parameters.json` and, one integer per line, to `motion_buddy_parameters.txt`, which becomes `weights.txt` on the Kria (see [deploy-model.md](deploy-model.md)).

| Parameter | Format | Bits |
|---|---|---|
| Weights | Q1.7 | 8 |
| Biases | Q1.7 | 8 |
| Threshold | Q4.12 | 16 |
| Leak | Q4.12 | 16 |

## `riverops` essentials

### Input shape

Always `[batch, time, channels, height, width]`, even for one time step (`T=1`). `riverops` doesn't add a time dimension or turn images into spikes for you.

### State

Each forward call starts with zero membrane potential and carries it over the time steps **within that call**. Pass the whole sequence in one call. Calling once per time step loses the state; use the functional API if you need step-by-step control.

### Layers (`riverops.nn`)

| Need | Use |
|---|---|
| Convolution + neuron | `SpikingConv2d` |
| … + batch norm | `SpikingConvBN2d` |
| … + dropout | `SpikingConvBNDropout2d` |
| Pool before the neuron | `SpikingConvAvgPool2d` variants |
| Shrink / enlarge a sequence | `AvgPool2d` / `NearestUpsample` |
| Feedback | `RecurrentSpikingConv2d` (its output can be 2) |
| Neurons | `PLIFNode` (leaky, learnable leak by default), `IFNode` (no leak) |

### Neuron arguments

| Argument | Meaning |
|---|---|
| `initial_tau` | Membrane time constant. With a learnable leak it must be > 1. |
| `learnable_leak` | `True` by default in 0.2.0. Set `False` for a fixed leak, and keep it `False` when reloading such a model. |
| `v_threshold` | Potential needed to spike. |
| `reset_mode` | `"HARD"` (to zero) or `"SOFT"` (subtract the threshold). The export does **not** record the reset mode. |

Set `first_layer=True` on the layer that receives the input, since that input isn't spikes.

### Training

Training uses surrogate gradients (e.g. `surrogate.ATan(alpha=2.0)`) and ordinary PyTorch losses and optimizers. The docs' generic example uses `nn.Linear` + `CrossEntropyLoss`, but **a Linear head can't be exported**. For the FPGA, end with a conv layer and count spikes, like the notebook does.

### Fusion

`fuse_model(model.eval())` folds BatchNorm (and supported avg-pool) into the convolutions. It returns a copy; compare its outputs with the original's.

### Quantization

- **`FixedPointFormat(total_bits, frac_bits, signed=True)`:** e.g. 8 bits with 4 fractional bits has steps of 1/16 and a range of −8 … 7.94.
- **`QuantConfig(...)`:** choose formats for weights, biases, potentials and thresholds, and switch each one on with its `enable_*_quant` flag.
- **Quantization-aware training (QAT):**
  1. `qat = prepare_qat(model, qc, fused_bn=True).train()`
  2. Create a **new** optimizer for `qat`.
  3. Fine-tune.
  4. `fuse_model(qat.eval())`.
- **Post-training evaluation only:** `prepare_ptq(model, qc, fuse=True)`.
- Fixed-point emulation is approximate, not bit-exact with the hardware. Compare accuracy as well as spike patterns: small changes near the threshold spread to later time steps.

### Export limits

Only a subset of operations is supported: no string padding, and cycles or multiple inputs/outputs aren't fully supported. Start from a small conv stack known to work, like the notebook's.

### Changelog, 0.2.0 vs older examples

| Older name | 0.2.0 name |
|---|---|
| `ConvLIF2d` | `SpikingConv2d` |
| `ConvBNLIF2d` | `SpikingConvBN2d` |
| `RecurrentConvLIF2d` | `RecurrentSpikingConv2d` |
| `LIFNode(tau=…)` | `PLIFNode(initial_tau=…, learnable_leak=False)` |
| `tau` / `threshold` arguments | `initial_tau` / `v_threshold` |
| `enable_state_quant` | `enable_potential_quant` |

## Before deploying, check the input size

`config.yaml` and despot need an input size of the form 320 ÷ 2ⁿ: 320, 160, 80, 40, 20, … The notebook's **32×32 isn't one of those.** The example `config.yaml` uses **80×80**. Either train at 80×80 or 40×40 (`F.interpolate` to that size, or `downsampling` 4 or 8), or confirm with the organisers how 32×32 is meant to be fed.
