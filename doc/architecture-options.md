# Network architecture: what riverops and the FPGA allow

What can change in the network's structure, as of riverops 0.2.0 and the SPLEAT firmware we use. The findings come from reading riverops' source (`riverops/export`, `riverops/nn`) and exporting candidate networks with `export_for_river`. Two of the options are implemented (spatial head, avg-pool downsampling, [below](#implemented)).

## What reaches the FPGA

The firmware is generated from `model.json`, a `RiverModelGraph`: nodes, edges between them, and outputs. Only what riverops' `export_to_river_model_graph` can lower is deployable:

| Building block (riverops) | In model.json | Used on our board so far | Notes |
|---|---|---|---|
| `SpikingConv2d`, `SpikingConvBN2d`, `SpikingConvBNDropout2d` | ✓ `ConvLIF2d` | ✓ every deployed model (ours, PR #4, the organisers' example) | square kernel and stride only, no dilation, no groups |
| `SpikingConv(BN)AvgPool2d`: conv → avg-pool → neuron | ✓ one `ConvLIF2d` (pool folded into the conv) | ✓ in effect: the firmware sees an ordinary conv | **implemented**, see below |
| Adding spikes, `conv(a + b)` (skip / residual connections) | ✓ two incoming edges into one `ConvLIF2d` | ✗ | shapes must match; a sum of spikes can be 2 |
| Several outputs (e.g. two heads) | ✓ `outputs: [a, b]` | ✗ | riverops' own README: multiple outputs "can lead to unpredictable behaviour" |
| `NearestUpsample` → conv | ✓ `ConvLIF2d` with `upscaling: 2` (one level only) | ✗ | decoder-style; little use for classification |
| `RecurrentSpikingConv2d`: input conv + feedback conv | ✓ `RecurrentConvLIF2d` | ✗ | no BatchNorm inside, no `export_params`, so the weights.txt layout is unknown |
| `torch.cat`, chunk, slicing | ✗ "not implemented in hardware" | – | branches can only merge by addition |
| Global average pooling (`ReduceMean`), Linear layers | ✗ | – | a conv whose kernel covers the whole map is the "fully connected" layer |
| Depthwise, grouped or dilated convs; non-square kernels | ✗ | – | riverops' `Conv2d` forces groups = 1, dilation = 1 |
| Neurons: `PLIFNode` (learnable leak), `IFNode` | threshold and leak per layer | ✓ PLIF | one leak per layer, not per channel |
| `reset_mode: SOFT` | not recorded by the export | – | the firmware resets hard: keep `HARD` |

**Headroom.** The v3 network uses 33 of the 108 BRAM36 we budget. PR #4 measured an FPGA latency of 0.15–0.74 ms per 10 ms frame, growing with spike activity. A network 2–3× larger still fits in memory and in real time.

**Our pipeline assumes a chain** of layers (`nn.Sequential`) in `training/snn/model.py`, `quant.py`, `export.py` (weights.txt = all weights, all biases, then threshold and leak per layer, in layer order) and `fpga.py`. Changes that keep a chain are small. A DAG needs graph-aware versions of all four: skips, two outputs, recurrence.

## Options, ranked for our problem

The remaining errors are on test person 14, who is small, at the left edge, with a moving distractor. Runs with different seeds differ by up to ~7 test points ([training-log.md](training-log.md) #23/#24).

**Chain-preserving, firmware-safe:**
1. **Spatial head** (implemented). The output layer keeps a small map, for example 5 × 2 × 2. A class's score is its spike count summed over time *and* positions, which is global sum pooling done in software. Every position uses the same classifier weights, so the head doesn't care where in the frame the gesture is. The old 2×2 → 1×1 k2 head had separate weights per position.
2. **Anti-aliased downsampling** (implemented). A stride-1 conv followed by average pooling instead of a strided conv, which is smoother and more robust to small shifts (Zhang 2019, *Making Convolutional Networks Shift-Invariant Again*). riverops folds the pool into the conv, so the FPGA runs one conv of kernel `k + pool − 1` and stride `pool`, at the same cost as a strided conv of that size.
3. **Wider layers** (e.g. 32 channels in layers 0–2): still a chain, about 2× BRAM. Training accuracy under augmentation is 65–70 %, so the network may be capacity-limited.

**Export works, firmware unproven** (a test firmware build comes first, `scripts/firmware_job.py`):

4. **Skip connections** by adding spikes: deeper networks.
5. **Two heads / an ensemble inside one network:** would average away part of the seed noise.
6. **A recurrent layer:** memory longer than the leak's τ ≈ 2–4 steps, for temporal gestures (dance vs lay-down). It's the most work: no BatchNorm, unknown weights layout, and a recurrent path is needed in quantization.

## Implemented

### Spatial head

The last layer may now output `classes × H × W` (before: only `classes × 1 × 1`). The model sums over time and over the H × W positions, so for a 1 × 1 output the result is exactly as before: #14 re-exports byte for byte. Example (`configs/arch/head.yaml`): replace the final `{out: classes, kernel: 2}` on the 2 × 2 map with `{out: classes, kernel: 1}`.

On the board, `spleat-server-config.yaml` gets `outputs: {channels: 5, height: H, width: W}`. Each frame's payload is then 5·H·W values, and **the reader must add each class's H·W values**. `kria/live_gesture_bridge.py` (`N_OUT`) and `scripts/fpga.py` expect 5 values today, so they need that change before deploying such a model.

### Avg-pool downsampling

A layer can have `pool: P` (with `stride: 1`): conv `kernel` with stride 1 → BatchNorm → average pool P × P → neuron (`SpikingConvBNAvgPool2d`, Dropout variant in training). On the FPGA it is one conv of kernel `kernel + P − 1`, stride `P`, same padding, and the output size equals the pooled one.

```yaml
- {out: 16, kernel: 5, padding: 1, pool: 4}     # 80 → 19, FPGA: k8 s4
- {out: 16, kernel: 5, padding: 2, pool: 2}     # 19 → 9,  FPGA: k6 s2
```

- **Stride 1 only.** riverops' `fuse_conv_avgpool` builds the folded kernel ignoring the conv stride, so a strided conv + pool would fold into a kernel that disagrees with the exported geometry. The config check refuses it.
- **No fusion-aware QAT.** riverops' fusion-aware QAT and `export_params` don't support the pool layers. A network with pooling is therefore folded into plain `SpikingConv2d` layers (`deployable()` in `train.py`) before quantization: formats, per-layer scaling, the fixed-point check and export all run on the folded network. With `qat_epochs: 0` (our default) nothing is lost. The folded network gives exactly the same spikes as the trained one (checked).
- **Memory.** `layer_shapes`, the memory estimate and the weights count use the folded kernel. For example, the first layer's k5 + pool 4 costs 2,048 weights as a k8 conv, against 1,568 for the old k7.

## Results

Comparison on the 80×80 s4 network with the 20-epoch recipe of `configs/optim/adam.yaml`, two seeds each (`configs/arch/`): see [training-log.md](training-log.md), runs #27–#32.

## Before deploying any model

- **Build the firmware for that model's formats.** The firmware job takes the weight and potential formats as options (`scripts/firmware_job.py submit --run <run>` reads them from the run's export), so a firmware is only right for models exported in the same formats. Our models use Q1.7 weights; the 8-bit firmware in PR #4 was built for that pipeline's format (integer / 64, i.e. Q2.6), so a Q1.7 `weights.txt` on it would act twice too strong. Different layer sizes (v3, spatial head) need their own build anyway.
- **Input scaling.** spleat-server needs `inputs.scaling = 4096 / downsampling²` (64 at 40×40) because despot sends block sums (measured on the board, [deploy-model.md](deploy-model.md)). Our export still writes `scaling: 4096` in `spleat-server-config.yaml`: use the value from deploy-model.md.
