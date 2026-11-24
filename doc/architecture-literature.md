# SNN architectures from the literature, within riverops

Which published spiking-network architectures for event-based gesture recognition can be built from
riverops 0.2.0's neurons and layer classes, so that `export_for_river` lowers them to a `model.json` for
the SPLEAT firmware generator, and how they did on our data. What riverops can export, and why, is in
[architecture-options.md](architecture-options.md#what-reaches-the-fpga); this page maps the literature
onto it.

## What riverops gives us

| Class | What it is | In model.json |
|---|---|---|
| `PLIFNode` | leaky integrate-and-fire, one learnable leak per layer (Fang et al. 2021) | per-layer threshold and leak |
| `IFNode` | integrate-and-fire, no leak | not usable inside the conv blocks (they always build a `PLIFNode`); exact replacement: `learnable_leak: false` with τ = 1e12 exports leak = 0 (any τ > 2¹³ rounds to 0 in Q4.12) and spikes like `IFNode`; the export doesn't record the neuron type ([architecture-options.md](architecture-options.md#layers-without-export_params)) |
| `SpikingConv2d`, `SpikingConvBN2d`, `SpikingConvBNDropout2d` | conv → (BN) → (dropout) → PLIF | `ConvLIF2d` |
| `SpikingConv(BN)(Dropout)AvgPool2d` | conv → (BN) → avg-pool → PLIF | `ConvLIF2d` (pool folded into the conv) |
| `RecurrentSpikingConv2d` | input conv + feedback conv from the layer's own previous spikes, two PLIF populations, spikes summed | `RecurrentConvLIF2d` |
| `NearestUpsample` | nearest-neighbour upsampling, only directly before a conv | `ConvLIF2d` with `upscaling` |
| `x + y` on spikes | element-wise sum feeding a conv | several incoming edges into one `ConvLIF2d` |

Not exportable: `torch.cat`/chunk/slicing, groups (depthwise), dilation, non-square kernels or strides,
max-pool, linear layers, global pooling, soft reset (not recorded; the firmware resets hard), adding
*currents* or *membrane potentials* (the neuron is fused inside the conv op), and any attention/matmul.

## Literature, mapped

| Architecture (paper) | Core building block | Fits riverops? | Our config |
|---|---|---|---|
| PLIF net for DVS128 Gesture (Fang et al., ICCV 2021, [arXiv:2007.05785](https://arxiv.org/abs/2007.05785)) | {conv3×3-BN-PLIF-maxpool2}×5 → FC → FC → voting | ✓ with avg-pool for max-pool, convs for the FC layers | `configs/lit/plif_vgg.yaml` |
| SLAYER DVS Gesture net (Shrestha & Orchard, NeurIPS 2018, [arXiv:1810.08646](https://arxiv.org/abs/1810.08646)) | pool4 → c5 → pool2 → c3 → pool2 → FC512 → FC | ✓, but it is close to our v4 chain (pool-then-conv = our k5+pool4 stem) | (covered by v4) |
| SEW-ResNet, ADD shortcut (Fang et al., NeurIPS 2021, [arXiv:2102.04159](https://arxiv.org/abs/2102.04159)) | block output = SN(conv(…)) **+ input spikes** | ✓ the shortcut adds spikes, and riverops folds `conv(a + b)` into a multi-input node | `configs/lit/sew_v4.yaml`, `configs/lit/sew_resnet.yaml` (+ control `plain_deep.yaml`) |
| Spiking ResNet (Zheng et al., AAAI 2021, tdBN) and MS-ResNet (Hu et al.) | shortcut added to the current / membrane before the neuron | ✗ the add would have to sit inside the fused conv-neuron op | – |
| Inception-style / NAS-found cells: AutoSNN (Na et al., ICML 2022), SNASNet forward cells (Kim et al., ECCV 2022, [arXiv:2201.10355](https://arxiv.org/abs/2201.10355)) | parallel branches of different kernels, merged | ✓ only if merged by addition (no concat) | `configs/lit/branch_sum.yaml` |
| Convolutional recurrent SNN, SCRNN (Xing et al., Front. Neurosci. 2020, [doi:10.3389/fnins.2020.590164](https://www.frontiersin.org/articles/10.3389/fnins.2020.590164)); SNASNet backward connections | feedback of a layer's spikes into itself on the next step | ✓ `RecurrentSpikingConv2d` | `configs/lit/recurrent.yaml` |
| DECOLLE (Kaiser et al., Front. Neurosci. 2020, [arXiv:1811.10766](https://arxiv.org/abs/1811.10766)) | conv layers with fixed random local readouts per layer | the network is a chain ✓; the readouts are a training method, not an architecture (several outputs are exportable but "can lead to unpredictable behaviour") | – |
| IBM TrueNorth DVS Gesture net (Amir et al., CVPR 2017) | grouped convs | ✗ groups | – |
| Spiking DenseNet | concatenation | ✗ concat (summing instead would apply one kernel to all sources, a different model) | – |
| Spiking MobileNet / depthwise-separable | depthwise convs | ✗ groups | – |
| Spikformer, Spike-driven Transformer (Zhou et al. 2023, Yao et al. 2023) | spiking self-attention | ✗ matmul, linear layers | – |
| Learnable synaptic delays (Hammouamri et al., ICLR 2024), temporal / dilated convs | dilated temporal kernels | ✗ dilation, no temporal conv | – |

## How the configs use it

The pipeline (`training/snn/`) takes two optional keys per layer:

- `inputs: [i, j, "input"]`: this layer reads the element-wise **sum** of those layers' spikes (indices of
  earlier layers, or `"input"` for the event frames). Without it, a layer reads the previous one, as
  before. All sources must have the same channels × H × W; the sum of two spike maps can be 2.
- `recurrent: true`: the layer is a `RecurrentSpikingConv2d` (kernel = 2·padding + 1, no pooling, no
  BatchNorm, no dropout). It holds two conv + neuron units, so it exports two weight sets and two
  threshold/leak pairs, and per-layer scaling applies the same k to both (same spikes).

A chain (no `inputs`, no `recurrent`) still builds an `nn.Sequential`, so existing configs and their exports
are unchanged (the v4 `model.json` is byte-identical to run #34's). Every config in `configs/lit/` builds,
lowers with `export_for_river`, quantizes (per-layer scaling, fixed-point emulation) and exports
`weights.txt`, and fits the KV260 memory estimate:

| Config | Layers | Weights | Neurons (potentials) | model.json |
|---|---|---|---|---|
| v4 (baseline) | 5 | 89,856 | 13,365 | 5 `ConvLIF2d` |
| `sew_v4` | 5 | 89,856 | 13,365 | 5 `ConvLIF2d`, layer 2 has 2 inputs |
| `sew_resnet` | 8 | 96,768 | 21,733 | 8 `ConvLIF2d`, layers 3 and 6 have 2 inputs |
| `plain_deep` | 8 | 96,768 | 21,733 | 8 `ConvLIF2d` (chain) |
| `branch_sum` | 6 | 96,256 | 19,141 | 6 `ConvLIF2d`, layer 3 has 2 inputs |
| `recurrent` | 5 | 92,160 | 19,141 | 1 `RecurrentConvLIF2d` + 4 `ConvLIF2d` |
| `plif_vgg` | 6 | 132,224 | 36,133 | 6 `ConvLIF2d` (chain) |

**Board status.** `plif_vgg` and `plain_deep` are chains: they deploy exactly like v4. For the others,
model.json lowers, but no firmware with multi-input or recurrent nodes has been built or run on our board,
and two things about their `weights.txt` are our assumptions:
- **Multi-input nodes** (`sew_v4`, `sew_resnet`, `branch_sum`): one weight set per node, applied to the
  sum of the incoming spikes (that is what riverops trains and lowers, `conv(a + b)`). Nothing confirms the
  RTL reads one weight set per node rather than one per incoming edge.
- **Recurrent layer** (`recurrent`): layer order, with the recurrent layer as its input unit then its
  feedback unit. riverops has no `export_params` for `RecurrentSpikingConv2d`. The export prints a
warning for them. A test firmware build (`scripts/firmware_job.py`) and an FPGA-vs-simulation check
([fpga-debugging.md](fpga-debugging.md)) come before any of them goes on the robot.

## Results

Default recipe (`motion_buddy_80_v4`: 80×80, 20 epochs, cosine lr, average of the last 5 epochs, full
augmentation, 8-bit per-layer export, no QAT), seed 42, only the network differs. These runs were trained
**on CPU** (the GPU server was unreachable); the baseline is run #34 from the GPU. Fixed point = the 8-bit
model as the FPGA runs it. Val = persons 11–12, test = 13–14. **With one seed, ±3–4 test points is noise**
([training-log.md](training-log.md) #23/#24).

RESULTS_TABLE
