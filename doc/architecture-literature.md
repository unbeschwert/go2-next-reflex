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

## Why these architectures

### How they were picked

1. **Exportable or not tried.** A network only matters if riverops can lower it to `model.json`, because
   that's the only way onto the FPGA. So the filter came first: every literature design was reduced to its
   core building block and checked against what `export_to_river_model_graph` accepts (table above).
   Anything needing concatenation, grouped/depthwise or dilated convs, max-pool, linear layers, attention,
   or an add *inside* the neuron was dropped, however well it does on DVS Gesture.
2. **Aimed at our known weak spots.** From [training-log.md](training-log.md): the errors are on test
   person 14 (small, at the left edge, with a moving distractor) and person 13 (several people in the
   frame); *sit*, *hello* and *dance* get confused, and these gestures differ partly in timing. Avg-pool
   downsampling, which keeps fine spatial detail, gave the only gain larger than the seed noise so far
   ([architecture-options.md](architecture-options.md#results)). So the candidates target spatial detail and
   scale, gradient flow through more layers, and temporal memory.
3. **One change at a time, on the default recipe.** Every config inherits `motion_buddy_80_v4.yaml` and
   changes only `model.layers` (and which layers get dropout). Data, augmentation, schedule, seed and the
   8-bit export are the same as run #34, so a difference is the architecture's.
4. **About the same size.** 90–97 k weights against v4's 90 k (only `plif_vgg` is larger, 132 k, to stay
   faithful to the paper's FC layer), so a gain can't simply be more capacity, and every one fits the
   KV260 memory estimate with room to spare.
5. **Cheapest to deploy first.** Where possible the change sits on the deployed v4 network itself, so a
   winner is close to the firmware we already run.

### Why each one

| Config | Hypothesis it tests | Why this design |
|---|---|---|
| `sew_v4` | A shortcut that carries layer 0's 19×19 spikes past layer 1 helps small, off-centre people (fine detail reaches the downsampling layer unprocessed). | SEW "ADD" shortcuts are the best residual form in the SEW-ResNet DVS Gesture ablation (97.9 % vs 91.7 % plain) and add spikes, which riverops exports. Placed around layer 1 because it is v4's only stride-1 layer, the only place an identity shortcut fits without a projection. Same layers and weights as v4, one extra edge: the smallest possible step from the deployed model. |
| `sew_resnet` | More depth helps, provided shortcuts keep it trainable. | Our earlier deeper network (#19) learnt slowly (sit at 0 % for person 14 after 10 epochs), the degradation problem residual networks address. Two SEW basic blocks (two 3×3 convs each), one per resolution (19×19 and 9×9), on the v4 stem and head. |
| `plain_deep` | Control for `sew_resnet`. | Identical layers without the shortcuts, as the "plain net" of the SEW paper's ablation. The pair separates "deeper" from "shortcuts". |
| `branch_sum` | Two receptive field sizes at once cope better with people at different scales (small person 14, crowded person 13). | Inception-style and NAS-found SNN cells (AutoSNN, SNASNet) run 3×3 and 5×5 branches in parallel. They usually concatenate, which riverops can't export, so the branches add, the merge riverops does support. |
| `recurrent` | Memory longer than the leak (τ ≈ 2–4 steps, 20–40 ms) separates gestures that differ in timing. | Convolutional recurrent SNNs (SCRNN) were built for event-based hand gestures, and SNASNet found that backward connections add up to 3 %. riverops ships `RecurrentSpikingConv2d` and exports it as its own node, so it's the one temporal mechanism we can deploy (no delays, no temporal convs). Put at layer 1, where the map is still 19×19 and the stride is 1, as its feedback conv needs. |
| `plif_vgg` | Gradual downsampling (four ×2 steps instead of v4's ×4 stem) keeps more spatial detail early. | The DVS128 Gesture network of the paper that introduced the PLIF neuron, the neuron we use. Its max-pool becomes avg-pool, which riverops folds into the conv, extending our only proven gain. A chain, so it deploys with today's pipeline and firmware generator. |

### Considered and left out

- **SLAYER's gesture net**: pool, then conv, which is what v4's k5 + pool 4 stem already does.
- **DECOLLE**: a chain of conv layers; what's new is its learning rule (fixed random readouts per
  layer), not the architecture.
- **Two heads / an ensemble in one network**: exportable as several outputs, but riverops itself warns
  that several outputs "can lead to unpredictable behaviour".
- **U-Net / upsampling decoders**: `NearestUpsample` exports, but decoders serve dense outputs
  (segmentation), not classification.
- **Spiking ResNet, MS-ResNet, DenseNet, MobileNet, Spikformer, learnable delays**: not exportable (see the
  table above).

### Limits of the comparison

One seed per architecture, 20 epochs, and no tuning per architecture: the literature nets were tuned on DVS
Gesture, often with more epochs. A result within ±3–4 test points of the baseline is not a difference.
Anything that looks better needs a second seed before it replaces v4.

## Built only from riverops primitives

Every layer of every config is a riverops class, and every config goes through riverops' own exporter
(`export_for_river`) to a `model.json` (checked for all seven, v4 included):

| Config | riverops classes |
|---|---|
| `sew_v4`, `sew_resnet`, `plain_deep`, `branch_sum` | `SpikingConvBN2d`, `SpikingConvBNAvgPool2d` (their Dropout variants in float training only), a `PLIFNode` in each |
| `recurrent` | the same, plus `RecurrentSpikingConv2d` for layer 1 |
| `plif_vgg` | `SpikingConvBNAvgPool2d`, `SpikingConvBN2d` |

The only code that isn't a riverops class is glue: the shortcuts and branch merges are a PyTorch `a + b` on
spike tensors, which riverops' exporter supports explicitly (it imports the `aten.add` and folds it into a
node with several incoming edges, `collapse_add`), and `LayerGraph` in `training/snn/model.py` only runs the
layers in order and sums their inputs.

**They are riverops versions of the published networks, not copies.** To fit the library:

| Published | Here | Why |
|---|---|---|
| max-pool (PLIF net, SLAYER) | avg-pool folded into the conv | riverops has no max-pool; avg-pool is our one proven gain anyway |
| fully connected layers | a conv whose kernel covers the whole map | no linear layers |
| branches concatenated (Inception, NAS cells) | branches added | `torch.cat` isn't exportable |
| 128 channels per layer (PLIF net) | 16 / 32 | our weight budget (~90 k, the size of v4) |
| shortcut added to the current or membrane (Spiking ResNet, MS-ResNet) | not built | the neuron is fused inside the conv op |
| IF neurons (some papers) | PLIF | the conv blocks always build a `PLIFNode` (an exact IF replacement exists, see the first table) |

**Implementable is not deployable yet.** The chains (`plif_vgg`, `plain_deep`) deploy exactly like v4. The
graphs with multi-input or recurrent nodes pass riverops, but no firmware with such nodes has been built or
run on our board, and their `weights.txt` layout is an assumption ([Board status](#how-the-configs-use-it)).

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
augmentation, 8-bit per-layer export, no QAT), seed 42, only the network differs. Trained 2026-11-24 on a
Runpod **NVIDIA A40** (48 GB, 2 runs at a time, ~100–145 s per epoch; [remote/RUNPOD.md](../remote/RUNPOD.md)),
queue `runs/lit/`; the baseline is run #34 (same config and seed, the team's A100). Fixed point = the 8-bit
model as the FPGA runs it. Val = persons 11–12, test = 13–14. **With one seed, ±3–4 test points is noise**
([training-log.md](training-log.md) #23/#24).

| Config | Float val / test | **Fixed point val / test** | Test vs v4 | Train acc (last epoch) | Test recall: sit / lay-down / hello / dance / no-action | Most predicted |
|---|---|---|---|---|---|---|
| v4 (#34, baseline) | 73.9 / 72.6 | **74.9 / 72.4** | – | – | (hello P14 25 %, dance P14 50 %) | – |
| `sew_v4` | 67.8 / 70.5 | **67.4 / 71.4** | −1.0 | 60.8 | 68 / 90 / 33 / 71 / 91 | no-action 173 of 555 |
| `sew_resnet` | 73.3 / 68.3 | **73.5 / 67.9** | −4.5 | 65.2 | 66 / 94 / 22 / 54 / 99 | no-action 201 |
| `plain_deep` | 68.2 / 50.1 | **66.9 / 49.5** | −22.9 | 64.3 | 78 / 88 / 24 / 27 / 34 | **sit 295** |
| `recurrent` | 71.0 / 53.7 | **72.7 / 53.0** | −19.4 | 55.4 | 48 / 98 / 40 / 50 / 30 | sit 183, lay-down 179 |
| **`branch_sum`** | 79.1 / 73.5 | **76.6 / 75.9** | **+3.5** | 65.0 | 42 / 92 / **58** / 82 / 99 | dance 162 |
| `plif_vgg` | 70.8 / 50.5 | **67.4 / 49.0** | −23.4 | 65.0 | 28 / 84 / 25 / 4 / 98 | **no-action 364** |

The 8-bit export costs at most 3.4 points on val and 1.5 on test in every run (weights clipped 0 %), and the
formats came out as before (Q1.7, `sew_v4` Q2.6; per-layer k up to 2, `plif_vgg` layer 0 k = 8), so the
differences are the networks', not the export's.

### What we expected and what we see

| Config | Expected | Seen |
|---|---|---|
| `sew_v4` | a small gain, mostly for small, off-centre person 14 (layer 0's fine detail skips layer 1) | **neutral**: −1.0 test, within noise; val −7.5. Hello stays the weak class (33 %) |
| `sew_resnet` vs `plain_deep` | the shortcuts make the deeper net trainable (SEW-ResNet ablation, our slow #19) | **confirmed, the clearest result**: +18.4 test points for the same 8 layers (67.9 vs 49.5), far beyond the seed noise |
| `sew_resnet` vs v4 | depth helps, once trainable | **not seen**: −4.5 test (borderline). The deep nets look undertrained at 20 epochs (train 64–65 %) |
| `recurrent` | longer memory separates gestures that differ in timing | **collapses on test only**: val 72.7 % is normal, test 53.0 %, and float is as bad (53.7 %), so it isn't the 8-bit export |
| `branch_sum` | two receptive field sizes help with people at different scales | **the only one above v4**: +3.5 test, +1.7 val. Hello, v4's weakest class, goes to 58 % recall (v4: 25–33 %), sit drops to 42 % |
| `plif_vgg` | gradual downsampling keeps detail, deploys like v4 | **collapses on test only**, the other way round: no-action for 364 of 555 clips, dance recall 4 % |

**Reading.** Three networks have normal validation accuracy (67–73 %) but collapse on the test people
(49–53 %), each onto one class: `recurrent` (sit and lay-down predicted 362 of 555 times, no-action recall
30 %), `plain_deep` (sit 295 times) and `plif_vgg` (no-action 364 times, i.e. mostly silent). The test people
are the distribution shift: person 13 has several people in the frame, person 14 is small at the left edge
with a large moving distractor, while the train and val people are single, clean subjects. So the common
factor is brittleness under that shift, not one direction: the feedback loop and the deep plain chain may
amplify background motion into a gesture class, while `plif_vgg` (3×3 kernels, four ×2 downsamplings, the
largest layer-0 scale k = 8) loses the gesture and stays quiet. Not yet checked per person. Validation
people look like the training people and don't show any of it, which makes val a poor guide: `plain_deep`
had one of the best single val epochs (75.6 %) and nearly the worst test.

**Overall:** one candidate, `branch_sum`, is above v4 on both val and test, but one seed and +3.5 test points
is at the edge of the seed noise (±3–4), so it is a candidate, not a result. The clearest effect is the
shortcut ablation (+18.4 points for `sew_resnet` over `plain_deep`). Four of six are at or below v4, which
isn't surprising: our bottleneck is generalising from 10 clean training people to 2 different-looking test
people, which architecture alone doesn't address. `branch_sum` is also not deployable yet: its merge is a
multi-input node, untested on the firmware (see "Implementable is not deployable yet" above).

**Confounds to keep in mind.** The deep nets (`sew_resnet`, `plain_deep`) have dropout on 6 layers instead
of v4's 3 (fair between the two of them, not against v4); `recurrent`'s layer 1 has no BatchNorm and no
dropout (riverops' recurrent layer has neither), and its output is a sum of two spike populations (0–2).

### Next

- **Two more seeds of `branch_sum`** (43, 44), and v4 with the same seeds on the same GPU: the only
  candidate above the baseline, and the one result that could change the default recipe.
- If it holds: a test firmware build for its multi-input node and an FPGA-vs-simulation check, which also
  settles the weights.txt layout question for the SEW configs.
- A per-person breakdown (`scripts/eval_per_person.py`) for `recurrent`, `plain_deep` and `plif_vgg`, to
  test the brittleness reading.
- A second seed for `sew_v4` and `sew_resnet`.
- `sew_resnet` with v4's dropout layers (removes the confound), and with more epochs.
- Before any multi-input or recurrent model goes on the board: a test firmware build and an
  FPGA-vs-simulation check.
