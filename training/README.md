# SNN training pipeline

Train a spiking neural network on event-camera clips on the team GPU server. Quantize it to the
formats the Kria KV260's SPLEAT firmware uses, check it fits the FPGA's on-chip memory, and get
every file the board needs. One YAML config drives all of it.

```
laptop                                     GPU server (A100)
──────                                     ─────────────────
remote/setup.sh      ── code + wheel ──►   uv env: torch (CUDA) + riverops
remote/push_data.sh  ── dataset ───────►   data/motion_buddy_dataset/
remote/train.sh cfg  ── check, sync ───►   tmux: prepare → float → QAT → fixed-point → export
remote/status.sh     ◄── log, GPU ──────   runs/<name>/train.log
remote/fetch.sh      ◄── results ───────   runs/<name>/{metrics.json, *.pt, deploy/}
                                                                   │
runs/<name>/deploy/DEPLOY.md ─► riversdk.dev firmware job ─► Kria KV260
```

## Folder layout

```
training/
  configs/
    motion_buddy_40.yaml   main config: 5 gestures, 40x40 input, notebook network adapted for the Kria
    motion_buddy_40_aug.yaml  same + data augmentation, 300 epochs (inherits via `base:`)
    motion_buddy_40_e300.yaml same, 300 epochs, no augmentation
    ablation/              100-epoch runs, one augmentation group each (doc/data-augmentation.md)
    smoke.yaml, smoke_aug.yaml  tiny 1-epoch runs to check the pipeline end to end
  snn/
    config.py    YAML → typed config + validation (despot-compatible sizes, splits by person, formats)
    events.py    .bin reader, events → per-frame counts
    data.py      find clips, split by person, cache as (N, T, 2, H, W) uint16
    model.py     layer shapes (no torch) + riverops model
    fpga.py      KV260 BRAM/URAM memory estimate (no torch)
    quant.py     choosing fixed-point formats, clipping statistics
    augment.py   data augmentation on GPU batches (doc/data-augmentation.md)
    train.py     float training, QAT, fixed-point evaluation
    export.py    model.json, weights.txt, Kria configs, memory report, DEPLOY.md
    __main__.py  command line: check | prepare | train | qat | export
remote/
  remote.env.example   server address template (copy to remote.env, which git ignores)
  setup.sh             install uv + training environment on the server, check GPU
  push_data.sh         upload dataset / extra recording folders (resumable)
  train.sh             start a run in tmux on the server
  qat.sh               redo formats + QAT + export of a run from its float_best.pt
  status.sh            sessions, GPU, log tail (-f to follow)
  fetch.sh             download a run to runs/<name>/
  stop.sh              stop a run
  _run.sh, _common.sh  internal
runs/    (git-ignored) one folder per run
cache/   (git-ignored) preprocessed event counts, reused across runs
```

## Quick start

```bash
# once
cp remote/remote.env.example remote/remote.env      # fill in user@host
# put riverops-0.2.0-py3-none-any.whl in data/ (riversdk.dev download list; proprietary: never commit it)
remote/setup.sh
remote/push_data.sh

# every run
uv run python -m training.snn check --config training/configs/motion_buddy_40.yaml   # laptop, instant
remote/train.sh training/configs/motion_buddy_40.yaml             # → run name printed
remote/status.sh <run> -f                                          # Ctrl-C stops following, not training
remote/fetch.sh <run>
open runs/<run>/deploy/DEPLOY.md
```

You need SSH key access to the server (`ssh user@host` must work without a password prompt).

## What a run does

1. **Check.** Validate the config, propagate shapes, and estimate FPGA memory. If the model can't fit, it stops before using the GPU (`fpga.fail_on_overflow`).
2. **Prepare.** Read every clip once (28 CPU workers on the server) and cache the per-frame event counts in `cache/`. Later runs with the same data settings reuse the cache.
3. **Float training.** Same recipe as the organisers' notebook:
   - spike-count targets of 80/20 % of T and an MSE loss
   - Adam, gradient clipping
   - weight clamp, ±2 px random shift
   - optional data augmentation (`augment:` block, off by default): [doc/data-augmentation.md](../doc/data-augmentation.md)
   - The best epoch is picked by validation accuracy, or the last epochs are averaged (see [Training recipe options](#training-recipe-options)).
4. **Choose formats** (see below).
5. **QAT** (skipped with `quant.qat_epochs: 0`). BatchNorm scales that weight decay shrank to ~0 are set to exactly 0, before and during QAT (otherwise QAT turns them into NaN; see [NaN during QAT](#nan-during-qat)). Then quantization-aware fine-tuning with BatchNorm **folded inside** the fake quantization (`prepare_qat(fused_bn=True)`). The network trains on the same rounded and clipped weights the FPGA will run. Each QAT epoch logs validation **and test** accuracy (`test acc`, column `test_acc` in `history.csv`). The kept epoch is still the best on validation, so test is only watched, never used to choose. To add QAT to a finished run: `remote/qat.sh <run> --set quant.qat_epochs=10`.
6. **Fixed-point check.** Run the folded, quantized model with integer-grid arithmetic (`prepare_ptq`) on validation and test. This is the accuracy to trust. Float, QAT and fixed-point accuracies are all in `metrics.json`.
7. **Export** to `runs/<run>/deploy/` (below).

The whole dataset is kept on the GPU, so an epoch over ~2,700 clips takes seconds, not minutes.

## Training recipe options

All off by default, so older configs train as before. `configs/motion_buddy_40_v3.yaml` turns them on: our pipeline combined with the training choices of a teammate's v3 notebook.

**Default recipe: `configs/motion_buddy_80_v4.yaml`** (since 2026-11-22). It is the v3 recipe at 80×80 with avg-pool downsampling, 20 epochs (cosine over 15, average of the last 5), and decay on the conv weights only. Network and training are those of the best tested setting: 68.6 / 69.2 % test in 8 bit for two seeds ([doc/training-log.md](../doc/training-log.md) #29/#30). On top it adds speed, zoom/rotation, noise and hot-pixel augmentation (noise scaled to 80×80); cutout stays off. On the board: despot downsampling 4, 80×80 inputs, `scaling` 256 (the export writes it).

| Option | What it does | Why it helps |
|---|---|---|
| `model.neuron.learnable_leak: true`, `initial_tau: 2.0` | Each layer learns its own leak, starting at τ = 2 (the potential keeps 50 % per step). Exported as that layer's leak in `weights.txt` | Neurons react to recent *motion* (~20 ms) instead of slowly built-up background events. Leftover potential from the previous gesture fades within ~10 steps, which matters because the FPGA never resets |
| `model.dropout: 0.2`, `dropout_layers: [1, 2, 3]` | Dropout2d on those layers' currents (riverops `SpikingConvBNDropout2d`), **float training only**. The model is rebuilt without it for quantization and export (same weights; riverops can't export the dropout layer type) | Spreads information across neurons; costs nothing on the FPGA |
| `train.lr_min`, `cosine_epochs` | Cosine learning rate from `lr` to `lr_min` over `cosine_epochs` (default `epochs − average_last`), then constant | At a constant 3e-3 the weights keep bouncing: validation swung 59 ↔ 80 % between epochs |
| `train.average_last: N`, `recompute_bn: true` | The final float model is the average of the last N epochs' weights instead of the best single epoch (stochastic weight averaging, SWA). The BatchNorm running statistics are then recomputed for the averaged weights with one pass over the training data, as `torch.optim.swa_utils.update_bn`; the log shows val with the averaged statistics and with the recomputed ones. `recompute_bn: false` averages the stored statistics instead (runs before 2026-11-22 05:30) | Lands in a flat minimum and avoids picking the luckiest of many noisy epochs on 2 validation people |
| `train.optimizer: adamw`, `weight_decay`, `decay_on` | AdamW: decoupled weight decay. Default `adam` = Adam with L2 decay, as the notebook. `decay_on: all` (default) decays every parameter, `conv_weights` only the conv weights (not BatchNorm γ/β or the learnable leaks); it applies to either optimizer, so the optimizer can be compared on its own (`configs/optim/`) | Adam's L2 decay hits parameters without a loss gradient hardest: the BatchNorm γ of dead channels went to ~0, which caused the [QAT NaN](#nan-during-qat). AdamW shrinks every weight by the same small factor per step. Tested once (run #21 in [doc/training-log.md](../doc/training-log.md)): worse on test, and 1e-2 decayed the conv weights *less* than Adam's 1e-4 (up to ~15× less in layer 3), so the strengths don't translate directly |
| `train.loss: ce+mse`, `ce_temperature: 10`, `ce_mse_weight: 1` | Cross-entropy on the spike counts / τ as logits, plus the notebook's MSE to the 80/20 % targets (`mse`, the default; `ce` alone is also possible). Mixup blends the labels like the inputs | Cross-entropy optimizes the ranking that the argmax uses; the MSE keeps the counts in the 20–80 % range (not saturated or silent, as the 8-bit thresholds and potentials are sized for) |
| `augment.mixup` | Blends two clips and their spike-count targets ([doc/data-augmentation.md](../doc/data-augmentation.md)) | Graded rather than overconfident outputs |
| v3 layer sizes (in the config) | 16 → 16 → **16** k6 s2 → **128 k6 s2** (2×2 output) → classes k2: 88.6 k weights instead of 175 k | The 2×2 output shares weights across positions, so it's partly position-independent. Our 256 × k9 layer on a 9×9 map is effectively fully connected, and fails on off-centre people. Changes `model.json`, so it needs its own firmware |
| Spatial head: last layer `{out: classes, kernel: 1}` on the 2×2 map | The output is a classes × H × W map; a class's score is its spike count summed over time **and** positions (always done; for a 1×1 output it's the same as before) | The same classifier weights at every position, so the head is translation-invariant (global sum pooling in software). On the board the reader must add each class's H·W outputs ([doc/architecture-options.md](../doc/architecture-options.md#spatial-head)) |
| Layer `pool: P` (with `stride: 1`) | Stride-1 conv → BatchNorm → P×P average pool → neuron (riverops `SpikingConvBNAvgPool2d`). Folded for the FPGA into one conv of kernel `kernel + P − 1`, stride `P`. A network with pooling is folded before quantization (no fusion-aware QAT for it) | Anti-aliased downsampling: smoother and more robust to small shifts than a strided conv, at the same FPGA cost. Configs: `configs/arch/` ([doc/architecture-options.md](../doc/architecture-options.md#avg-pool-downsampling)) |

Not known from the v3 notebook (it isn't on GitHub), so these are our assumptions: dropout after layers 1–3, the augmentation probabilities, mixup α = 0.4, cosine over the first 30 of 40 epochs.

## Quantization and the FPGA memory

### KV260 on-chip RAM

The KV260's K26 SOM (XCK26) has two kinds:

| | Blocks | Each block | Shapes available |
|---|---|---|---|
| Block RAM (BRAM) | 144 | 36 Kb (or two 18 Kb halves) | 32K×1, 16K×2, 8K×4, **4K×9**, **2K×18**, 1K×36, 512×72 |
| UltraRAM (URAM) | 64 | 288 Kb | 4K×72 only |

A memory always takes whole blocks, and every value goes into the next native width up. So the bit width decides the block count, not just the number of bits:

| Value width | Native slot | Values per BRAM36 |
|---|---|---|
| 8 or 9 bit | 9 bit | 4096 |
| 10–18 bit | 18 bit | 2048, i.e. twice the BRAM |

For the main config, `check` prints: **8-bit weights → 54 BRAM36, 9-bit → 54, 16-bit → 96.5.**

### SPLEAT and the firmware form

SPLEAT gives every layer its own processing unit, and on-chip memory is its limit (Courtois et al., arXiv:2406.17617). The firmware form on riversdk.dev asks for one global weight format and one potential format, and defaults to **Q4.12 / Q4.12 (16-bit weights)**.

The pipeline therefore uses:

- **8-bit weights** (`quant.weight_bits`). They fill 9-bit BRAM words, so they cost half of the 16-bit default. 9 bits would cost the same BRAM, if you need one more bit of precision.
- **"auto" weight int bits.** After float training, BatchNorm is folded into the convolutions. The pipeline then picks the fewest integer bits that hold the folded weights (99.9th percentile) and **all** folded biases. The total width stays fixed, so any int bit not needed becomes a fractional bit, i.e. more precision for the same memory. The notebook fixed weights at Q1.7 and only clamped *before* folding, so folded values past ±1 were silently clipped at export. `export_summary.json` reports how many values ended up clipped.
- **Per-layer formats through the threshold** (`quant.layer_scaling: true`). The weights must stay 8-bit (hardware constraint), and the firmware takes one weight format for all layers. But the layers need very different ranges: layer 0's BN-folded weights reach ±6–8 (its inputs are tiny 8×8-block averages, so BatchNorm scales it up), while layer 3, with 95 % of the weights, stays within ±0.5. One format chosen for all of them is decided by layer 0's 800 weights: Q3.5 for the augmented model. Then 46 % of layer 3's weights round to 0 (SQNR 17 dB). Run `describe_layers(choose_formats(model, cfg)[1]["layers"])` (`quant.py`) for this table; every run also prints it.

  A LIF layer spikes identically if its weights, bias **and threshold** are all divided by k: the potential just scales by 1/k, the leak is multiplicative, and the hard reset goes to 0. `weights.txt` holds one threshold per layer. Scaling by k = 2ⁿ is the same as giving that layer n more integer bits at the same width. So with `layer_scaling`, each layer's own format is computed (from its weights and, with `layer_scale_biases: true`, its biases). The shared format is the smallest of those, and every other layer gets k = 2^(own − shared) with threshold 1/k. Same 8-bit weights, same `model.json`; only the weight-format field of the firmware form changes.

  Measured on the augmented 300-epoch model (float 84.0 / 70.3 %, val / test), 8-bit fixed point:

  | Export | Val | Test |
  |---|---|---|
  | one format (Q3.5) + 10 epochs QAT | 82.1 | 66.3 |
  | per-layer (Q1.7, k = 8, 2, 1, 2, 2) + QAT | 82.8 | 66.3 |
  | **per-layer, no QAT** | **83.2** | **67.9** |
  | per-layer sized for weights only (biases clipped) + QAT | 55.6 | 36.8 |

  Per-layer formats without fine-tuning lose only ~1–2 points, and QAT on top doesn't help. **Biases must fit**: clipping them broke the model, and QAT hid it (train 93 %, val 50 %, because fused-BN QAT rebuilds the bias around batch statistics in training mode). An earlier test that scaled only layer 0 but kept the format chosen with layer 0 included showed no gain; that test was flawed. **Confirmed on the board** (2026-11-22, `motion-buddy-80-v4`, k = [2, 1, 1, 1, 2]): FPGA and fixed-point simulation give the same prediction on 19 of 20 clips, which they couldn't if SPLEAT ignored the per-layer thresholds ([doc/fpga-debugging.md](../doc/fpga-debugging.md#7-checks-on-the-8080-avg-pool-model-2026-11-22)).
- **16-bit Q4.12 potentials.** Each neuron's potential is stored between time steps, in 18-bit BRAM words. Q4.12 gives the spleat-server `scaling` 4096 for one input unit, divided by downsampling² because despot sends block sums and training uses means.

### `fpga.py` estimate

Per layer, it counts:
- weights: C_out·C_in·k² words
- biases: C_out words
- potentials: C_out·H_out·W_out words
- an input-event FIFO (`event_fifo_depth` entries)

Then it maps each memory to blocks: all in BRAM, and alternatively with the biggest memories moved to URAM. It keeps `fpga.reserve` (25 %) free for the GenX320 IP, DMA and the rest of the design.

**It is an estimate:** SPLEAT's real memory layout isn't public. The firmware job on riversdk.dev is the real test. If a job fails for resources, shrink the biggest memory, which `memory_report.md` shows. In the main config that's layer 3's 165,888 weights; try 128 channels instead of 256.

## What the export produces

Everything lands in `runs/<run>/deploy/`, and `DEPLOY.md` has the exact values filled in.

| File | Goes to |
|---|---|
| `model.json` | riversdk.dev → New firmware (board KV260, Prophesee IP GenX320, **the formats from DEPLOY.md**) |
| `weights.txt` | `/root/weights.txt`: all weights, all biases, threshold/leak per layer; no trailing newline |
| `spleat-server-config.yaml` | `/root/.config/spleat-server/config.yaml`: input H×W, scaling = 2^potential_frac / downsampling² for "mean" frames (despot sends block sums: 64 at 40×40, 256 at 80×80 with Q4.12; see [doc/deploy-model.md](../doc/deploy-model.md)), outputs = classes × the last layer's H × W |
| `metavision_camera.json` | `/root/metavision_camera.json`: frequency = 1e6 / time_window_us, downsampling |
| `memory_report.md/.json` | BRAM/URAM estimate for the chosen formats |
| `parameters.json`, `export_summary.json` | integers per layer, clipping statistics |

Re-export from a run's checkpoint without retraining: `uv run --group train python -m training.snn export --run-dir runs/<run>`. This needs the train group installed.

Redo everything after float training (formats, QAT, fixed-point check, export) from `float_best.pt`: `remote/qat.sh <run>` on the server, or `python -m training.snn qat --run-dir runs/<run>`.

## NaN during QAT

Symptom: the QAT epochs show `spikes 0.0` and val loss exactly `16.0000` (all-zero output), and the export says 100 % of the weights are clipped (they are NaN). Cause: channels whose input never changes. Weight decay alone shrinks their BatchNorm γ towards 0, and fusion-aware QAT divides by it (the gradient by its square) → inf → NaN everywhere. The pipeline now zeroes any |γ| < 1e-6 before QAT and after every QAT step, and stops with an error naming the parameter if a weight becomes non-finite. For runs made before this fix, `remote/qat.sh <run>` redoes QAT. Details: [doc/data-augmentation.md](../doc/data-augmentation.md#related-fix-nan-in-qat-after-long-training).

## Things to keep consistent

| Setting | Training (config) | Kria |
|---|---|---|
| Input size | `data.downsampling` (320 / ds) | despot `downsampling`, server `inputs.height/width` |
| Frame length | `data.time_window_us` | despot `frequency` |
| Pixel value | `data.encoding` | what despot sends: **unconfirmed** (below) |
| Weight format | chosen at training, in DEPLOY.md | firmware form "Weights quantization" |
| Potential format | `quant.potential_*` | firmware form "Potentials quantization", server `scaling` |
| Number of outputs | `data.classes` | server `outputs.channels`, ROS `Gestures.msg` (11 → N) |

**Input encoding is unconfirmed.** The notebook averages each block of pixels (`encoding: mean`). We don't know whether despot sends block averages, sums or 0/1 spikes. If it's different, the network sees inputs scaled wrongly and its outputs are meaningless even though everything runs. Ask the organisers. Then set `data.encoding` to match (`mean` / `sum` / `binary`) and retrain.

## Training on your own recordings

1. Record with `river-tools capture` (`doc/capture-data.md`).
2. Cut each recording to 1 s around the gesture.
3. Name the files `gesture_<NN>-<class>-<style>-<take>.bin` in a folder per person, under one dataset folder.
4. Upload them: `remote/push_data.sh data/my_dataset`.
5. Copy a config and set `data.root: data/my_dataset`, `data.classes` and the person splits.
6. Run `check`, then `train.sh`.

Only classes listed in `data.classes` are used, so you can also train on a subset of the Motion Buddy gestures.
