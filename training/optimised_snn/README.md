# Optimised SNN with quantization (8-bit, 40×40 firmware)

Training, 8-bit export and FPGA testing for the models that run on the Kria with the **8-bit nb40
firmware** (2×40×40 input, 5 outputs, downsampling 8). Built on `riverops` (not on PyPI). This is
a separate pipeline from `training/snn/`; the deployed weights are in
[`models/optimised_snn/`](../../models/optimised_snn/).

## What makes it fit the hardware

- **8-bit weights as the firmware reads them** (`--qat-bits 8h`): weights and biases are
  integer / 64, measured on the FPGA. Quantization-aware training uses exactly that format.
- **Continuous-stream training** (`--warmup-steps 40`): the FPGA never starts from zero
  potentials, so every clip is preceded by 40 context frames and only the clip is scored. Training
  from zero state lost 5-10 points on hardware.
- **Spike penalty** (`--spike-reg 3`): same accuracy with 30× fewer synaptic operations
  (59k instead of 1.75M per 10 ms step); FPGA latency drops from 0.74 to 0.15 ms per frame.
- **Quiet-scene samples** (`--quiet-prob 0.15 --quiet-events 12`): no-action samples made of
  near-empty frames, so the gesture outputs stay quiet when nothing moves.
- **Mirrored labels** (`--mirror 1:2`): the left/right flip augmentation swaps turn-left and
  turn-right. `--unused 3` keeps an output without data silent.

## Results on the FPGA

Own gestures (heart / turn left 360 / turn right 360 / no gesture), model `r6b-reg3`, people
never seen in training (sessions 29, 30, 360 clips), replayed into the board:

| accuracy | robot rule: correct / wrong / missed | false triggers on no-action | latency (median) |
|---|---|---|---|
| **99.2 %** | 98.3 % / 0 % / 1.7 % | 1.4 % | 0.15 ms per frame |

Organisers' classes (sit / lay down / hello / dance), model `r4-nb40-qatfull`, unseen sessions
04-06: 75.9 % on the FPGA (validation session 13: 79.7 %), 0.74 ms per frame.

## Input scaling on the board (important)

despot sends **block sums**; training uses **block means**. Set
`inputs.scaling = 4096 / downsampling²` in `/root/.config/spleat-server/config.yaml`: **64** for
these 40×40 models ([`kria/board_config/config.yaml`](../../kria/board_config/config.yaml)).
With the shipped 4096 the network gets 64× its training input (FPGA replay: 98.8 % → 37.5 %).
`fpga_eval.py` sends block sums like despot, so it expects the same setting.

## Files

| file | |
|---|---|
| `train_snn_v8.py` | training, QAT and export (`weights_q8h.txt` = board format) |
| `train_v8_alex.slurm` | job script for one A100 |
| `fpga_eval.py` | replay dataset clips into the SPLEAT server: accuracy, robot rule, latency (`--as-none`: false triggers on foreign clips) |
| `spleat_sim.py`, `sim_check.py` | integer model of the accelerator (matches the FPGA) |
| `eval_params.py`, `fit_fpga.py`, `collect_results.py` | score exported weights, fit fixed-point details, tabulate runs |
| `weights_convert.py`, `fpga_probe.py` | weights .txt ↔ riverops .json; send single frames |
| [`dataset/`](dataset/README.md) | cued recorder, clip cutter with quality filters, exporter, zip packer, `gestures.json` (how to record: [`dataset/README.md`](dataset/README.md)) |

## Train

```bash
python train_snn_v8.py --zip go2_gestures_s20-30.zip --arch nb40 --val 28 --test 29,30 \
  --class-names "Heart,Turn left,Turn right,Namaskaram,No action" --unused 3 --mirror 1:2 \
  --qat-bits 8h --epochs 5 --qat-epochs 30 --qat-lr-scale 1.0 --spike-reg 3 \
  --quiet-prob 0.15 --quiet-events 12 \
  --neg-zip motion_buddy_full.zip --neg-eval 04,05,06,13
```

Dataset: 11 people (sessions 20-30), 2,442 one-second clips, split by person. The zip
(649 MB) is shared separately; `dataset/export_sessions.py` rebuilds it from recordings. Sessions 31-33
add follow me (walk on the spot) for the people of sessions 20-22 (`go2_gestures_s20-33.zip`).

## Deploy and test

```bash
scp ../../models/optimised_snn/own_gestures/weights_r6b-reg3_056ee113.txt root@192.168.77.1:/root/weights.txt
scp ../../kria/board_config/config.yaml root@192.168.77.1:/root/.config/spleat-server/config.yaml
ssh root@192.168.77.1 'echo own > /root/model_classes && systemctl restart spleat-server'

ssh root@192.168.77.1 '/root/inference.sh stop'      # the camera client must not feed the server
python fpga_eval.py --size 40 --zip go2_gestures_s20-30.zip --sessions 29,30 \
  --class-names "Heart,Turn left,Turn right,Namaskaram,No action"
ssh root@192.168.77.1 '/root/inference.sh start --headless'
```

`/root/model_classes` (`own` or `dataset`) selects the bridge actions and viewer labels
(`kria/inference.sh`, `kria/live_gesture_bridge.py`).
