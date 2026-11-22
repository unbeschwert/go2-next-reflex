# Go2 Next Reflex: Wacky Wavers

**Hand gestures → event camera → spiking neural network on an FPGA → a Unitree Go2 that reacts.**

Team 3, **Wacky Wavers**, Durance AI track *"Give a robot dog its next reflex"*:
HackLab × BRISA robotics hackathon, CIC Berlin, 21–22 November 2026.

A Prophesee GenX320 event camera watches a person. A spiking neural network (SNN) on the AMD Kria
KV260's FPGA recognises the gesture, and a bridge on the same board turns it into a Unitree Go2 command
over ROS 2. Everything runs on the board, so the dog works without a laptop (SD-card autostart); a web
dashboard on the laptop watches, tunes, drives, records and trains.

```
GenX320 event camera ─► camera client ─► SNN on the Kria KV260 FPGA ─► gesture bridge ─► ROS 2 ─► Unitree Go2
   (10 ms frames)       (block sums)     (SPLEAT accelerator, 5 outputs)  (0.5 s window,     (sport API)
                                                                          rules, safety)
```

An overview of what we changed and why: [doc/Gesture-controlled_Go2_what_we_fixed_and_how_it_works_now.pdf](doc/Gesture-controlled_Go2_what_we_fixed_and_how_it_works_now.pdf).

## What the robot does

Our own gestures (`own` action set, the default):

| Gesture | Go2 reaction |
|---|---|
| Heart: arms make a heart above the head | Finger heart (sport API 1036) |
| Turn left: left arm out to the side and back, repeated | Full 360° turn to the left, closed loop on the IMU heading |
| Turn right: right arm out to the side and back, repeated | Full 360° turn to the right |
| Follow me: walk on the spot | Walks 1 m forward (closed loop on odometry); LiDAR person following with `--follow lidar` |
| Anything else | Nothing |

The Durance AI team's dataset gestures (`dataset` action set) map Sit, Lay down, Hello and Dance to the
matching Go2 tricks. `/root/model_classes` on the board selects the set. A trick starts only when one
output dominates the last 0.5 s and holds for a moment, so stray movement does not trigger it. In robot
mode the LiDAR obstacle check must report the space around the dog clear
([kria/live_gesture_bridge.py](kria/live_gesture_bridge.py)).

## Results

| What | Result | Source |
|---|---|---|
| Our gestures, people never seen in training, on the FPGA (8-bit 40×40 model `r6b-reg3`) | **99.2 %** of 360 clips; robot rule 98.3 % correct, 0 % wrong, 1.4 % false triggers | [training/optimised_snn/README.md](training/optimised_snn/README.md) |
| FPGA time per 10 ms frame, same model | **0.15 ms** (59k instead of 1.75M synaptic operations per step) | same |
| Durance AI team's gestures | 75.9 % on unseen people 04-06; 79.7 % on person 13, our validation person (the original 40×40 model: 60.5 % on person 13) | same, overview PDF |
| Input scaling fix on the board (`4096 / downsampling²`) | FPGA replay accuracy 37.5 % → 99.4 % | [doc/deploy-model.md](doc/deploy-model.md) |
| Firmware build strategies, 8-bit nb40 | 0.184 ms median per frame for all three, about 3.78 W during replay, bit-exact outputs | [firmware_performance_testing/README.md](firmware_performance_testing/README.md) |
| 80×80 pipeline, default recipe `motion_buddy_80_v4` | 72.4 % test (fixed point) | [doc/training-log.md](doc/training-log.md) |
| Robot reaction | about 0.6 s from the start of a gesture to the command; a 360° turn takes about 4.5 s | [firmware_performance_testing/README.md](firmware_performance_testing/README.md), overview PDF |

## Run it

**On the board** (Kria KV260, `ssh root@192.168.77.1`; scripts from [kria/](kria/) in `/root/`):

```bash
/root/inference.sh start --headless    # camera -> FPGA -> bridge, commands go to RViz on the laptop only
/root/inference.sh robot --dry-run     # on the Go2's network, decisions are only logged
/root/inference.sh robot               # the real Go2 reacts (LiDAR obstacle check on)
/root/inference.sh stop | status
```

**Dashboard** on the laptop ([dashboard/README.md](dashboard/README.md)): board status, live camera view
with the detected gesture, every gesture setting, simulator with RViz in the browser, manual driving with
dead-man and E-STOP, guided recording, dataset browser with each clip's network spikes, training on any GPU
machine, model install with backup, SD-card autostart.

```bash
cd dashboard && ./run.sh               # http://127.0.0.1:8800
```

**RViz on the laptop**: `ros2/rviz/fetch_go2_model.sh` once, then
`source ros2/env_board_link.sh && ros2 launch ros2/rviz/view_go2.launch.py`.

**Python environment**: `uv sync`; for training `uv sync --group train` (needs the riverops wheel in
`data/`, which is not in the repository).

## From recording to the robot

1. **Record**: guided recording with automatic labels, [training/optimised_snn/dataset/README.md](training/optimised_snn/dataset/README.md)
   (or the dashboard's Record page); plain `river-tools` capture in [doc/capture-data.md](doc/capture-data.md).
2. **Train**: the YAML-driven pipeline in [training/README.md](training/README.md) (`python -m training.snn`,
   GPU server via [remote/](remote/)), or the 8-bit 40×40 pipeline in [training/optimised_snn/README.md](training/optimised_snn/README.md).
3. **Firmware**: build on riversdk.dev with `scripts/firmware_job.py`, [doc/deploy-model.md](doc/deploy-model.md).
4. **Deploy**: bitstream, server config, camera config and weights to the board ([doc/deploy-model.md](doc/deploy-model.md)),
   or dashboard → Board → Models → Install.
5. **Run**: `inference.sh` as above; check on the FPGA with `training/optimised_snn/fpga_eval.py` or `scripts/fpga_debug/`.

## Repository structure

| Path | Contents |
|---|---|
| [doc/](doc/README.md) | Team notes: board setup, data, training, augmentation, architecture options, training log, deployment, FPGA debugging, the overview PDF |
| [training/](training/README.md) | SNN training pipeline driven by YAML configs (`python -m training.snn check/prepare/train/qat/export`), quantization and export for the Kria |
| [training/optimised_snn/](training/optimised_snn/README.md) | 8-bit quantization-aware training for the 40×40 firmware, FPGA replay tests, integer simulator of the accelerator, [dataset tools](training/optimised_snn/dataset/README.md) |
| [remote/](remote/) | Run training on a GPU server: set up, push data, train, follow, fetch results |
| [scripts/](scripts/) | Event and augmentation viewers, firmware jobs, per-person evaluation, FPGA client, FPGA debugging checks (`fpga_debug/`) |
| [firmware_performance_testing/](firmware_performance_testing/README.md) | Replay recorded gestures on the KV260 to compare firmware latency, board power and output spikes |
| [dashboard/](dashboard/README.md) | Web dashboard for the whole system |
| [kria/](kria/) | Runs on the board: `inference.sh`, the gesture bridge and its settings schema, LiDAR follow, `go2_cmd.py`, obstacle light, live SNN view (:8080), raw camera view (:8081), recorder, `board_config/` |
| [ros2/](ros2/) | Laptop side: ROS 2 environment (Jazzy or Humble), simulator, animated Go2 in RViz |
| [models/](models/) | Deployed weights with their metrics (`optimised_snn/own_gestures/`, `optimised_snn/dataset_gestures/`) and network graphs for the firmware generator |

## Team

**Wacky Wavers**: Team 3, Durance AI track.

| Name | LinkedIn |
|---|---|
| Bhanuteja Vadlamudi | [linkedin.com/in/bhanuteja-vadlamudi-25291a135](https://www.linkedin.com/in/bhanuteja-vadlamudi-25291a135/) |
| Cherian Grundmann | [linkedin.com/in/cherian-grundmann](https://www.linkedin.com/in/cherian-grundmann) |
| Grigorii Vydrevich | |
| Ritayan Dhara | [linkedin.com/in/ritayandhara](https://www.linkedin.com/in/ritayandhara) |
| Nirmal Kumar Ramadoss | [linkedin.com/in/nirmal-kumar-ramadoss-b9367a25a](https://www.linkedin.com/in/nirmal-kumar-ramadoss-b9367a25a/) |
| Litim Jihen | [linkedin.com/in/litim-jihen-a10ba7239](https://www.linkedin.com/in/litim-jihen-a10ba7239) |

Thanks to the Durance AI team for the challenge, the hardware, the RIVER SDK and the Motion Buddy dataset.

## License

Our code and documentation are released under the [MIT License](LICENSE).

Not covered by it:
- vendored in `dashboard/static/vendor/`, under their own licenses (included there): three.js (MIT),
  noVNC (MPL-2.0) with pako (MIT);
- not part of this repository: the riverops library and its wheel, the Durance AI team's firmware,
  weights and Motion Buddy dataset, the go2-kria workspace with the Unitree messages, and the Unitree Go2
  model (BSD-3-Clause, downloaded by `ros2/rviz/fetch_go2_model.sh`);
- parts of `doc/` are condensed from the Durance AI team's RIVER SDK documentation.
