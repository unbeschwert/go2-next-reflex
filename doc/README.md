# Documentation

Team notes for the Durance AI track: **"Give a robot dog its next reflex"**. Record event-camera data, train a spiking neural network (SNN), run it on the Kria KV260 FPGA, and make the Unitree Go2 react through ROS 2.

Most of this is condensed from the organisers' RIVER SDK docs at https://riversdk.dev/docs/ (login required), current as of 2026-11-21.

## Contents

| File | What's in it |
|---|---|
| [kria-setup.md](kria-setup.md) | Board setup: cables, SSH/UART, network to the Go2, starting the SPLEAT server and camera app |
| [capture-data.md](capture-data.md) | Recording your own event clips with `river-tools` |
| [motion-buddy-dataset.md](motion-buddy-dataset.md) | The supplied gesture dataset: file format, classes, statistics, viewing clips |
| [training-snn.md](training-snn.md) | Training an SNN with `riverops`: the reference notebook, the library, quantization and export |
| [../training/README.md](../training/README.md) | **Our training pipeline**: GPU server, configs, KV260 memory-aware quantization, export to the Kria |
| [data-augmentation.md](data-augmentation.md) | **Data augmentation** for robustness: what each one does and why, config, ablation runs, the QAT NaN fix |
| [deploy-model.md](deploy-model.md) | Putting a trained SNN on the Kria: bitstream, server config, camera config, weights file |
| [fpga-debugging.md](fpga-debugging.md) | **Learning record**: why the FPGA and the simulation disagreed (potentials never reset), each hypothesis and its test |
| [go2-kria-overview.md](go2-kria-overview.md) | ROS 2 entry points, topics, reference action map and Yocto build for the Kria |
| [gesture-pipeline.md](gesture-pipeline.md) | FPGA inference, Python bridge for the Go2, and [laptop simulation in Docker with RViz](gesture-pipeline.md#docker-and-rviz-on-wayland) |

## End-to-end workflow

```
 1. Capture        river-tools capture  ──► samples/*.bin            (capture-data.md)
 2. Inspect        scripts/view_events.py                            (motion-buddy-dataset.md)
 3. Train          remote/train.sh on the GPU server ──► runs/<name>/  (training/README.md)
 4. Export         export_for_river + export_params
                   ──► model.json (layers) + weights.txt (integers)
 5. Firmware       riversdk.dev "firmware jobs" (model.json)
                   ──► spleat.bit.bin + spleat.dtbo                  (deploy-model.md)
 6. Deploy         scp bitstream, config.yaml, metavision_camera.json, weights.txt to the Kria
 7. ROS 2          install go2-kria's Humble overlay, copy kria/ scripts, check output order
 8. Run            inference.sh start --headless, then inference.sh robot --dry-run
```

At run time:

```
GenX320 ─► despot (events → frames, 100 Hz, downsampled) ─► SPLEAT server (SNN on the FPGA, TCP :3333)
        ─► live_gesture_bridge.py (+ obstacle_detection) ─► /api/sport/request ─► Go2
```

The supplied 11-class C++ launch and the laptop simulator share the reference
pipeline. [Go2 entry points](go2-kria-overview.md#entry-points)

## Downloads (from the RIVER SDK download list)

All of these go in `data/`, which git ignores.

| File | What it is |
|---|---|
| `riverops-0.2.0-py3-none-any.whl` | The SNN training library |
| `river_tools-0.1.0-py3-none-any.whl` | Recording, playback and deployment tool |
| `durance-ai-handgesture-recognition.zip` | Motion Buddy dataset (4.09 GB, extracts to `motion_buddy_dataset/`) |
| `yocto-hackathon-img.gz` | SD-card image for the Kria, to re-flash if the board breaks |
| `x86_64_sdk_installer.sh` / `arm_sdk_installer.sh` | Yocto cross-compile SDK for your laptop type |
| `go2-kria-generated.zip` | The ROS 2 workspace (`go2-kria`) |
| `ros-docker-original.zip` | Original generic Humble and Yocto Docker environment with an example package |

Re-flashing the SD card (**check the device name with `lsblk` first: `dd` overwrites whatever you point it at**):

```bash
gzip -dc yocto-hackathon-img.gz | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
```

## Things that don't line up (to check with the organisers)

| | Training notebook | Kria deployment docs | `go2-kria` ROS code |
|---|---|---|---|
| Input size | 32×32 (320 averaged down) | `config.yaml` example: **80×80**; despot can only shrink by powers of 2 (320, 160, 80, 40, 20, …), so **32×32 can't be produced** | – |
| Time step | 10 ms (100 steps per second) | despot `frequency: 100.0` → 10 ms ✓ | – |
| Number of outputs | 5 classes | `config.yaml` example: 5 ✓ | expects **11** scores (DVS-Gesture) ✗ |
| Potential format | Q4.12 | `scaling: 4096` = 2¹² ✓ | – |

Train at a size despot can produce, such as 80×80 or 40×40. The tracked Python
bridge accepts five scores. Check its action table against your model's output
order before enabling robot commands. [Python bridge mapping](gesture-pipeline.md#decision-rules)
