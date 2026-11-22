# Go2 Next Reflex

**Robotics Hackathon 2026 – Durance AI / Durance Robot Lab track: "Give a robot dog its next reflex"**

## Challenge

Build a new event-vision behavior for a **Unitree Go2 X**:

1. **Record** event-camera data
2. **Train** a spiking neural network (SNN)
3. **Deploy** it on a **AMD Kria KV260** board
4. **React** – make the robot respond through **ROS 2**

## Pipeline

```
Event camera ──► Data recording ──► SNN training ──► Kria KV260 inference ──► ROS 2 ──► Unitree Go2 X
```

## Repository structure

| Path | Contents |
|---|---|
| `doc/` | Team notes: board setup, data capture, dataset, training, deployment, FPGA debugging, gesture pipeline |
| `training/` | SNN training pipeline (`python -m training.snn`), configs, quantization and export for the Kria |
| `training/optimised_snn/` | 8-bit quantization-aware SNN for the 40×40 firmware: training, FPGA replay tests, integer simulator, dataset tools (deployed models in `models/optimised_snn/`) |
| `remote/` | Scripts to run training on the GPU server (setup, push data, train, fetch results) |
| `scripts/` | Helper scripts: event and augmentation viewers, FPGA client (`fpga.py`), FPGA debugging checks (`fpga_debug/`) |
| `kria/` | Runs on the Kria: inference control script, FPGA output → Go2 command bridge, live SNN view |
| `ros2/` | ROS 2 side on the laptop: env files, simulator, animated Go2 in RViz |
| `models/` | Model graph JSON examples for the riversdk.dev firmware generator |

Start with [doc/gesture-pipeline.md](doc/gesture-pipeline.md) to run the gesture → Go2 pipeline.
For laptop-only testing, follow [Docker and RViz on Wayland](doc/gesture-pipeline.md#docker-and-rviz-on-wayland).

## Team

_TBD_

## License

_TBD_
