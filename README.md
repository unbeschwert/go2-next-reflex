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
| `doc/` | Team notes: board setup, data capture, dataset, training, deployment, ROS 2 side |
| `kria/` | Runs on the Kria: inference control script, FPGA output → Go2 command bridge, live SNN view |
| `laptop/` | Runs on the laptop: ROS 2 env files, simulator, animated Go2 in RViz |
| `models/` | Model graph JSON examples for the riversdk.dev firmware generator |
| `scripts/` | Helper scripts (event recording viewer) |

Start with [doc/gesture-pipeline.md](doc/gesture-pipeline.md) to run the gesture → Go2 pipeline.

## Team

_TBD_

## License

_TBD_
