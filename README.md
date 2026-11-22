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

## Event-camera labeling demo

Use [the CVAT demo](labeling/README.md) to turn three clips from the local Durance gesture archive into labeling tasks. The `hello` clip includes a starter palm track. The other two clips are ready for annotation.

## Five-class gesture training

Use the [training instructions](training/README.md) to train a PyTorch spiking gesture classifier on the local archive.

The [left-palm pilot](training/README.md#40-40-left-palm-forecast) includes a 40×40 Q2.6 model, held-out results, and a coordinate bridge.

## Team

_TBD_

## License

_TBD_
