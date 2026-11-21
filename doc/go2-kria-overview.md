# `go2-kria` reference workspace – overview

`data/go2-kria` is the **ROS 2 side** of an existing demo: the event camera sees a **hand gesture**, an SNN on the FPGA classifies it, and the Go2 performs a **trick** (sit, wave, dance…).

It contains no camera or SNN code. Per its README, the FPGA overlay, camera client and model files "come from your board deployment". This workspace starts once the SNN has produced scores.

> Note: `data/` is git-ignored, so this workspace is not part of the repo. Get it from `go2-kria-generated.zip` on the RIVER SDK download list (see [README.md](README.md)). Board setup is in [kria-setup.md](kria-setup.md), and the SPLEAT side in [deploy-model.md](deploy-model.md).

## Our plan vs. what exists

Our plan: capture events from a **Prophesee GenX320 (MIPI)** on the Kria KV260 FPGA → **train an SNN** → turn its outputs into **Unitree Go2 SDK commands** that move the dog.

## Data flow

```
GenX320 ──► camera client ("despot") ──► SPLEAT server (SNN on the FPGA)
                                              │  TCP, port 3333 (or 41017)
                                              │  frame = 4-byte length + CBOR {"payload": 11 × float32}
                                              ▼
                                   gesture_node   (mean over the last 50 frames)
                                              │  /durance/gestures  (float32[11])
LiDAR ─► obstacle_detection ─► /durance/clearance (CLEAR / OBSTACLE / UNKNOWN)
                                              ▼
                                   action_bridge  (highest score → Unitree api_id; only sends when CLEAR)
                                              │  /api/sport/request
                                              ▼
                                         Unitree Go2
```

## Packages (`src/`)

| Package | What it does |
|---|---|
| `gesture_node` | Connects to the SPLEAT server over TCP, reads each frame's 11 floats, averages them over a 50-frame window (`window` parameter), and publishes to `/durance/gestures`. Drops any frame whose payload isn't exactly 11 floats. |
| `action_bridge` | Takes the highest score, looks up a trick in `action_map.cpp`, sends it on `/api/sport/request`, then ignores new input until that trick's time is up. Tricks that leave the robot sitting or lying down are followed by Stand Up (`1004`). Only sends when clearance is `CLEAR`. |
| `obstacle_detection` | Uses the Go2's LiDAR (`/utlidar/robot_pose`, `/utlidar/height_map_array`). Blocks motion if anything between 0.2 m and 4 m high is within 0.35–0.5 m of the robot. |
| `msgs` | Custom messages: `Gestures.msg` (11 scores in DVS-Gesture class order), `Clearance.msg`, `BBox.msg`. |
| `unitree_api`, `unitree_go` | Copies of Unitree's ROS 2 message definitions (`Request`, `Response`, `SportModeState`, `HeightMap`, …). |
| `bringup` | `pipeline.launch.py` starts everything. |
| `qualia_sim` | Fake SPLEAT server on port 3333 that sends random scores every 2 s. |
| `robot_sim` | Fake Go2: answers sport requests and publishes `/sportmodestate`. |
| `debug_node` | `gesture_monitor`, a live bar chart of the scores in the terminal. |
| `profiling_node` | Publishes how many gesture messages per second arrive, on `/profiling/fps_gestures`. |
| `common` | Shared helper (clearance state → text). |

### Gesture → action table (`action_bridge/src/action_map.cpp`)

| # | Gesture (DVS-Gesture) | api_id | Action | Busy (s) |
|---|---|---|---|---|
| 0 | Hand clapping | 1009 | Sit (then Stand Up) | 5 |
| 1 | Right hand wave | 1004 | Stand Up | 3 |
| 2 | Left hand wave | 1005 | Lie down (then Stand Up) | 5 |
| 3 | Right arm cw | 1010 | Rise from sit | 3 |
| 4 | Right arm ccw | 1016 | Hello (waves a paw) | 5 |
| 5 | Left arm cw | 1017 | Stretch | 3 |
| 6 | Left arm ccw | 1020 | Content | 3 |
| 7 | Arm roll | 1022 | Dance | 15 |
| 8 | Air drums | 1029 | Rear up and greet | 3 |
| 9 | Air guitar | 1036 | Heart | 5 |
| 10 | Other gestures | – | No action | – |

## Building and running

- The build runs on a laptop in Docker (x86_64 Ubuntu 22.04) and cross-compiles for the Kria's ARM chip using the Durance Yocto SDK:
  ```bash
  mkdir -p sdk && cp /path/to/sdk_x86_64_installer.sh sdk/installer.sh
  ./manage.sh image     # build the Docker image with the SDK
  ./manage.sh check
  ./manage.sh build     # or: ./manage.sh build --packages-up-to gesture_node
  ./manage.sh package   # → kria-runtime.tar.gz
  ```
  It needs the SDK installer, which is **not** in the folder.
- `kria-runtime.tar.gz` is a ready-built copy. Copy it to the board, unpack it, source it, and run `smoke.py`. That's an end-to-end test on loopback with fake camera and LiDAR data, and it doesn't need the robot.
- To run with the robot: `ros2 launch bringup pipeline.launch.py despot:=false port:=3333`. Set the CycloneDDS config to the network adapter facing the robot (`192.168.123.x`). See `data/go2-kria/README.md` for the full steps.

## What changes for our plan

1. **The SNN's outputs.** This code expects 11 gesture scores (DVS-Gesture), but the reference Motion Buddy model has **5** (Sit, Lay down, Hello, Dance, No action), matching `outputs.channels: 5` in the server config. So even the reference model needs this change. If ours outputs something else (a different number of classes, or direction/speed values), change:
   - `msgs/msg/Gestures.msg`, the size of the score array
   - `GESTURE_COUNT` and the table in `action_bridge/src/action_map.cpp`
   - `gesture_node`, which expects exactly `scores.size() * 4` bytes per frame
2. **"Move the dog around" isn't supported yet.** `action_bridge` only sends a trick's ID (`request.header.identity.api_id`) and never sends parameters. Walking needs Unitree's Move command (`api_id` 1008) with speeds in `request.parameter`, something like `{"x": vx, "y": vy, "z": vyaw}`. *(This is from memory of the Unitree sport API, so check it against the Unitree SDK docs.)* This is probably the main ROS 2 work, either a new mode in `action_bridge` or a separate node.
3. **Low-confidence scores still trigger a trick.** It always picks the highest score and has no minimum; only the "Other" class does nothing. For continuous movement, add a minimum confidence plus a timeout that stops the robot.
4. **The SNN's output format is fixed by the existing code.** If our model runs on the SPLEAT server, it must send the same frames: 4-byte big-endian length + CBOR map with a `"payload"` byte string of little-endian float32s. `qualia_sim.cpp` and `scripts/smoke.py` show exactly how to build them.

## Useful topics

| Topic | Type | Direction |
|---|---|---|
| `/durance/gestures` | `msgs/Gestures` | gesture_node → action_bridge |
| `/durance/clearance` | `msgs/Clearance` | obstacle_detection → action_bridge |
| `/api/sport/request` | `unitree_api/Request` | action_bridge → Go2 |
| `/api/sport/response` | `unitree_api/Response` | Go2 → action_bridge |
| `/utlidar/robot_pose` | `geometry_msgs/PoseStamped` | Go2 → obstacle_detection |
| `/utlidar/height_map_array` | `unitree_go/HeightMap` | Go2 → obstacle_detection |
| `/sportmodestate` | `unitree_go/SportModeState` | Go2 → anyone |
| `/profiling/fps_gestures` | `std_msgs/Float32` | profiling_node → Foxglove etc. |
