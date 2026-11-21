# `go2-kria` reference workspace – overview

`data/go2-kria` contains the supplied ROS 2 workspace for the Go2 reflex demo.
Extract it from `go2-kria-generated.zip`. The original cross-compilation example is
in `ros-docker-original.zip`, which extracts to `data/ros-docker`. Both directories
are Git-ignored. The camera client, FPGA overlay and model come from the board
deployment. [Downloads](README.md#downloads-from-the-river-sdk-download-list) · [Board setup](kria-setup.md) · [SPLEAT deployment](deploy-model.md)

## Entry points

| Entry point | Gesture input | ROS 2 behavior |
|---|---|---|
| `ros2 launch bringup pipeline.launch.py` | 11 DVS-Gesture outputs over SPLEAT TCP | `gesture_node` publishes `/durance/gestures` and `action_bridge` selects a timed trick |
| `kria/inference.sh robot` | Five outputs over SPLEAT TCP | Starts the compiled `obstacle_detection` node and the tracked `live_gesture_bridge.py`, which selects a trick from its own action table |
| `ros2/sim_playground.py` | 11 synthetic DVS-Gesture outputs | Starts the C++ launch with fake SPLEAT and LiDAR data, and runs `robot_sim` |

Both gesture bridges publish `unitree_api/msg/Request` on `/api/sport/request` and
gate gesture-triggered requests on `msgs/msg/Clearance` on `/durance/clearance` in
robot mode. The C++ bridge handles rejection on `/api/sport/response`. The Python
bridge uses fixed timers for its actions and follow-up. [Five-output bridge and action table](gesture-pipeline.md#decision-rules)

| DDS domain | Use |
|---|---|
| `0` | Kria to physical Go2 via the robot-facing USB Ethernet adapter |
| `42` | Kria to laptop RViz via the Kria's built-in Ethernet port |
| `73` | Loopback for the laptop playground and the supplied board smoke test |

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
                                   action_bridge  (highest score → Unitree api_id)
                                              │  /api/sport/request
                                              ▼
                                         Unitree Go2
```

## Packages (`src/`)

| Package | What it does |
|---|---|
| `gesture_node` | Connects to the SPLEAT server over TCP, reads each frame's 11 floats, averages them over a 50-frame window (`window` parameter), and publishes to `/durance/gestures`. Drops any frame whose payload isn't exactly 11 floats. |
| `action_bridge` | Takes the highest score, looks up a trick in `action_map.cpp`, and starts it when clearance is `CLEAR`. It stays busy for the mapped duration or until a rejection. Sit and Lie Down schedule a timed Stand Up (`1004`) that bypasses the initial clearance check. |
| `obstacle_detection` | Checks the Go2's LiDAR pose and height map for obstacles. By default it checks cells 0.35–0.5 m from the robot and 0.2–4 m above the floor. |
| `msgs` | Custom messages: `Gestures.msg` (11 scores in DVS-Gesture class order), `Clearance.msg`, `BBox.msg`. |
| `unitree_api`, `unitree_go` | Copies of Unitree's ROS 2 message definitions (`Request`, `Response`, `SportModeState`, `HeightMap`, …). |
| `bringup` | `pipeline.launch.py` starts `gesture_node`, `action_bridge`, `profiling_node` and `obstacle_detection`, with an optional camera client. |
| `qualia_sim` | Fake SPLEAT server on port 3333 that sends random scores every 2 s. |
| `robot_sim` | Fake Go2: answers sport requests and publishes `/sportmodestate`. |
| `debug_node` | `gesture_monitor`, a live bar chart of the scores in the terminal. |
| `profiling_node` | Publishes how many gesture messages per second arrive, on `/profiling/fps_gestures`. |
| `common` | Shared helper (clearance state → text). |

### Reference action table

The table is defined in `data/go2-kria/src/action_bridge/src/action_map.cpp`.

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

## Docker build and deployment

`data/ros-docker` is the original generic Humble and Yocto build environment. Its
`manage.sh build` creates the builder image and `manage.sh enter` starts a shell
with the current directory mounted. The folder ships an example ROS package.

`data/go2-kria` has its own builder and packaging scripts for the 12 Go2 packages.
Its `Containerfile` runs on AMD64 Ubuntu 22.04. The Durance Yocto SDK supplies the
host-native colcon, CMake, Python and CycloneDDS IDL generator, along with the
AArch64 target compiler and sysroot. The build produces a ROS 2 Humble overlay with
the C++ executables and generated AArch64 Python 3.12 message bindings.

To rebuild for the board, start at the repository root with the x86_64 SDK installer
from the same Yocto release as the board image.

```bash
cd data/go2-kria
mkdir -p sdk
cp ../x86_64_sdk_installer.sh sdk/installer.sh
./manage.sh image
./manage.sh check
./manage.sh build
./manage.sh package
```

Colcon writes `install-yocto/`. `manage.sh package` archives that overlay and
`smoke.py` into `kria-runtime.tar.gz`. The workspace also ships a prebuilt copy.
Its packaged launch file and the `Gestures`, `Clearance` and `Request` definitions
match the supplied source. On the Kria, source ROS 2 Humble and the extracted
overlay. `smoke.py` exercises the C++ pipeline with fake inputs on domain 73 and can
run with the Go2 disconnected. The tracked board scripts expect the overlay at
`/opt/go2-current` and are copied separately to `/root/`.
[Board installation steps](gesture-pipeline.md#board-setup-once-per-sd-card)

The original C++ launch can also run on the Kria with `ros2 launch bringup
pipeline.launch.py despot:=false port:=3333` once SPLEAT and the camera client are
running. It expects an 11-score model. Use domain 0 with the robot-facing
CycloneDDS configuration. The supplied `data/go2-kria/README.md` has the full steps.

The laptop RViz workflow builds the same source for the desktop using ROS 2 Jazzy.
[Docker and RViz on Wayland](gesture-pipeline.md#docker-and-rviz-on-wayland)

## Adapting the C++ pipeline

The reference `gesture_node` drops a five-score frame before looking at its scores.
To run a five-output model through the C++ path, update the `Gestures.msg` array
length, `GESTURE_COUNT` in the action bridge header, and the table in
`action_bridge/src/action_map.cpp` together. `gesture_node` derives the expected
byte count from the message array size. SPLEAT frames use a four-byte big-endian
length followed by a CBOR map whose `payload` is a byte string of little-endian
float32 scores. `scripts/smoke.py` shows how the test generates those frames.

The C++ action bridge uses the highest score without a confidence threshold. It
picks class 0 for an all-zero input, so the laptop playground sends the `Other` class at idle.
The reference implementation sends discrete tricks by setting
`header.identity.api_id` and leaving `Request.parameter` empty.

## Useful topics

| Topic | Type | Direction |
|---|---|---|
| `/durance/gestures` | `msgs/msg/Gestures` | `gesture_node` → `action_bridge` |
| `/durance/clearance` | `msgs/msg/Clearance` | `obstacle_detection` → either gesture bridge |
| `/api/sport/request` | `unitree_api/msg/Request` | Either gesture bridge → Go2 or `robot_sim`, plus the RViz animator when connected |
| `/api/sport/response` | `unitree_api/msg/Response` | Go2 or `robot_sim` → C++ `action_bridge` and diagnostics |
| `/utlidar/robot_pose` | `geometry_msgs/msg/PoseStamped` | Go2 or playground → `obstacle_detection` |
| `/utlidar/height_map_array` | `unitree_go/msg/HeightMap` | Go2 or playground → `obstacle_detection` |
| `/sportmodestate` | `unitree_go/msg/SportModeState` | Go2 or `robot_sim` → state subscribers |
| `/profiling/fps_gestures` | `std_msgs/msg/Float32` | `profiling_node` → diagnostics |
