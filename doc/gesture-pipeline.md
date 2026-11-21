# Gesture pipeline: event camera → FPGA SNN → Go2 (working setup)

What is running on our Kria today: the GenX320 event camera feeds the **best model** SNN on the
SPLEAT FPGA, a small Python bridge turns its outputs into Unitree sport commands, and the **real
Go2 reacts to gestures**. A 3D Go2 in RViz and a browser view are available for testing without
the robot. Code: `kria/` (runs on the board), `ros2/` (ROS 2 side, runs on the laptop).

This is a Python alternative to the C++ `gesture_node` + `action_bridge` in `go2-kria`
([go2-kria-overview.md](go2-kria-overview.md)): no cross-compiling, and it handles the 5-output
model directly.

```
Kria KV260
  GenX320 ─► despot_send_and_forget ─► SPLEAT server :3333 (SNN on FPGA, ~7.5 ms, 100 Hz)
                                              │ 5 scores / 10 ms
                                              ▼
                                   kria/live_gesture_bridge.py ──► /api/sport/request ──► Go2
                                              ▲                     (domain 0, USB-Ethernet,
                       obstacle_detection ────┘                      192.168.123.99 → .161)
                       (/durance/clearance from the Go2 LiDAR)
  kria/snn_input_viewer.py ─► http://192.168.77.1:8080  (what the SNN sees + output bars)

Laptop (192.168.77.2, cable to the Kria's built-in port)
  browser view · RViz 3D Go2 (ros2/rviz) · local simulator (ros2/sim_playground.py)
```

## Files

| Path | Where it runs | What it does |
|---|---|---|
| `kria/inference.sh` | Kria `/root/` | One control script: `start [--headless]`, `robot [--dry-run]`, `stop`, `status` |
| `kria/live_gesture_bridge.py` | Kria (or laptop) | SPLEAT outputs → firing rates (0.5 s window) → trick decision → `/api/sport/request` |
| `kria/snn_input_viewer.py` | Kria | Passively decodes the despot↔SPLEAT traffic on loopback; serves the 80×80 SNN input + output bars on port 8080 |
| `kria/robot_link.sh` | Kria | ROS 2 env for the real Go2 (domain 0, `/etc/cyclonedds/cyclonedds-go2.xml`) |
| `kria/laptop_link.sh` | Kria | ROS 2 env for the laptop link (test domain 42 over `eth0`) |
| `kria/board_screen_pub.py` | Kria | Publishes the board's X screen as `/kria/camera_view` (optional) |
| `kria/event_web_viewer.py` | Kria | Raw 320×320 camera in a browser via v4l2. **Needs the camera to itself** - stop inference first |
| `ros2/env.sh` | Laptop | Jazzy + locally unpacked CycloneDDS, loopback-only domain 73 (simulator) |
| `ros2/env_board_link.sh` | Laptop | Same, but domain 42 over the cable to the Kria (RViz mirrors the board) |
| `ros2/sim_playground.py` | Laptop | Fake SPLEAT server + `robot_sim` + real pipeline; type gesture numbers to test the action mapping |
| `ros2/rviz/` | Laptop | Animated Go2 in RViz driven by `/api/sport/request` (`fetch_go2_model.sh` downloads the URDF once) |
| `models/gesture_snn_example.json` | - | Example River Model Graph (2×80×80 → 5) used to test the firmware generator |

Not in the repo (organiser material): `best_model.json`, `best_model_parameters.txt`
(the weights) and `firmware-kv260 (3).tar.gz` (bitstream + dtbo) from the riversdk.dev job.

## Board setup (once per SD card)

From the laptop:

```bash
# Go2 ROS 2 runtime (prebuilt in go2-kria-generated.zip)
scp data/go2-kria/kria-runtime.tar.gz root@192.168.77.1:/tmp/
ssh root@192.168.77.1 'r=$(mktemp -d /opt/go2-XXXXXX) && tar -xzf /tmp/kria-runtime.tar.gz -C $r && ln -sfn $r /opt/go2-current'

# our scripts
scp kria/* root@192.168.77.1:/root/

# best model: firmware from the riversdk.dev job + its 16-bit weights
scp spleat.bit.bin spleat.dtbo root@192.168.77.1:/usr/lib/firmware/xilinx/spleat/
scp best_model_parameters.txt root@192.168.77.1:/root/weights.txt
ssh root@192.168.77.1 'systemctl stop spleat-server; xmutil unloadapp; xmutil loadapp spleat; systemctl start spleat-server'
```

Back up the original `/root/weights.txt` and `/usr/lib/firmware/xilinx/spleat/` first.
`config.yaml` and `metavision_camera.json` stay as shipped (2×80×80 in, 5 out, scaling 4096,
100 Hz, downsampling 4) - they match `best_model.json`.

**Firmware and weights must belong together.** The shipped firmware uses 8-bit weights
(−128…127); the best-model firmware expects the 16-bit `best_model_parameters` (−15894…11041).
Mixing them gives all-zero outputs. A bitstream generated with *Prophesee IP = None* has no camera
interface (its `spleat.dtbo` is ~5 KB instead of ~14.6 KB) - always pick **GenX320**.

## Run it on the real Go2

Cables: laptop ↔ Kria built-in Ethernet; USB-Ethernet adapter on the Kria ↔ Go2. Go2 standing,
~2 m free space, someone holding the remote.

```bash
ssh root@192.168.77.1
/root/inference.sh start --headless   # camera → FPGA SNN, browser view on :8080
/root/inference.sh robot --dry-run    # decisions are logged, robot gets nothing
/root/inference.sh robot              # live
tail -f /tmp/bridge.log               # what it decides
/root/inference.sh robot --dry-run    # stop commanding (or use the remote)
```

Watch http://192.168.77.1:8080 to see which motion raises which output.

### Decision rules (`kria/live_gesture_bridge.py`)

| Setting | Value | Meaning |
|---|---|---|
| `WINDOW` | 50 frames | firing rate of each output over the last 0.5 s |
| `MIN_RATE` / `MIN_MARGIN` | 0.50 / 0.20 | winner fires ≥ 50 % and beats the next *gesture* by 20 points (the "none" output is not compared) |
| `STABLE_S` | 0.4 s | winner must hold this long |
| `COOLDOWN_S` | 2 s | camera input is discarded while a trick runs and for 2 s after |
| `QUIET_RATE` | 0.20 | re-arm only after all gesture outputs drop below this (breaks self-triggering loops) |
| `--require-clear` | robot mode | trick starts only while `/durance/clearance` is CLEAR (0.5 m radius) |

`ACTIONS` maps outputs to tricks: o0 → Sit (+ Stand up), o1 → Hello, o2 → Dance, o3 → Stretch,
o4 → none. o4 is clearly "no gesture" (it dominates when nobody moves); **o0–o3 are placeholders**
until each dataset gesture has been checked against the bars.

## Test without the robot

- **Board + RViz:** on the board `/root/inference.sh start` (uses `laptop_link.sh`, domain 42),
  on the laptop `source ros2/env_board_link.sh && ros2 launch ros2/rviz/view_go2.launch.py`.
  Test commands by hand: `ros2 topic pub --once /api/sport/request unitree_api/msg/Request "{header: {identity: {api_id: 1016}}}"`.
- **Laptop only:** build `go2-kria/src` with colcon on Jazzy, then `python3 ros2/sim_playground.py`.
  CycloneDDS was installed without sudo:
  `apt-get download ros-jazzy-cyclonedds ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-rosidl-generator-dds-idl nlohmann-json3-dev ros-jazzy-iceoryx-{binding-c,hoofs,posh}`
  and `dpkg-deb -x` each into `$GO2_EXTRA` (see `ros2/env.sh`).

The RViz dog is an animation of the commanded trick, not a physics simulation.

## Gotchas we hit

- **The event camera must not move.** Mounted on the robot (or hand-held), its own motion makes
  the model fire o0 at ~1.0 and the dog loops Sit → Stand up. Put it on a table or tripod facing
  the person.
- **Only one program can use the camera**: despot (inference), river-tools capture and
  `event_web_viewer.py` exclude each other. `snn_input_viewer.py` is passive and always fine.
- **Laptop firewall (ufw):** river-tools needs `sudo ufw allow in on eno1 from 192.168.77.0/24 to any port 3333 proto tcp`;
  ROS 2 to/from the board needs `sudo ufw allow in on eno1 from 192.168.77.0/24`.
- **Laptop network:** a NetworkManager profile with a static IP is needed
  (`nmcli con add type ethernet ifname eno1 con-name kria ipv4.method manual ipv4.addresses 192.168.77.2/24`);
  after re-plugging the cable run `nmcli con up kria`. `durance-kria.local` does not resolve - use `192.168.77.1`.
- **Board shell is BusyBox:** no `pkill`, `timeout`, `head -c`. Paste commands one line at a time.
- Humble (board) ↔ Jazzy (laptop) works for these topics; the "string data is not null-terminated"
  messages on the board are harmless.
- Streaming the raw camera via v4l2 without the sensor's event rate controller can overflow the
  MIPI receiver ("Stream Line Buffer Full"); `event_web_viewer.py` enables ERC and auto-restarts.
