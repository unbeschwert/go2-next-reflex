# Local version: the setup as it runs on our laptop + Kria

A snapshot of the working folders used on the hackathon laptop and the Kria on 2026-11-21, kept in
their **original layout with the original paths** (`~/go2-hackathon/...` on the laptop,
`/root/...` on the board). The `gesture-pipeline` branch has the same code reorganised for the repo
(`kria/`, `laptop/`) with portable paths and fuller docs. Use this branch to reproduce the exact
local setup.

What it does: GenX320 event camera → SNN ("best model") on the Kria's SPLEAT FPGA → Python bridge
→ Unitree sport commands → the real Go2 does Sit / Hello / Dance / Stretch. RViz and a browser
view show the same decisions without the robot.

## What is in this branch

| Path in this branch | Local path | Contents |
|---|---|---|
| `local/go2_viz/` | `~/go2-hackathon/go2_viz/` | Bridge, board control script, viewers, RViz animator. The board's `/root/*.py` and `/root/*.sh` are copies of these files |
| `local/laptop_ws/` | `~/go2-hackathon/laptop_ws/` | ROS 2 Jazzy env files and the simulator (`src` is a symlink to the go2-kria sources, not committed) |
| `local/board/` | Kria `/root/` | Files that only exist on the board: `laptop_link.sh`, plus `event_web_viewer.py` (raw camera viewer) |
| `local/models/` | `~/go2-hackathon/models/` | `gesture_snn_example.json` (our example network for the firmware generator) |

Not committed: `go2-kria/` and `best_model*` / firmware / weights (organiser material),
`ros_extra/` (unpacked .deb packages), `laptop_ws/{build,install,log}`, `go2_viz/meshes` and the
URDFs (downloaded, 25 MB), `board_backup/` (original weights and firmware).

### `local/go2_viz/`

| File | Runs on | Purpose |
|---|---|---|
| `inference.sh` | Kria `/root/inference.sh` | `start [--headless]` camera → FPGA + bridge + browser view · `robot [--dry-run]` switch the bridge to the real Go2 · `stop` · `status` |
| `live_gesture_bridge.py` | Kria (or laptop) | SPLEAT outputs → firing rates → trick → `/api/sport/request`; `--require-clear`, `--dry-run`, `--no-viz` |
| `snn_input_viewer.py` | Kria | http://192.168.77.1:8080: the 80×80 SNN input and the 5 output bars (passive, always safe) |
| `board_screen_pub.py` | Kria | Board screen → `/kria/camera_view` (only used with `inference.sh start` without `--headless`) |
| `robot_link.sh` | Kria | ROS 2 env for the Go2: domain 0, `/etc/cyclonedds/cyclonedds-go2.xml` |
| `go2_animator.py`, `view_go2.launch.py`, `go2.rviz` | Laptop | Animated Go2 in RViz, driven by `/api/sport/request`; also shows the score bars |

### `local/laptop_ws/`

| File | Purpose |
|---|---|
| `env.sh` | ROS 2 Jazzy + CycloneDDS from `~/go2-hackathon/ros_extra/root`, loopback-only DDS domain 73, workspace overlay |
| `env_board_link.sh` | Same, but domain 42 over the cable to the Kria (use for RViz with the board) |
| `sim_playground.py` | Fake SPLEAT server + `robot_sim` + the go2-kria pipeline; type gesture numbers to test the mapping |

## Machines and network

| Machine | Address | Notes |
|---|---|---|
| Laptop (Ubuntu 24.04, ROS 2 Jazzy) | `eno1` 192.168.77.2 | NetworkManager profile `kria`. After re-plugging the cable the laptop falls back to `car-share-lan`: run `nmcli con up kria` |
| Kria KV260 (Yocto, ROS 2 Humble) | `eth0` 192.168.77.1 | `ssh root@192.168.77.1` (no password). `durance-kria.local` does not resolve |
| Go2 | 192.168.123.161 | Via the AX88179 USB-Ethernet adapter on the Kria (`enu1u1c2`, 192.168.123.99) |

ROS 2 domains: **73** laptop-only simulator (loopback) · **42** laptop ↔ Kria (RViz) · **0** Kria ↔ Go2.
Laptop firewall rules added: TCP 3333 from 192.168.77.0/24 (river-tools) and all traffic from
192.168.77.0/24 on `eno1` (ROS 2).

## State of the board

| Item | On the Kria |
|---|---|
| Firmware | best model: `firmware-kv260 (3).tar.gz` → `/usr/lib/firmware/xilinx/spleat/{spleat.bit.bin,spleat.dtbo}` |
| Weights | `best_model_parameters (2).txt` → `/root/weights.txt` (16-bit, 638,337 values) |
| Network | `best_model.json`: 6 ConvLIF layers, 2×80×80 → 5 |
| Config | `/root/.config/spleat-server/config.yaml` and `/root/metavision_camera.json` unchanged |
| Go2 ROS runtime | `kria-runtime.tar.gz` extracted to `/opt/go2-*`, linked as `/opt/go2-current` |
| Original model backup | laptop `~/go2-hackathon/board_backup/` (`weights.txt`, `config.yaml`, `firmware/`) |

## Commands

**Real robot** (Go2 standing, ~2 m free space, someone holding the remote):

```bash
ssh root@192.168.77.1
/root/inference.sh start --headless   # camera → FPGA SNN + browser view
/root/inference.sh robot --dry-run    # log decisions only
/root/inference.sh robot              # live: gestures move the Go2
tail -f /tmp/bridge.log
/root/inference.sh robot --dry-run    # stop commanding (or use the remote)
```

**Board + RViz on the laptop** (no robot needed; `inference.sh start` puts the bridge on domain 42):

```bash
source ~/go2-hackathon/laptop_ws/env_board_link.sh
ros2 launch ~/go2-hackathon/go2_viz/view_go2.launch.py
ros2 topic pub --once /api/sport/request unitree_api/msg/Request "{header: {identity: {api_id: 1016}}}"   # manual test
```

**Laptop-only simulator:**

```bash
source ~/go2-hackathon/laptop_ws/env.sh
python3 ~/go2-hackathon/laptop_ws/sim_playground.py        # type 0-10, n, o, q
ros2 launch ~/go2-hackathon/go2_viz/view_go2.launch.py     # second terminal, same env
```

**Restore the original model on the board:**

```bash
ssh root@192.168.77.1 '/root/inference.sh stop'
scp ~/go2-hackathon/board_backup/firmware/spleat.{bit.bin,dtbo} root@192.168.77.1:/usr/lib/firmware/xilinx/spleat/
scp ~/go2-hackathon/board_backup/weights.txt root@192.168.77.1:/root/weights.txt
ssh root@192.168.77.1 'systemctl stop spleat-server; xmutil unloadapp; xmutil loadapp spleat; systemctl start spleat-server'
```

## Recreate the local setup

```bash
mkdir -p ~/go2-hackathon && cd ~/go2-hackathon

# 1. organiser workspace (from the RIVER SDK download list)
unzip ~/Downloads/go2-kria-generated.zip -d go2-kria

# 2. CycloneDDS & friends for Jazzy without sudo
mkdir -p ros_extra/debs && cd ros_extra/debs
apt-get download ros-jazzy-cyclonedds ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-rosidl-generator-dds-idl \
  nlohmann-json3-dev ros-jazzy-iceoryx-binding-c ros-jazzy-iceoryx-hoofs ros-jazzy-iceoryx-posh
for d in *.deb; do dpkg-deb -x "$d" ../root; done
cd ~/go2-hackathon

# 3. code from this branch
cp -r <repo>/local/go2_viz <repo>/local/laptop_ws <repo>/local/models .
ln -s ../go2-kria/go2-kria/src laptop_ws/src
(source laptop_ws/env.sh && cd laptop_ws && colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF)

# 4. Go2 model for RViz (unitreerobotics/unitree_ros, BSD-3-Clause)
cd go2_viz && B=https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/go2_description
mkdir -p meshes && curl -sfL -o go2_description.urdf $B/urdf/go2_description.urdf
for m in base calf calf_mirror foot hip thigh thigh_mirror; do curl -sfL -o meshes/$m.dae $B/meshes/$m.dae; done
sed "s#package://go2_description/dae/#file://$PWD/meshes/#g" go2_description.urdf > go2.urdf && cd ..

# 5. laptop network profile for the board
nmcli con add type ethernet ifname eno1 con-name kria ipv4.method manual ipv4.addresses 192.168.77.2/24 ipv6.method link-local autoconnect no

# 6. board: Go2 runtime + scripts (+ best model firmware/weights, see "State of the board")
scp go2-kria/go2-kria/kria-runtime.tar.gz root@192.168.77.1:/tmp/
ssh root@192.168.77.1 'r=$(mktemp -d /opt/go2-XXXXXX) && tar -xzf /tmp/kria-runtime.tar.gz -C $r && ln -sfn $r /opt/go2-current'
scp go2_viz/{inference.sh,live_gesture_bridge.py,snn_input_viewer.py,board_screen_pub.py,robot_link.sh} \
    <repo>/local/board/laptop_link.sh root@192.168.77.1:/root/
```

## Things to know

- **Keep the event camera still** (table or tripod). On the robot, its own motion makes output o0
  fire continuously and the dog loops Sit → Stand up.
- Output mapping in `live_gesture_bridge.py` (`ACTIONS`): o0 Sit, o1 Hello, o2 Dance, o3 Stretch,
  o4 none. o4 is confirmed as "no gesture"; o0–o3 are still to be checked against the dataset gestures.
- Trigger rules: ≥ 0.50 firing rate, 0.20 lead over the next gesture, held 0.4 s; camera ignored
  during a trick + 2 s; re-arm after all gesture outputs drop below 0.20; robot mode only acts
  when the LiDAR clearance is CLEAR.
- Only one program can use the camera: despot (inference), river-tools capture and
  `event_web_viewer.py` exclude each other. `snn_input_viewer.py` is passive.
- The board shell is BusyBox: no `pkill` or `timeout`; paste commands one line at a time.
