# Durance AI dashboard

One web page to run the gesture-controlled Unitree Go2: event camera → spiking network on the Kria KV260
FPGA → gesture bridge → robot. It runs on the operator's laptop and reaches the board and training
machines over SSH only.

```
 browser ──HTTP (127.0.0.1:8800)──▶ dashboard (FastAPI, this folder)
                                     ├─ SSH ──▶ Kria board 192.168.77.1: status, inference modes, settings file,
                                     │            camera view (SSH tunnel), teleop, model deploy, autostart
                                     │            └─ ROS 2 domain 0 ──▶ Go2 (via the board's USB-Ethernet)
                                     ├─ SSH ──▶ training machines you add (cloud GPU or Slurm cluster)
                                     ├─ local simulator (ROS 2 domain 88, loopback): fake board + real
                                     │    gesture bridge + Go2 animator, RViz in the browser (container)
                                     └─ river-tools: cued recording from the board's camera
```

## Start

```bash
cd dashboard
./run.sh                 # first run creates .venv (needs uv); then open http://127.0.0.1:8800
./run.sh --test          # 36 automated tests (no board, robot or training machine needed)
```

The first visit asks you to create an operator account (password of at least 10 characters). To keep it
running: `durance-dashboard.service` (systemd user service, instructions inside). Configuration is
optional: see `config.example.toml` (copy it to `~/.config/durance-dashboard/config.toml`).

What each part needs on the laptop:

| Part | Needs |
|---|---|
| Home, Live, Board, Drive | Ethernet link to the board (192.168.77.1) and `ssh root@192.168.77.1` working with a key; the board's host key in `~/.ssh/known_hosts` |
| Real dog | Board in robot mode, `/root/robot_link.sh` on the board (Board → Install board software) |
| Simulator | ROS 2 Jazzy or Humble with the go2-kria workspace (`ros2/env.sh`, see [doc/gesture-pipeline.md](../doc/gesture-pipeline.md)); the dataset zip `own_gestures_s20-30.zip` in `paths.share` |
| RViz in the browser | Docker usable by your user; the Go2 model downloaded once with `ros2/rviz/fetch_go2_model.sh` |
| Record | `river-tools` ([doc/capture-data.md](../doc/capture-data.md)); set `paths.recorder_python` to its Python |
| Training | A GPU machine or Slurm cluster added in Training → Machines; the `riverops` package (`paths.riverops`) |

## Where it finds the project files

Nothing needs configuring in this repository: the dashboard looks up each file by name next to its own folder.

| Files | Folder |
|---|---|
| Board software: `live_gesture_bridge.py`, `bridge_settings.py`, `lidar_follow.py`, `inference.sh`, `robot_link.sh` | `kria/` |
| Go2 animator, `go2.urdf` + meshes, `go2.rviz` | `ros2/rviz/` |
| ROS environment: `env.sh` (simulator, loopback), `env_board_link.sh` (board test mode, domain 42) | `ros2/` |
| FPGA integer model `spleat_sim.py`, trainer `train_snn_v9.py` | `training/optimised_snn/` |
| `gestures.json`, `cut_takes.py` (Record page) | `training/optimised_snn/dataset/` |
| Weights to deploy (`optimised_snn/<classes>/weights_*.txt`, `fetched/<run>/` from training) | `models/` |

Recordings and datasets stay outside the repository: `paths.data` (clips, default `~/go2-hackathon/data`),
`paths.share` (dataset zips), `paths.captures` (raw takes, `~/river-captures`), the Durance AI team's
dataset zip (`paths.organisers_zip`). The older laptop layout (`~/go2-hackathon` with `go2_viz/`) works too.

## Pages

| Page | What it does |
|---|---|
| Home | Is the system ready: board, robot link, live stream, safety state, warnings; start/stop inference |
| Live | Live event-camera view, detected gesture, output bars with thresholds, event log; every gesture-control setting with its own reset button, presets (Responsive / Balanced / Strict) |
| Simulation | Real RViz of the simulated robot or of the board's test mode, in the browser; local simulator with a virtual person |
| Drive | Manual control: dead-man joystick or W/S A/D Q/E keys, speed limit, sport commands (stand, sit, heart, ...), STOP |
| Record | Guided recording of labelled gesture takes from the board's camera, live preview, cut into 1 s clips |
| Dataset | Browse every dataset, play a clip next to the network's spikes for it, relabel or reject clips |
| Training | Machines (add a cloud GPU or cluster), new run (dry run first), jobs, runs table, fetch weights |
| Board | Inference modes, model deploy with backup/restore, install board software, logs, SD-card mode |
| Security | Audit log of every action, password change |

## Safety

- **E-STOP** on every page (button or Escape). In parallel and without waiting for the board lock, it
  sends a StopMove burst through the manual-control link, pauses gesture control (the bridge aborts a
  running turn or follow within 0.2 s and sends StopMove), sends StopMove with `go2_cmd.py` on the board,
  and switches manual control off. While it is active nothing can move the robot. Clearing it does not
  resume gestures: that is a separate, deliberate step.
- **Manual control** is a dead-man control: moves must keep arriving or the robot stops after 0.4 s. The
  teleop process on the board enforces this too, and also stops the robot when its SSH link drops. Speeds
  are clamped on both sides (0.5 m/s forward, 0.3 m/s sideways, 1.0 rad/s turning). Manual mode pauses
  gesture control.
- Everything that can move the real dog asks first: robot mode, manual control of the real dog, risky
  commands (lie down, dance, rear up, damp), SD-card mode.
- **Board lock** (`/root/BOARD_LOCK`, shared with the other tools): every board change takes the lock, and
  taking it from someone else needs a confirmation. E-STOP ignores the lock.
- **Robot mode** always runs with the LiDAR obstacle check (`--require-clear`). Deploying a model never
  re-enters robot mode by itself.
- **Model deploy**: the new weights must match the firmware's network (value count), the upload is
  md5-verified, the old weights are backed up on the board (`/root/dashboard_backups/`) and can be restored
  with one click.
- The settings file can never make the bridge more trigger-happy than allowed: every value has a range and
  anything invalid falls back to the tuned default (schema: `kria/bridge_settings.py`).

## Security

- Binds to **127.0.0.1** only. Exposing it needs `allow_lan = true` and HTTPS in front (reverse proxy,
  `secure_cookie = true`).
- Login with scrypt-hashed passwords (`users.json`, mode 0600), 5 attempts per minute, server-side
  sessions (HttpOnly, SameSite=Strict cookie), a CSRF token on every state change, same-origin and Host
  header checks (DNS rebinding), strict Content-Security-Policy, no third-party code at runtime (three.js
  and noVNC are vendored in `static/vendor/`).
- No arbitrary commands: every remote command is a fixed template and every variable part is validated
  and shell-quoted. Training flags are whitelisted with types and ranges; job names, run names, clip
  paths and session numbers must match strict patterns (path traversal is refused).
- SSH: key-based, `BatchMode` (never prompts), strict host keys, one multiplexed connection per host.
  Nothing is installed on the board for status, live view or manual control (helper scripts are sent
  with the command); only "Install board software", deploy and SD-card mode write files, all backed up.
- **Audit log** of every action (user, time, parameters, result): Security page, `audit.jsonl` in the
  state folder (`~/.local/share/durance-dashboard`).

## Board software

The dashboard needs on the board: `live_gesture_bridge.py` with live settings, `bridge_settings.py`,
`lidar_follow.py` and `robot_link.sh`. Board → "Install board software" copies them from `kria/` (the old
files are backed up first). Until then Home shows a warning and the settings and the E-STOP pause are not
applied by the bridge (E-STOP still sends StopMove directly).

## SD-card mode (autostart)

Board → Auto mode → Install. The board then starts by itself at power-on, without the laptop: it waits
`start_delay_s`, clears a pause left from the last session (unless `keep_pause`), starts camera + FPGA
model (`inference.sh start --headless`, 3 tries), waits up to `wait_go2_s` for the Go2 on its cable and
enters robot mode (or `robot-dry` / `test`). If the Go2 never appears it stays in test mode and sends no
robot commands. Files on the board: `/root/durance_auto.sh`, `/root/durance_auto.conf`,
`/etc/systemd/system/durance-auto.service`, log `/root/durance_auto.log`. To switch it off without the
laptop, create `/root/DURANCE_AUTO_DISABLE` (for example with the SD card in a reader).

## Simulator

Simulation → Start. The first start replays 6 recorded clips per gesture (session 29) through the FPGA
integer model once (about a minute, then cached). The fake board streams those outputs and their camera
frames; the **real** `live_gesture_bridge.py` decides and commands the Go2 animator, all on ROS domain 88
over loopback, so it can never reach the board or the dog. The virtual-person buttons choose the gesture,
and every setting on the Live page works exactly as on the board.

## RViz in the browser

The Simulation page shows the real RViz 2 (Go2 meshes, `go2.rviz`) running in the `durance-rviz` container:
a virtual screen with VNC on 127.0.0.1 only, a fresh VNC password per start, read-only root, no
capabilities, your user id. The browser gets the picture only through the dashboard's logged-in WebSocket
(`/api/rviz/vnc`), drawn by the vendored noVNC client. Source "Simulator" follows the local simulator
(domain 88); "Board test mode" follows the board's test mode (domain 42, an animator is started on the
laptop if none runs).

One-time setup:
1. Docker for your user: `sudo usermod -aG docker $USER`, then log out and in.
2. The Go2 model: `ros2/rviz/fetch_go2_model.sh` (downloads the URDF and meshes, about 25 MB).
3. Simulation → "Build RViz image" (`docker build -t durance-rviz:1 docker/rviz`, downloads ROS 2 Jazzy once).

The image only holds RViz: the Go2 model and the RViz config are mounted read-only from `ros2/rviz/`, so
changing them needs no rebuild.

## Training

The dashboard ships with no training machine. Training → Machines → **Add machine**:
- **GPU machine** (a cloud VM: the run starts in the background) or **Slurm cluster** (sbatch / squeue / scancel);
- host, user and port, or an alias from `~/.ssh/config`;
- the login key: one of your standard keys in `~/.ssh`, or **Generate new key** and paste the public key
  into the cloud provider. Passwords are never stored;
- the work folder on the machine, its Python, and for Slurm the job script name.

Then **Server ID**: compare the server's ID with the one your provider shows, and **Trust this server**.
**Test** shows a checklist (SSH login, work folder, GPU, Python + PyTorch, riverops, training script + data,
Slurm), **Prepare** copies
`train_snn_v9.py`, `riverops` and the chosen dataset zips to the work folder.

Machines are stored in the private `clusters.toml` next to the config (mode 0600; the dashboard refuses to
start when others can read it). The pages show only a machine's name and whether it is reachable; connection
problems appear as a short reason ("login refused", "connection timed out") and the full text goes to the
dashboard's own log.

A run: New run → preset, dataset, architecture → **Dry run** (checks the machine and shows the command,
starts nothing) → Start training → Jobs → Runs table → **Fetch** copies the weights to
`models/fetched/<run>/` → Board → Deploy. Only `nb40` (40x40, 8-bit firmware) and `best80` (80x80,
firmware 3) run on the firmwares that exist today; other architectures need a firmware build first.

## Container

Docker must be usable by your user (see above). The compose plugin is optional:

```bash
cd dashboard
docker/run-dashboard.sh build     # image durance-dashboard:1.0.0 (python:3.12-slim + the app)
docker/run-dashboard.sh start     # http://127.0.0.1:8800 ; the first visit creates the login
docker/run-dashboard.sh status    # "Up ... (healthy)"
docker/run-dashboard.sh logs      # follow the log
docker/run-dashboard.sh stop      # users and audit log stay in the volume durance-state
```

The container uses host networking (board link and loopback), runs as your user with a read-only root, no
capabilities and no-new-privileges. It mounts `~/.ssh` read-only, the repository as `/project`,
`~/river-captures`, and keeps its state in the volume `durance-state`. `docker-compose.yml` holds the same
settings. Do not run the container and `./run.sh` at the same time: both use port 8800
(`CONFIG=other.toml docker/run-dashboard.sh start` uses another config and port).

In the container: board, dataset, settings, manual control, E-STOP, training (with `riverops` inside the
mounted folder, see `docker/config.container.toml`). From the laptop install
(`./run.sh`): the simulator (ROS 2), recording (river-tools) and starting the RViz container. Giving the
dashboard container the Docker socket would make it root-equivalent on the laptop, so that is not done.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Board not available ... not reachable" | Cable or network profile: `nmcli con up kria`, laptop address 192.168.77.2 |
| "Host key verification failed" | Board re-flashed: `ssh-keygen -R 192.168.77.1`, then `ssh root@192.168.77.1` once |
| Camera view shows "camera not available" | Inference not running on the board (Home → Start) |
| Warning "scaling is X, but ... needs Y" | The board's `config.yaml` does not match the input size: `inputs.scaling` = 4096 / downsampling² (64 for 40x40, 256 for 80x80) |
| Settings show "not applied" | Old bridge on the board: Board → Install board software, then restart inference |
| Drive: "ROS environment missing" | `/root/robot_link.sh` (robot) or `/root/laptop_link.sh` (test) missing on the board |
| Simulator: "ROS 2 environment not available" | `ros2/env.sh` fails: ROS 2 or the go2-kria workspace missing (`GO2_WS`) |
| Simulation: "Go2 model ... missing" | Run `ros2/rviz/fetch_go2_model.sh` once |
| Training: "no training machine" | Expected until you add one (Training → Machines) |
| Logs | Dashboard: terminal or `journalctl --user -u durance-dashboard`; helpers: `<state>/logs/`; board: Board → Logs |

## Code layout

```
durance_dash/      backend: app.py (routes), config.py (paths, file lookup), security.py, board.py, live.py,
                   settings_svc.py, robot.py, sim.py, rviz.py, recording.py, dataset.py, spikes.py,
                   training.py, machines.py, models.py, procs.py, sshrun.py, events.py
  board_scripts/   run on the board, sent over SSH (status probe, status/log tap, dead-man teleop)
  local_scripts/   laptop helpers (fake board, pose tap, headless recorder)
board/             SD-card mode files installed on the board
docker/            dashboard container, RViz container (docker/rviz/)
static/            browser UI (vanilla JS modules, vendored three.js and noVNC)
docs/API.md        HTTP API
tests/             pytest suite
```
