# Durance AI dashboard

One web page to run the gesture-controlled Unitree Go2: event camera → spiking network on the Kria KV260
FPGA → gesture bridge → robot. It runs on the operator's laptop and reaches the board and training
machines over SSH only.

```
 browser ──HTTP (127.0.0.1:8800)──▶ dashboard (FastAPI, this folder)
                                     ├─ SSH ──▶ Kria board 192.168.77.1: status, inference modes, settings file,
                                     │            camera view (SSH tunnel), teleop, model deploy, autostart
                                     │            └─ ROS 2 domain 0 ──▶ Go2 (via the board's USB-Ethernet)
                                     ├─ SSH ──▶ GPU servers you add for training
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
| Training | A GPU server reachable over SSH, added in Training → Machines; the riverops wheel in `data/` (`paths.riverops`, never committed); `rsync` |
| Board → Build firmware | Your riversdk.dev session in `~/.config/riversdk/session` (`scripts/firmware_job.py`, [doc/deploy-model.md](../doc/deploy-model.md)) |

## Where it finds the project files

Nothing needs configuring in this repository: the dashboard looks up each file by name next to its own folder.

| Files | Folder |
|---|---|
| Board software: `live_gesture_bridge.py`, `bridge_settings.py`, `lidar_follow.py`, `inference.sh`, `robot_link.sh` | `kria/` |
| Go2 animator, `go2.urdf` + meshes, `go2.rviz` | `ros2/rviz/` |
| ROS environment: `env.sh` (simulator, loopback), `env_board_link.sh` (board test mode, domain 42) | `ros2/` |
| FPGA integer model `spleat_sim.py` (spike view) | `training/optimised_snn/` |
| `gestures.json`, `cut_takes.py` (Record page) | `training/optimised_snn/dataset/` |
| Training: our pipeline `python -m training.snn` and its YAML configs | `training/snn/`, `training/configs/` (`paths.training_configs`) |
| Runs of our pipeline (from Training → Fetch or `remote/fetch.sh`), installable as a board setup | `runs/` (`paths.runs`) |
| The team's weights files (`optimised_snn/<classes>/weights_*.txt`, `board_ready/`), replace only weights.txt | `models/` |

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
| Training | Machines (add a GPU server), new run from a YAML config (dry run first), jobs, runs table, fetch runs |
| Board | Inference modes, install a run (firmware + configs + weights) or deploy weights, build firmware, backup/restore, install board software, logs, SD-card mode |
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
- **Model deploy** (see [Models](#models)): a run installs only with firmware built for its network and
  formats, a server config whose input, outputs and scaling fit it, and classes the bridge knows; a weights
  file must match the firmware's network (value count) and weight format. Uploads are md5-verified, what the
  board ran before is backed up (`/root/dashboard_backups/`) and Restore puts it back with one click; a setup
  whose firmware does not load is rolled back at once.
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
  and shell-quoted. Training configs must be files of `training/configs/`, the form's keys are whitelisted
  with types and ranges and the result is validated by the pipeline's own loader; job names, run names, clip
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

The dashboard ships with no training machine. A training machine is a GPU server reached over SSH (e.g. the
one in `remote/remote.env`); runs start in the background there and the dashboard tracks their process.
Training → Machines → **Add machine**:
- host, user and port, or an alias from `~/.ssh/config`;
- the login key: one of your standard keys in `~/.ssh`, or **Generate new key** and paste the public key
  into the server's `~/.ssh/authorized_keys`. Passwords are never stored;
- the work folder on the server (a copy of the repository; the server's existing checkout works), its
  Python 3 and, if it is not on the PATH or in `~/.local/bin`, its uv.

Then **Server ID**: compare the server's ID with its real host key, and **Trust this server**.

Machines are stored in the private `clusters.toml` next to the config (mode 0600; the dashboard refuses to
start when others can read it). The pages show only a machine's name and whether it is reachable; connection
problems appear as a short reason ("login refused", "connection timed out") and the full text goes to the
dashboard's own log.

Training runs our pipeline (`python -m training.snn`, [training/README.md](../training/README.md)); the
work folder is a copy of the repository, as `remote/setup.sh` makes it:
- **Prepare**: installs uv if the machine has none (copied from this laptop), copies the code (git's tracked
  and untracked-but-not-ignored files) and the riverops wheel, runs `uv sync --group train` and uploads the
  dataset session folders you choose (`data/motion_buddy_dataset/NN/`; only changed files move).
- **Test**: SSH login, work folder, GPU, uv, code, PyTorch + riverops, data sessions.

A run: New run → **Config** (a YAML file of `training/configs/`, the default recipe first) → its values for
the common keys (seed, epochs, learning rate, optimizer, loss, weight averaging, QAT, splits, augmentation
probabilities) → **Dry run**: `python -m training.snn check` on this laptop (network, FPGA memory) and a
check of the machine, starts nothing → **Start training**: copies the code again and runs
`uv run --group train python -m training.snn train --config <yaml> --run-dir runs/<name>-<time> --set ...`,
one `--set section.key=value` per value you changed (the run's `config.yaml` and `run_info.json` record
them) → Jobs, log → Runs table (fixed-point val/test accuracy) → **Fetch** copies `runs/<name>/` to this
laptop's `runs/` → Board → Models → **Build firmware** (if no firmware for its network exists yet) →
**Install**. Other config keys: edit the YAML file; the form only offers keys it validates.

Older `clusters.toml` files may still have `kind` / `slurm_script`: they are ignored, and a Slurm cluster
entry is skipped (with a line in the dashboard log).

## Models

Board → Models lists two kinds:
- **Runs of our pipeline** (`runs/<name>/` with `deploy/`, `config.yaml`, `metrics.json`). Everything comes
  from the run's own files: the network from `deploy/model.json`, the formats from
  `deploy/export_summary.json`, the classes and splits from `config.yaml`, the accuracy (fixed point, what
  the FPGA computes) from `metrics.json`. A new YAML config needs no change in the dashboard. The firmware
  is any folder below `runs/` with `spleat.bit.bin`, `spleat.dtbo`, the `model.json` it was built from and
  `job.json` with its formats (`scripts/firmware_job.py` downloads it as `runs/<name>/firmware-job<ID>/`); a
  run uses one with the same graph and formats, its own first. **Install** writes the firmware, the server
  config, the camera's downsampling and frame rate (the board's sensor biases and noise filter stay), the
  weights and the classes (`board/install_setup.sh`, the steps of
  `firmware_performance_testing/switch_setup.sh`). A run that cannot work on the board is listed with the
  reason: no firmware yet, a server config from before the input-scaling fix (`scaling` must be
  4096 / downsampling² for despot's block sums), classes in an order the bridge does not know, or a graph that
  is not a plain chain of ConvLIF2d layers.
- **The team's weights files** under `models/`, recognised by their value count (`models.ARCHS`); they
  replace only `weights.txt` and are refused when the board's firmware is for another network or weight
  format.

The board status names the running firmware when it is one of the folders below `runs/` (by md5) and warns
when weights and firmware do not belong together. The Dataset page's spike view runs both kinds through the
integer model `spleat_sim.py`; for a run, with its layers and the fixed-point shifts of its formats. On 60
test clips of motion-buddy-80-v4 it gave the same winner as our fixed-point emulator on 57 (the board
matched the emulator on 19 of 20, [doc/fpga-debugging.md](../doc/fpga-debugging.md)), so it is a view of
the network's activity, not an exact replay.

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

In the container: board, dataset, settings, manual control, E-STOP, training (the riverops wheel inside the
mounted repository, see `docker/config.container.toml`; a machine without uv must get it from the laptop
install's Prepare). Board → Build firmware needs `uv` and the riversdk.dev session, so it runs from the
laptop install. From the laptop install
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
                   training.py (our pipeline on GPU machines), machines.py, models.py (weights catalogue),
                   snnruns.py (runs of our pipeline, their firmware), procs.py, sshrun.py, events.py
  board_scripts/   run on the board, sent over SSH (status probe, status/log tap, dead-man teleop)
  local_scripts/   laptop helpers (fake board, pose tap, headless recorder)
board/             files for the board: SD-card mode, install_setup.sh (sent over SSH to install a setup)
docker/            dashboard container, RViz container (docker/rviz/)
static/            browser UI (vanilla JS modules, vendored three.js and noVNC)
docs/API.md        HTTP API
tests/             pytest suite
```
