# Durance AI dashboard: HTTP API (v1)

Backend: FastAPI on the laptop (`127.0.0.1:8800` by default). Every path below starts with `/api`.
The browser UI (`static/`) is served from `/` by the same server. All bodies are JSON.

## Conventions

- **Auth.** Session cookie `dd_session` (HttpOnly, SameSite=Strict). Every request other than
  `GET /api/auth/state`, `POST /api/auth/setup` and `POST /api/auth/login` needs a logged-in session (401 otherwise).
- **CSRF.** Every non-GET request needs the header `X-CSRF-Token: <csrf>`. The value comes from
  `GET /api/auth/state` (field `csrf`) and from the login/setup response. 403 if missing or wrong.
- **Errors.** `{"detail": "text"}` with status 4xx/5xx. 422 = invalid value (detail says which).
- **Confirmation.** Risky actions answer **409** `{"detail": "...", "confirm": "<question to show the user>"}`
  when the body does not contain `"confirm": true`. The UI shows the question in a dialog and, if the
  user agrees, sends the same request again with `"confirm": true`.
- **Tasks.** Slow actions return `{"task_id": "t17"}`. Progress: `GET /api/tasks/{id}` and SSE event `task`.
- **Events.** One Server-Sent-Events stream `GET /api/events` (cookie auth). Event names and payloads below.
  The UI opens it once after login and reconnects automatically.

## Auth

| Method, path | Body | Answer |
|---|---|---|
| GET `/auth/state` | | `{setup_needed, logged_in, user, csrf}` (`csrf` only when logged in) |
| POST `/auth/setup` | `{username, password}` (password >= 10 chars; only while `setup_needed`) | auth state |
| POST `/auth/login` | `{username, password}` | auth state (429 after 5 failures/min) |
| POST `/auth/logout` | | `{}` |
| POST `/auth/password` | `{old, new}` | `{}` |

## Overview and safety

GET `/overview` →
```json
{"target": {"mode": "board", "label": "Kria board 192.168.77.1 (SSH)"},
 "board": {"reachable": true, "ssh": true, "ping_ms": 1.2, "mode": "test",
           "processes": {"camera": true, "server": true, "bridge": true, "viewer": false, "obstacle": false},
           "model": {"md5": "056ee113", "name": "own_gestures_nb40_q8h", "classes": "own", "input": 40},
           "lock": {"owner": "laptop-2", "purpose": "80x80 test", "since": "07:48", "mine": false},
           "autostart": {"installed": false, "enabled": false, "mode": "robot"}},
 "robot": {"manual": false, "link": "rviz"},
 "live": {"connected": true, "fps": 99, "status": "listening", "last_command": {"label": "Heart", "t": 1795310000.1}},
 "training": {"configured": true, "host": "GPU training", "ok": true, "detail": "reachable, 290 ms", "running_jobs": 6},
 "safety": {"estop": false, "paused": false},
 "alerts": [{"level": "warn", "text": "Board locked by laptop-2 (80x80 test)"}]}
```
`board.mode`: `stopped | test | robot | robot-dry | unknown`. `robot.link`: `go2` (real dog, domain 0),
`rviz` (board test mode, domain 42), `sim` (local simulator), `none`.

| Method, path | Body | Answer |
|---|---|---|
| POST `/target` | `{mode: "board" \| "sim"}` | overview |
| POST `/estop` | `{}` (never needs confirm) | `{ok, actions: ["StopMove sent", "gestures paused", ...]}` |
| POST `/estop/clear` | `{}` | `{ok}` (gestures stay paused until resumed in Operation) |

## Board (Ethernet / SSH)

| Method, path | Body | Answer |
|---|---|---|
| GET `/board/status` | | overview `board` object (with `firmware: {md5, known, label, weights, potentials}`: the running bitstream, named when it is a firmware folder below runs/) + `{time, ros_domain, input_size, scaling, weights_md5}` |
| POST `/board/lock` | `{action: "acquire" \| "release" \| "takeover", purpose}` (takeover: confirm) | `{lock}` |
| POST `/board/inference` | `{action: "start" \| "start-headless" \| "robot" \| "robot-dry" \| "stop"}` (robot: confirm) | task |
| GET `/board/logs?name=bridge\|obstacle\|camera\|server&lines=200` | | `{name, lines: []}` |
| GET `/board/models` | | `{models: [...]}`: weights files `{id, name, path, md5, arch, input, classes, source, kind: "weights", formats: {weights}, metrics: {val, test, synops} \| null, deployed}` and runs of our pipeline `{id, name, kind: "setup", source: "runs", arch (config name), input, classes, class_names, formats: {weights, potentials, ...}, network: {size, layers, output}, splits, downsampling, server: {input, scaling, outputs}, firmware: {label, job, rel, weights, potentials, bit_md5} \| null, problems: [text], metrics: {val, test, kind}, md5, values, deployed}` (no local paths) |
| POST `/board/deploy` | `{model_id, classes: "own" \| "dataset"}` (confirm) | task. A run (`kind: "setup"`) installs a full setup: firmware, server config, camera downsampling/frame rate, weights, classes (backup first, rollback if the firmware does not load; 409 if it has `problems`). A weights file replaces weights.txt (409 if the board's firmware is for another network or weight format) |
| POST `/board/restore` | `{}` (confirm) | task (puts back what the board ran before the last deploy: the weights, or the whole setup) |
| POST `/board/firmware` | `{model_id}` (confirm) | task: `scripts/firmware_job.py submit --run runs/<name> --download` for a run without firmware (riversdk.dev session needed) |
| GET `/board/autostart` | | `{installed, enabled, active, conf: {mode, classes, start_delay_s}}` |
| POST `/board/autostart` | `{action: "install" \| "enable" \| "disable" \| "uninstall", conf?}` (install/enable: confirm) | task |

Autostart = "SD card mode": the board starts camera, FPGA model and gesture bridge by itself at boot
(`conf.mode`: `robot` = real Go2 with LiDAR obstacle check, `robot-dry` = decide but never command, `test`).

## Live view and gesture-control settings

- GET `/live/camera` → MJPEG stream (`multipart/x-mixed-replace`), use as `<img src="/api/live/camera">`.
- GET `/live/snapshot` → one JPEG.

GET `/settings` →
```json
{"schema": [{"key": "window_s", "label": "Observation window", "group": "Detection",
             "help": "How long the bridge watches the outputs ...", "type": "float",
             "min": 0.2, "max": 1.5, "step": 0.05, "unit": "s"}],
 "defaults": {"window_s": 0.5}, "values": {"window_s": 0.5},
 "classes": ["Heart", "Turn left 360", "Turn right 360", "Follow me", "none"],
 "mode": "test", "applied": true}
```
`type` is `float`, `int` or `bool`. `applied` = the bridge has picked up the current values.

| Method, path | Body | Answer |
|---|---|---|
| PUT `/settings` | `{values: {key: value, ...}}` (only changed keys) | same as GET (422 if out of range) |
| POST `/settings/reset` | `{key: "window_s"}` or `{}` = all | same as GET |

SSE `live` (10 per second):
```json
{"t": 1795310000.12, "connected": true, "rates": [0.05, 0.62, 0.1, 0.0, 0.3],
 "labels": ["Heart", "Turn left 360", "Turn right 360", "Follow me", "none"],
 "fps": 99, "window_frames": 50, "status": "listening",
 "candidate": 1, "candidate_s": 0.12, "last_trick": null, "busy_s": 0.0, "paused": false,
 "thresholds": {"min_rate": [0.5, 0.5, 0.5, 0.5], "quiet": 0.35, "margin": 0.3, "beat_none": 0.2, "stable_s": 0.2}}
```
`status`: `listening | candidate | waiting-for-calm | busy | paused | no-board`.

SSE `log`: `{"t": ..., "source": "bridge", "kind": "command" | "won" | "ready" | "held" | "turn" | "info" | "error", "line": "..."}`.

## Simulation (RViz) page

The local simulator runs the real gesture bridge against a fake FPGA (replayed model outputs and camera
frames of recorded clips) and the Go2 animator (what RViz shows), on an isolated ROS domain (88), so
everything can be tried without the board or the dog. The same 3D view also follows the board's test
mode (domain 42, the animator on the laptop that the board's bridge commands).

| Method, path | Body | Answer |
|---|---|---|
| GET `/sim/state` | | `{running, person, components: {fake_board, bridge, animator, rviz, tap}, domain, view_source: "sim" \| "board-test", people: ["none", "heart", "left", "right"]}` |
| POST `/sim/start`, `/sim/stop` | `{}` | sim state |
| POST `/sim/person` | `{gesture: "none" \| "heart" \| "left" \| "right"}` | sim state (the virtual person now performs it) |
| POST `/sim/rviz` | `{action: "start" \| "stop", source: "sim" \| "board-test"}` | sim state (desktop RViz window on the laptop) |
| POST `/sim/view` | `{source: "sim" \| "board-test"}` | sim state (which robot the browser 3D view follows) |
| POST `/sim/reset` | `{}` | sim state (simulated robot back to the origin) |
| GET `/sim/model` | | `{root: "base", joints: [{name, type, parent, child, xyz: [3], rpy: [3], axis: [3], lower, upper}], links: [name]}` |

SSE `pose` (20 per second while a view source is running):
```json
{"t": 1795310000.1, "source": "sim", "x": 0.0, "y": 0.0, "z": 0.32, "qx": 0, "qy": 0, "qz": 0.38, "qw": 0.92,
 "yaw": 0.78, "joints": {"FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5},
 "action": "Turn left", "cmd": {"vx": 0.0, "vy": 0.0, "wz": 1.5}}
```
Frames: `world` (floor, z up) to `base` (robot body). Joint angles in rad, revolute about `axis`.

## RViz (real RViz in a container, shown in the Simulation page)

The `durance-rviz` Docker image runs RViz 2 (Jazzy, Go2 meshes, `go2.rviz` from `ros2/rviz/` or `go2_viz/`) on a virtual screen with a VNC
server bound to 127.0.0.1. The browser shows it with the vendored noVNC client
(`static/vendor/novnc-1.5.0/core/rfb.js`) over the dashboard's own authenticated WebSocket, so RViz is never
reachable without a dashboard login.

| Method, path | Body | Answer |
|---|---|---|
| GET `/rviz/state` | | see below |
| POST `/rviz/start` | `{source: "sim" \| "board-test"}` | rviz state (restarts the container if it shows the other source) |
| POST `/rviz/stop` | `{}` | rviz state |
| POST `/rviz/build` | `{}` | `{task_id}` (docker build of the image, several minutes the first time) |
| WS `/rviz/vnc` | | RFB (VNC) byte stream, binary frames. Cookie auth + same-origin check |

```json
{"docker": {"ok": true, "detail": null}, "image": {"name": "durance-rviz:1", "present": true, "build_task": null},
 "container": {"running": true, "source": "sim", "started": 1795310000.0, "error": null},
 "vnc": {"path": "/api/rviz/vnc", "password": "k3...", "width": 1600, "height": 900}}
```
`vnc` is null while the container is not running. Connect with
`new RFB(el, (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + vnc.path,
{credentials: {password: vnc.password}})`, `rfb.scaleViewport = true`, `rfb.background = 'transparent'`.
`docker.ok` false (detail says why, e.g. the user is not in the docker group) or `image.present` false: show
what to do instead of the viewer.

## Health

GET `/health` (no auth) → `{ok, version, uptime_s}`.

## Manual robot control

| Method, path | Body | Answer |
|---|---|---|
| GET `/robot/state` | | `{manual, link, teleop, limits: {vx, vy, wz}, deadman_s, last_cmd}` |
| GET `/robot/commands` | | `{commands: [{name, api_id, label, group, confirm, note}]}` |
| POST `/robot/manual` | `{enabled}` (enabling on the real dog: confirm) | robot state. Manual mode pauses gesture control |
| POST `/robot/command` | `{name}` (commands with `confirm: true` need confirm) | `{ok, sent}` |
| POST `/robot/move` | `{vx, vy, wz}` m/s, m/s, rad/s | `{ok, clamped: {vx, vy, wz}}` |
| POST `/robot/stop` | `{}` | `{ok}` |

Move is a dead-man control: the UI sends `/robot/move` every 100 ms **while a key or the joystick is held**.
If no move arrives for `deadman_s` (0.4 s) the robot stops by itself. On release the UI sends `/robot/stop`.
Speeds are clamped to `limits` on the server.

SSE `robot`: robot state when it changes.

## Recording and labelling

| Method, path | Body | Answer |
|---|---|---|
| GET `/record/options` | | `{classes: [{id, name, how, styles}], defaults: {minutes, ready_s, switch_s, styles, no_action}, next_session}` |
| GET `/record/sessions` | | `{sessions: [{session, person, raw_takes, labeled_takes, clips: {"heart": 120}, total_clips}]}` |
| POST `/record/start` | `{session, person, classes: [1,2,3], minutes, styles: "normal,fast,large", no_action, shuffle}` (confirm: stops board inference, the camera can only stream to one program) | record state |
| GET `/record/state` | | see below |
| POST `/record/control` | `{action: "go" \| "pause" \| "resume" \| "redo" \| "skip" \| "stop"}` | record state |
| GET `/record/preview` | | JPEG of the live events |
| POST `/record/cut` | `{session, best: 0, recut: false}` | task (cuts the labelled takes into 1 s clips) |

Record state (also SSE `record`, 4 per second while recording):
```json
{"running": true, "connected": true, "phase": "go", "step": 3, "steps": 14,
 "gesture": {"id": 1, "name": "heart", "how": "both hands meet on top of your head ..."},
 "style": "fast", "hint": "...", "seconds_left": 12.4, "events_per_s": 152000,
 "saved": [{"file": "gesture_01-heart-normal-take01.bin", "events": 512000, "seconds": 20.0}],
 "plan": [{"cls": 1, "name": "heart", "style": "normal", "seconds": 20.0}], "session": "34", "error": null}
```
`phase`: `idle | connecting | waiting | ready | go | paused | done | error`
(`waiting` = connected, waiting for the operator to press Go; `ready` = countdown before a take).

## Dataset

| Method, path | Body | Answer |
|---|---|---|
| GET `/dataset/sources` | | `{sources: [{id, name, kind: "folder" \| "zip", writable, sessions: ["20", ...], classes: {"1": "heart"}}]}` |
| GET `/dataset/summary?source=` | | `{classes: {"1": "heart"}, sessions: [{session, counts: {"1": 64}, total}]}` |
| GET `/dataset/clips?source=&session=&cls=&offset=0&limit=100` | | `{total, clips: [{path, session, cls, label, style, take, bytes}]}` |
| GET `/dataset/clip?source=&path=&size=80` | | `{meta: {path, session, cls, label, events, duration_s}, size, frames, fps, data}` |
| POST `/dataset/relabel` | `{source, path, cls}` (confirm) | `{path}` (new path) |
| POST `/dataset/reject` | `{source, path}` (confirm) | `{moved_to}` |

| GET `/dataset/spikes?source=&path=&model=` | | the clip's network spikes, see below (`model` optional: default = the board model for that session's gestures) |
| GET `/dataset/spike-models` | | `{models: [{id, name, arch, classes, bits, kind}]}` (models the FPGA simulator can replay: the team's networks and every run of our pipeline) |

Spikes (the clip replayed through the FPGA integer model `spleat_sim.py` (`training/optimised_snn/` or `go2_viz/`), which picks the same gesture as the
FPGA on 98 % of clips; 60 blank frames first because the FPGA never resets; about 1 s per clip, then cached):
```json
{"model": {"id": "a1b2c3d4e5f6", "name": "own_gestures_nb40_q8h_weights", "arch": "nb40", "classes": "own", "md5": "056ee113", "bits": "8h"},
 "labels": ["Heart", "Turn left 360", "Turn right 360", "Follow me", "none"], "frames": 100, "fps": 100, "input_size": 40,
 "outputs": [[0, 1, 0, 0, 0], "... one row of 5 spikes (0/1) per 10 ms frame"],
 "layer_spikes": [[9, 17, 0, 33, 1], "... spikes per layer per frame"], "neurons": [5776, 5776, 648, 256, 5],
 "totals": [20, 88, 18, 0, 18], "rates": [0.2, 0.88, 0.18, 0.0, 0.18], "predicted": 1, "true": 1,
 "events_per_frame": [30, "... camera events per frame"], "settle_frames": 60}
```
Frame `i` of `outputs` / `layer_spikes` / `events_per_frame` = frame `i` of `/dataset/clip` (same 10 ms bins).

`data` = base64 of `uint8[frames][2][size][size]`: event counts per 10 ms frame, channel 0 = OFF, 1 = ON,
clipped at 255. Play at `fps` (100) or slower.

## Training (optional GPU machine over SSH, off by default)

No machine configured: `GET /train/hosts` → `{hosts: []}`, the other training calls → 409 "no training machine configured". Replies never contain host names, user names, paths or SSH commands.


| Method, path | Body | Answer |
|---|---|---|
| GET `/train/hosts` | | `{hosts: [{id, label, ok, detail}]}` |
| GET `/train/options` | | `{pipeline: "training.snn", presets: [{id: "training/configs/<file>.yaml", label, name, classes, sessions, data_root, args: {key: value}}], flags: [{key: "train.epochs" ..., type, min, max, choices, default, help}], prepare_datasets: [{id, label}], datasets: [], archs: []}`. Presets are the YAML configs (default recipe first), `args` their values for the form's keys |
| POST `/train/submit` | `{host, name, preset: <config id>, args: {key: value}, dry_run}` (confirm unless dry_run) | `{job_id, command, dry_run, output}`. Every value that differs from the config becomes `--set key=value`, validated by the pipeline's loader (422 otherwise). Dry run: `training.snn check` on the laptop (its report is in `command`) and a check of the machine |
| GET `/train/jobs?host=` | | `{jobs: [{id, name, state, time, node, reason}]}` |
| POST `/train/cancel` | `{host, job_id}` (confirm) | `{ok}` |
| GET `/train/log?host=&job=&lines=200` | | `{lines: []}` |
| GET `/train/runs?host=` | | `{runs: [{name, config, val, test, kind, finished, deployable, classes}]}` (from `runs/*/metrics.json` on the machine; fixed-point accuracy when there is one) |
| POST `/train/fetch` | `{host, run}` | task (copies `runs/<run>/` to the laptop's runs/; it then shows up in `/board/models`) |

### Training machines (added in the dashboard)

A GPU server reachable with an SSH **key** (no passwords are stored); the dashboard starts each run in the background
there (`setsid nohup`) and tracks its process (`<workdir>/.dd_jobs`). Stored in the private `clusters.toml` (0600).
Lists show only `label` and reachability.

| Method, path | Body | Answer |
|---|---|---|
| GET `/train/hosts` | | `{hosts: [{id, label, ok, detail}]}` (`detail` = "reachable, 290 ms" or a short reason) |
| GET `/train/machines/{id}` | | `{id, label, host, user, port, identity_file, ssh_host, workdir, python, uv, max_submits_per_hour}` (edit form only) |
| POST `/train/machines` | the same fields (`id` new or existing) | the saved machine (422 names the invalid field) |
| POST `/train/machines/{id}/delete` | `{}` (confirm) | `{ok}` (removes it from the dashboard only) |
| GET `/train/keys` | | `{keys: [{ref, name, source: "~/.ssh" \| "dashboard", public}]}`; `ref` goes into `identity_file`; private halves are never sent |
| POST `/train/keys` | `{name}` | a new ed25519 key `{ref, name, source, public}`: paste `public` into the cloud provider's SSH keys |
| GET `/train/machines/{id}/server-id` | | `{keys: [{type, fingerprint, trusted}]}` = the server's identity ("SHA256:..."), shown as "Server ID" |
| POST `/train/machines/{id}/server-id` | `{fingerprint}` | `{trusted}` (409 if the server now shows a different ID) |
| POST `/train/machines/{id}/test` | `{}` | `{ok, steps: [{name, ok, detail}]}`: SSH login, work folder, GPU, uv, code, PyTorch + riverops, data sessions |
| POST `/train/machines/{id}/prepare` | `{datasets: ["organisers", "own"]}` (confirm) | task: uv (if missing), the repository code + riverops wheel, `uv sync --group train`, the session folders 01-14 / 20-30 of data/motion_buddy_dataset |

Typical first use: generate key → paste public key at the provider → add machine (host, user, key, folder) → Server ID →
Trust → Test → Prepare → Dry run → Start.

## Tasks and audit

- GET `/tasks` → `{tasks: [task]}`, GET `/tasks/{id}` → task:
  `{id, kind, label, state: "running" | "done" | "failed", started, finished, log: ["..."], result}`. SSE `task` = task.
- GET `/audit?limit=200` → `{entries: [{t, user, action, params, result}]}` (newest first).

SSE `alert`: `{"level": "info" | "warn" | "danger", "text": "..."}`.
