#!/usr/bin/env python3
"""Live bridge: Kria FPGA gesture model -> Go2 sport commands (RViz / robot_sim).

Connects to the SPLEAT server on the board (the camera must be feeding it, e.g.
`despot_send_and_forget --to_display 0` on the Kria), turns the model's 5 binary
outputs per 10 ms step into firing rates over a sliding window, shows them as bars
next to the 3D dog in RViz, and publishes a /api/sport/request when one output
clearly and stably wins.

    source ~/go2-hackathon/laptop_ws/env.sh
    python3 ~/go2-hackathon/go2_viz/live_gesture_bridge.py [--host 192.168.77.1] [--port 3333]

--classes own (default) = our own gestures (go2_viz/gestures.json, model trained on
sessions 20+): 0 heart -> finger heart, 1 turn-left-360 -> the robot turns a full
circle to its left, 2 turn-right-360 -> full circle to its right, 3 unused
(namaskaram, dropped), 4 no gesture.
--classes dataset = the organisers' Motion Buddy classes (sit / lay down / hello /
dance), for the organisers' weights.
Turns are closed loop on the IMU yaw (/sportmodestate), like `go2_cmd.py turn`:
Move (1008) is re-sent at 20 Hz until the measured heading has changed by 360 deg,
then StopMove (1003). Without yaw data (simulator) the turn runs open loop on time.
"""
import argparse
import json
import math
import os
import socket
import struct
import threading
import time
from collections import deque

import rclpy
from rclpy.qos import qos_profile_sensor_data
from unitree_api.msg import Request
from visualization_msgs.msg import Marker, MarkerArray

try:                                   # gesture-control settings shared with the dashboard
    import bridge_settings as BS
except ImportError:                    # bridge_settings.py not copied next to this file: fixed rules
    BS = None

N_OUT = 5                # classes; an output map of 5 x H x W is summed per class (class_spikes)
WINDOW = 50              # frames (100 Hz) -> 0.5 s sliding window
WINDOW_S = 0.5           # the window is measured in TIME: the camera client only sends frames that
MIN_FRAMES = 25          # contain events (quiet scene: 10-20 per second instead of 100), so with a
                         # window counted in frames the bars showed firing from seconds ago. Frames
                         # that never came count as silence: rate = spikes / max(frames, MIN_FRAMES)
MIN_RATE = 0.50          # winner must fire in >= 50 % of the window
MIN_MARGIN = 0.20        # ...and beat the runner-up GESTURE by 20 points
                         # (the "no gesture" output is not part of this comparison:
                         #  it keeps firing in the background even during gestures)
STABLE_S = 0.40          # ...continuously for this long
COOLDOWN_S = 0.5         # ignore the camera this long after a trick finishes (was 2.0 s: the camera is
                         # static and does not see the robot, so this only skips the last motion)
TURN_SETTLE_S = 0.5      # after the StopMove of a turn, before the cooldown starts (was 1.0 s)
QUIET_RATE = 0.20        # the gesture that just ran re-arms once its output drops below this
                         # (breaks loops where the robot's own motion looks like the same gesture)...
REARM_MAX_S = 1.5        # ...or at the latest this long after the robot became free. Every other gesture
                         # is armed as soon as the robot is free. (Until 2026-11-22 all gestures waited
                         # for calm: 4-14 s of dead time after each trick in the logs.)
# Busy time per api_id without a real robot (no --require-clear = RViz animator, whose animations are
# shorter than the Go2's): go2_animator.py plays Finger heart and Hello for 3.5 s
TEST_BUSY_S = {1036: 3.5, 1016: 3.5}

def class_spikes(p, n_out):
    """One output frame's payload (float32 per output neuron, classes x H x W, as spleat-server sends it) ->
    1.0/0.0 per class: did any position of the class spike. A 1x1 output map is one value per class; a
    spatial head (training/snn model.head, outputs.height/width > 1 in the server config) has H*W per class."""
    n = len(p) // 4
    if n == 0 or len(p) % 4 or n % n_out:
        return None
    v = struct.unpack("<%df" % n, p)
    per = n // n_out
    return [1.0 if any(x > 0 for x in v[i * per:(i + 1) * per]) else 0.0 for i in range(n_out)]

# output index -> (action, busy seconds, follow-up api_id or None, label)
# action = sport api_id, or ("turn", degrees) = closed-loop turn (+ = left / anticlockwise)
ACTION_SETS = {
    # our own gestures (go2_viz/gestures.json; training label = id - 1):
    # gesture_01 heart, 02 turn-left-360, 03 turn-right-360, 04 namaskaram (unused), 05 no-action
    "own": {
        0: (1036, 5.0, None, "Heart"),                    # finger heart with the front paws
        1: (("turn", 360.0), 0.0, None, "Turn left 360"),  # busy time = until the turn ends
        2: (("turn", -360.0), 0.0, None, "Turn right 360"),
        3: (("walk", 1.0), 0.0, None, "Follow me"),       # walk on the spot -> the dog walks 1 m
                                                          # forward (closed loop on odometry), like the
                                                          # turns; --follow lidar = LiDAR person following
        4: None,                                          # no gesture
    },
    # class order of the Motion Buddy dataset / riverops notebook (organisers' weights):
    # gesture_01 Sit, 02 Lay down, 03 Hello, 04 Dance, 05 No action
    "dataset": {
        0: (1009, 4.0, 1004, "Sit"),        # Sit, then Stand up
        1: (1005, 5.0, 1004, "Lay down"),   # StandDown (lie down), then Stand up
        2: (1016, 4.0, None, "Hello"),      # wave a front paw
        3: (1022, 10.0, None, "Dance"),     # Dance 1
        4: None,                            # No action
    },
}
ACTIONS = ACTION_SETS["own"]

# Per class set, a tuned rule that replaces MIN_RATE / MIN_MARGIN / STABLE_S / QUIET_RATE:
# TEST_RULES in laptop / RViz test mode (no --require-clear), ROBOT_RULES on the real robot
# (--require-clear). Class sets without an entry use the strict rule above.
# beat_none = the winner must also fire at least this much more often than the "none" output.
TEST_RULES = {
    # weights 056ee113 (run r6b-reg3, own gestures, not silent: o4 fires on no-action). Tuned on
    # simulated per-frame outputs of persons 28-30 + organisers' 04,13 with this time window: per 1 s
    # clip from a quiet start 97.7 % right, 0 % wrong, false triggers 1.0 % on own no-action and
    # 0.0 % on foreign motion, reaction median 0.37 s. quiet 0.35: gesture bars sit near 0.2 during
    # other motion with this model, so 0.20 would rarely re-arm.
    # min_frames 40: on near-empty frames this model's Heart output fires by itself (0.5-0.6 per
    # frame, bias-driven), and a quiet scene still sends 6-30 such frames/s. Dividing by at least
    # 40 frames per 0.5 s means a quiet scene would need 80 frames/s before Heart reaches 0.5; real
    # gestures arrive at 78-100 frames/s (measured live 2026-11-22), so they are not affected.
    "own": {"min_rate": {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50}, "margin": 0.30, "stable_s": 0.20,
            "quiet": 0.35, "beat_none": 0.20, "min_frames": 40, "follow_hold_s": 0.50},
    # weights e9ec792c (organisers' classes), tuned on the FPGA's outputs for unseen people
    # (sessions 04,05,06,13): Sit and Hello rest at ~0.3 there while Lay down and Dance rest near 0,
    # so every gesture gets its own trigger line. Same false-trigger rate as before (4.6 % of 1 s
    # no-action clips), correct triggers 47 % -> 55 %. The winner must also beat the "none" output
    # (Hello kept winning at ~0.6 while "none" was at 0.6-0.9; true gestures: "none" 0.1-0.2).
    "dataset": {"min_rate": {0: 0.60, 1: 0.50, 2: 0.60, 3: 0.30}, "margin": 0.15, "stable_s": 0.30,
                "quiet": 0.45, "beat_none": 0.0},
}
ROBOT_RULES = {
    # same rule, held a little longer before the real robot moves (false on own no-action stays
    # 1.0 %, reaction 0.47 s); the strict rule's re-arm line 0.20 is too low for this model
    "own": dict(TEST_RULES["own"], stable_s=0.30),
}
RULE, ROBOT_RULE = TEST_RULES.get("own"), ROBOT_RULES.get("own")

MOVE_ID, STOP_ID = 1008, 1003
TURN_MAX_RATE = 1.5      # rad/s, --turn-rate (was 0.8 = 8.4 s per full circle; 1.5 -> ~4.6 s)
TURN_MIN_RATE = 0.40     # rad/s - the real Go2 does not rotate at 0.25 (turns hung 3 deg short until the
                         # 30 s timeout, 2026-11-22)
TURN_KP = 3.0            # rad/s of turn rate per rad of remaining angle (slows down in the last ~30 deg)
TURN_TOL_DEG = 6.0       # stop when this close to the target (was 3)
YAW_STALE_S = 0.5        # closed loop: stop the turn if /sportmodestate goes quiet this long
WALK_VX = 0.5            # m/s, "Follow me" walk (--walk-speed; was 0.3, faster on request 2026-11-22)
WALK_MIN_VX = 0.25       # m/s, slowest speed near the end of the walk (the real Go2 stood still at
                         # 0.10-0.13 m/s: walk stopped at 0.87 of 1.0 m until the timeout, 2026-11-22)
WALK_KP = 1.0            # m/s of speed per m of remaining distance (slows down in the last 0.3 m)
WALK_TOL_M = 0.08        # stop when this close to the target distance
COLORS = [(0.95, 0.45, 0.2), (0.3, 0.75, 1.0), (0.85, 0.35, 0.9), (0.35, 0.9, 0.45), (0.6, 0.6, 0.6)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def recv_exact(s, n):
    """Read exactly n bytes. recv(MSG_WAITALL) can return less when the timeout hits."""
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise OSError("server closed connection")
        buf += chunk
    return bytes(buf)


# Camera shake guard: when the event camera itself moves (bumped tripod, pulled cable, mounted on the
# robot) the whole picture fires and the network reports gestures that nobody made (2026-11-22, real Go2:
# Turn left every 8 s with a shaking camera; each Heart jolted the camera through its cable). Measured
# per 10 ms input frame, share of the 40x40 cells with events: people gesturing <= 0.26 live and <= 0.21
# in 600 training clips; shaking camera 0.46 median (77 % of frames >= 0.35).
SHAKE_COVER = 0.30       # a frame where at least this share of the picture fires...
SHAKE_FRAMES = 3         # ...this many times within SHAKE_WIN_S = the camera is moving
SHAKE_WIN_S = 0.5
SHAKE_HOLD_S = 1.0       # gestures stay ignored until this long after the last such burst


class ShakeGuard:
    """Watches the camera's input frames on the loopback link camera client -> SPLEAT server (passive
    capture like snn_input_viewer.py; needs root) and tells when the whole picture is firing."""

    def __init__(self, server_port=3333):
        self.server_port, self.hits, self.tripped_at, self.frames = server_port, deque(), -1e9, 0
        self.lock, self.error = threading.Lock(), ""
        try:
            import numpy
            self.np = numpy
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            self.sock.bind(("lo", 0))
        except (ImportError, OSError, AttributeError) as e:
            self.error = str(e)
            return
        threading.Thread(target=self.capture, daemon=True).start()

    @property
    def ok(self):
        return not self.error

    def moving(self, now):
        return now - self.tripped_at < SHAKE_HOLD_S

    def on_input(self, body):
        k = body.find(b"\x67payload")
        if k < 0 or k + 11 > len(body):
            return
        h = body[k + 8]
        if h == 0x59:
            n, off = struct.unpack(">H", body[k + 9:k + 11])[0], k + 11
        elif h == 0x5a:
            n, off = struct.unpack(">I", body[k + 9:k + 13])[0], k + 13
        else:
            return                                      # outputs / small messages
        side = int(round((n // 8) ** 0.5))
        if side < 8 or side * side * 8 != n or off + n > len(body):
            return
        a = self.np.frombuffer(body, "<f4", count=2 * side * side, offset=off).reshape(2, side * side)
        cover = self.np.count_nonzero(a[0] + a[1]) / (side * side)
        now = time.monotonic()
        self.frames += 1
        if cover >= SHAKE_COVER:
            with self.lock:
                self.hits.append(now)
                while self.hits and now - self.hits[0] > SHAKE_WIN_S:
                    self.hits.popleft()
                if len(self.hits) >= SHAKE_FRAMES:
                    self.tripped_at = now

    def capture(self):
        streams = {}                                    # client port -> reassembly buffer
        while True:
            try:
                pkt, addr = self.sock.recvfrom(70000)
            except OSError:
                time.sleep(0.5)
                continue
            if addr[2] == socket.PACKET_OUTGOING:       # loopback shows every packet twice
                continue
            ip = pkt[14:]
            if len(ip) < 20 or ip[9] != 6:
                continue
            tcp = ip[(ip[0] & 15) * 4:]
            sp, dp = struct.unpack(">HH", tcp[:4])
            if dp != self.server_port:
                continue
            data = tcp[(tcp[12] >> 4) * 4:]
            if not data:
                continue
            buf = streams.setdefault(sp, bytearray())
            buf += data
            while len(buf) >= 4:
                n = struct.unpack(">I", buf[:4])[0]
                if n == 0 or n > 200_000 or (len(buf) > 4 and buf[4] & 0xE0 != 0xA0):
                    j = buf.find(b"\x62id", 5)              # lost sync: skip to the next CBOR map
                    if j < 5:
                        buf.clear()
                        break
                    del buf[:j - 5]
                    continue
                if len(buf) < 4 + n:
                    break
                body = bytes(buf[4:4 + n])
                del buf[:4 + n]
                self.on_input(body)
            if len(streams) > 64:                          # old client connections
                streams = {sp: streams[sp]}


class Bridge:
    def __init__(self, host, port, require_clear=False, dry_run=False, viz=True, classes="own",
                 settings_path=None, status_path=None, turn_rate=None, shake_guard=True):
        self.host, self.port = host, port
        self.shake, self.shake_logged = None, False
        if shake_guard:
            g = ShakeGuard(port)
            if g.ok:
                self.shake = g
                log(f"camera shake guard on: ignore gestures while >= {SHAKE_COVER:.0%} of the picture fires "
                    f"in {SHAKE_FRAMES} frames within {SHAKE_WIN_S} s (+{SHAKE_HOLD_S} s)")
            else:
                log(f"camera shake guard unavailable ({g.error}) - gestures are not checked for camera motion")
        self.classes, self.settings_path, self.status_path = classes, settings_path, status_path
        self.cli_turn_rate = turn_rate
        self.rule, self.paused, self.settings, self.settings_errors = None, False, {}, {}
        self.settings_stamp, self.settings_version = None, 0
        self.fps, self._fps_mark = 0.0, (time.monotonic(), 0)
        self.require_clear, self.dry_run = require_clear, dry_run
        self.hist = deque(maxlen=8 * WINDOW)       # (arrival time, spikes) of the last frames
        self.window_from = time.monotonic()        # the window only holds frames newer than this
        self.lock = threading.Lock()
        self.connected = False
        self.frames = 0
        rclpy.init()
        self.node = rclpy.create_node("live_gesture_bridge")
        self.req_pub = self.node.create_publisher(Request, "/api/sport/request", 10)
        self.bar_pub = self.node.create_publisher(MarkerArray, "/go2_viz/scores", 10)
        self.busy_until = 0.0
        self.follow_up = None
        self.candidate, self.cand_since = None, 0.0
        self.last_trick = None   # output of the last trick until it is re-armed (calm or REARM_MAX_S)
        self.status = "waiting for board..."
        # Real robot safety: only start a trick while obstacle_detection reports CLEAR
        self.clearance, self.clear_names, self.last_hold_log = None, {}, 0.0
        if require_clear:
            from msgs.msg import Clearance
            self.CLEAR = Clearance.CLEAR
            self.clear_names = {Clearance.CLEAR: "CLEAR", Clearance.OBSTACLE: "OBSTACLE",
                                Clearance.UNKNOWN: "UNKNOWN"}
            self.node.create_subscription(Clearance, "/durance/clearance", self.on_clearance, 1)
        # heading for closed-loop turns (the real Go2 publishes it; the simulator may not)
        self.yaw, self.yaw_time, self.turning = None, 0.0, False
        self.pos, self.walking = None, False                 # odometry x, y for the "Follow me" walk
        try:
            from unitree_go.msg import SportModeState
            self.node.create_subscription(SportModeState, "/sportmodestate", self.on_state,
                                          qos_profile_sensor_data)
        except ImportError:
            log("unitree_go messages not found: turns will run open loop")
        # LiDAR height map + pose for follow mode (only subscribed when a follow action exists)
        self.lidar_map, self.lidar_pose, self.map_time, self.following = None, None, 0.0, False
        if any(a and isinstance(a[0], tuple) and a[0][0] == "follow" for a in ACTIONS.values()):
            try:
                from geometry_msgs.msg import PoseStamped
                from unitree_go.msg import HeightMap
                self.node.create_subscription(HeightMap, "/utlidar/height_map_array", self.on_map,
                                              qos_profile_sensor_data)
                self.node.create_subscription(PoseStamped, "/utlidar/robot_pose", self.on_pose,
                                              qos_profile_sensor_data)
            except ImportError:
                log("HeightMap / PoseStamped messages not found: follow mode unavailable")
        self.node.create_timer(0.05, self.decide)
        if viz:
            self.node.create_timer(0.1, self.draw)
        self.node.create_timer(2.0, self.report)
        self.reload_settings()
        self.node.create_timer(0.2, self.reload_settings)    # pause / E-STOP must act within 0.2 s
        if status_path:
            self.node.create_timer(0.1, self.write_status)
        log(f"mode: {'DRY-RUN (no commands sent)' if dry_run else 'LIVE (commands sent)'}, "
            f"obstacle check {'ON' if require_clear else 'OFF'}")

    # ---------- live settings (dashboard) ----------
    def reload_settings(self):
        """Re-read the settings file when it changed; invalid values fall back to the tuned defaults."""
        global WINDOW_S, MIN_FRAMES, COOLDOWN_S, REARM_MAX_S, TURN_MAX_RATE
        if BS is None:
            return
        try:
            st = os.stat(self.settings_path) if self.settings_path else None
            stamp = (st.st_mtime_ns, st.st_size) if st else None
        except OSError:
            stamp = None
        if stamp == self.settings_stamp and self.settings:
            return
        base = {"turn_rate": self.cli_turn_rate} if self.cli_turn_rate else None
        values, errors = BS.load(self.settings_path, self.classes, robot=self.require_clear, base=base)
        old = self.settings
        WINDOW_S, MIN_FRAMES = values["window_s"], values["min_frames"]
        COOLDOWN_S, REARM_MAX_S, TURN_MAX_RATE = values["cooldown_s"], values["rearm_max_s"], values["turn_rate"]
        self.rule = {"min_rate": {i: values[f"min_rate_o{i}"] for i in range(4)}, "margin": values["margin"],
                     "stable_s": values["stable_s"], "quiet": values["quiet"], "beat_none": values["beat_none"],
                     "min_frames": values["min_frames"], "follow_hold_s": values.get("follow_hold_s", 0.0)}
        self.paused = bool(values["paused"])
        self.settings, self.settings_errors, self.settings_stamp = values, errors, stamp
        self.settings_version += 1
        changed = {k: v for k, v in values.items() if old.get(k) != v}
        if old and changed:
            log("settings changed: " + ", ".join(f"{k}={v}" for k, v in sorted(changed.items())))
        if errors:
            log(f"settings file problems (defaults used for these): {errors}")

    def write_status(self):
        """Atomic JSON snapshot for the dashboard (10 per second)."""
        now = time.monotonic()
        t_mark, f_mark = self._fps_mark
        if now - t_mark >= 1.0:
            self.fps, self._fps_mark = (self.frames - f_mark) / (now - t_mark), (now, self.frames)
        r, n = self.window()
        if not self.connected:
            status = "no-board"
        elif self.paused:
            status = "paused"
        elif now < self.busy_until:
            status = "busy"
        elif self.status == "waiting-for-calm":
            status = "waiting-for-calm"
        elif self.candidate is not None:
            status = "candidate"
        else:
            status = "listening"
        rule = self.rule or {}
        busy = self.busy_until - now
        data = {
            "t": time.time(), "connected": self.connected, "rates": [round(x, 3) for x in r],
            "labels": [ACTIONS[i][3] if ACTIONS.get(i) else "none" for i in range(N_OUT)],
            "fps": round(self.fps, 1), "window_frames": n, "status": status, "detail": self.status,
            "candidate": self.candidate, "candidate_s": round(now - self.cand_since, 2) if self.candidate is not None else 0.0,
            "last_trick": self.last_trick, "busy_s": -1.0 if busy == float("inf") else round(max(0.0, busy), 2),
            "paused": self.paused, "mode": "robot" if self.require_clear else "test", "dry_run": self.dry_run,
            "classes": self.classes, "turning": self.turning, "following": self.following,
            "thresholds": {"min_rate": [rule.get("min_rate", {}).get(i, MIN_RATE) for i in range(4)],
                           "quiet": rule.get("quiet", QUIET_RATE), "margin": rule.get("margin", MIN_MARGIN),
                           "beat_none": rule.get("beat_none", 0.0), "stable_s": rule.get("stable_s", STABLE_S)},
            "settings_version": self.settings_version, "settings_errors": self.settings_errors,
            "clearance": self.clear_names.get(self.clearance) if self.require_clear else None,
        }
        try:
            tmp = self.status_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.status_path)
        except OSError:
            pass

    def on_clearance(self, msg):
        if msg.state != self.clearance:
            log(f"obstacle check: {self.clear_names.get(msg.state, msg.state)}"
                + (f" ({msg.range:.2f} m)" if msg.range == msg.range else ""))
        self.clearance = msg.state

    def on_state(self, msg):
        self.yaw, self.yaw_time = float(msg.imu_state.rpy[2]), time.monotonic()
        self.pos = (float(msg.position[0]), float(msg.position[1]))

    def on_map(self, msg):
        self.lidar_map = (msg.data, msg.width, msg.height, msg.resolution, tuple(msg.origin))
        self.map_time = time.monotonic()

    def on_pose(self, msg):
        q, p = msg.pose.orientation, msg.pose.position
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.lidar_pose = (p.x, p.y, yaw)

    # ---------- board connection ----------
    def reader(self):
        while rclpy.ok():
            s = None
            try:
                s = socket.create_connection((self.host, self.port), timeout=3)
                s.settimeout(3)
                self.connected = True
                log(f"connected to FPGA model server {self.host}:{self.port}")
                while rclpy.ok():
                    head = recv_exact(s, 4)
                    n = struct.unpack(">I", head)[0]
                    d = recv_exact(s, n)
                    # CBOR map {"payload": <byte string>, ...}: "payload" key, then the
                    # byte-string header (0x40+len for len < 24, 0x58 <len> up to 255)
                    k = d.find(b"\x67payload")
                    if k < 0 or k + 10 > len(d):
                        continue
                    h = d[k + 8]
                    if 0x40 <= h < 0x58:
                        L, off = h - 0x40, k + 9
                    elif h == 0x58:
                        L, off = d[k + 9], k + 10
                    elif h == 0x59 and k + 11 <= len(d):                 # >= 256 bytes: big spatial maps
                        L, off = struct.unpack(">H", d[k + 9:k + 11])[0], k + 11
                    else:
                        continue
                    spikes = class_spikes(d[off:off + L], N_OUT) if off + L <= len(d) else None
                    if spikes is None:
                        continue
                    with self.lock:
                        self.hist.append((time.monotonic(), spikes))
                        self.frames += 1
            except OSError as e:
                self.connected = False
                log(f"board connection lost ({e}); retrying in 1 s")
                time.sleep(1)
            finally:
                if s is not None:
                    s.close()

    # ---------- decision ----------
    def publish(self, api_id, parameter=""):
        m = Request()
        m.header.identity.api_id = api_id
        m.header.identity.id = int(time.time() * 1000) % (1 << 62)
        m.parameter = parameter
        self.req_pub.publish(m)

    def send(self, api_id, label):
        if self.dry_run:
            log(f"DRY-RUN would send -> {label} (api_id={api_id})")
            return
        self.publish(api_id)
        log(f"ROBOT COMMAND -> {label} (api_id={api_id})")

    def window(self):
        """Firing rate of every output over the last WINDOW_S seconds, and the frames in it."""
        now = time.monotonic()
        with self.lock:
            while self.hist and now - self.hist[0][0] > WINDOW_S:
                self.hist.popleft()
            n = len(self.hist)
            sums = [sum(f[1][i] for f in self.hist) for i in range(N_OUT)]
        return [x / max(n, MIN_FRAMES) for x in sums], n

    def start_follow(self, label):
        """Follow the person in front with the LiDAR (lidar_follow.py). Busy until it ends."""
        if self.lidar_map is None or self.lidar_pose is None or time.monotonic() - self.map_time > 1.0:
            log(f"{label}: no LiDAR height map / robot pose - follow mode not started")
            self.busy_until = time.monotonic() + COOLDOWN_S
            return
        self.following = True
        self.busy_until = float("inf")                     # set again when following ends
        threading.Thread(target=self.follow_worker, args=(label,), daemon=True).start()

    def follow_worker(self, label):
        import lidar_follow as LF
        t0 = time.monotonic()
        f = LF.Follower(t0)
        last_map, last_log, why = 0.0, 0.0, ""
        log(("DRY-RUN " if self.dry_run else "ROBOT COMMAND -> ") + f"{label}: LiDAR follow mode on "
            f"(keep {LF.FOLLOW_DIST:.1f} m, max {LF.MAX_VX} m/s)")
        try:
            while rclpy.ok():
                now = time.monotonic()
                if now - self.map_time > 1.5:
                    why = "LiDAR height map stopped"
                    break
                if self.require_clear and self.clearance != self.CLEAR:
                    why = f"obstacle check {self.clear_names.get(self.clearance, 'no data')}"
                    break
                if self.paused:
                    why = "paused (E-STOP or manual control)"
                    break
                if self.map_time > last_map:                # one tracking step per new height map
                    last_map = self.map_time
                    people = LF.find_people(*self.lidar_map[:4], self.lidar_map[4], self.lidar_pose[:2])
                    vx, wz, done = f.update(people, self.lidar_pose, now)
                    if done:
                        why = f.reason
                        break
                    if now - last_log > 1.0:
                        log(f"follow: {f.state} | {len(people)} people seen | cmd vx {vx:.2f} m/s, yaw {wz:+.2f} rad/s")
                        last_log = now
                    if not self.dry_run:
                        self.publish(MOVE_ID, json.dumps({"x": round(vx, 3), "y": 0.0, "z": round(wz, 3)}))
                time.sleep(0.02)
        finally:
            if not self.dry_run:
                self.publish(STOP_ID)
            log(f"follow mode off: {why or 'stopped'} after {time.monotonic() - t0:.1f} s"
                + ("" if self.dry_run else ", StopMove sent"))
            self.following = False
            self.busy_until = time.monotonic() + COOLDOWN_S

    def start_walk(self, metres, label):
        """Walk straight forward by `metres`. Busy until the walk has finished."""
        if self.dry_run:
            log(f"DRY-RUN would send -> {label} (walk {metres:.1f} m forward at {WALK_VX} m/s, "
                f"Move {MOVE_ID} + StopMove {STOP_ID})")
            self.busy_until = time.monotonic() + metres / WALK_VX + COOLDOWN_S
            return
        self.walking = True
        self.busy_until = float("inf")                     # set again when the walk ends
        threading.Thread(target=self.walk_worker, args=(metres, label), daemon=True).start()

    def walk_worker(self, metres, label):
        t0 = time.monotonic()
        closed = self.pos is not None and t0 - self.yaw_time < YAW_STALE_S
        start, done_m, why = self.pos, 0.0, "done"
        timeout = metres / WALK_MIN_VX + 3.0 if closed else metres / WALK_VX
        log(f"ROBOT COMMAND -> {label}: walk {metres:.1f} m forward, max {WALK_VX} m/s "
            + ("(closed loop on odometry)" if closed else "(open loop on time: no /sportmodestate)"))
        try:
            while rclpy.ok():
                now = time.monotonic()
                if self.paused:
                    why = "paused (E-STOP or manual control)"
                    break
                if self.require_clear and self.clearance != self.CLEAR:
                    why = f"obstacle check {self.clear_names.get(self.clearance, 'no data')}"
                    break
                if closed:
                    if now - self.yaw_time > YAW_STALE_S:
                        why = "odometry stopped"
                        break
                    done_m = math.hypot(self.pos[0] - start[0], self.pos[1] - start[1])
                    remaining = metres - done_m
                    vx = max(WALK_MIN_VX, min(WALK_VX, WALK_KP * remaining))
                else:
                    done_m = WALK_VX * (now - t0)
                    remaining, vx = metres - done_m, WALK_VX
                if remaining <= WALK_TOL_M:
                    break
                if now - t0 > timeout:
                    why = "timeout"
                    break
                self.publish(MOVE_ID, json.dumps({"x": round(vx, 3), "y": 0.0, "z": 0.0}))
                time.sleep(0.05)
        finally:
            self.publish(STOP_ID)
            log(f"walk {why}: {done_m:.2f} of {metres:.1f} m in {time.monotonic() - t0:.1f} s, StopMove sent")
            self.walking = False
            self.busy_until = time.monotonic() + TURN_SETTLE_S + COOLDOWN_S

    def start_turn(self, degrees, label):
        """Turn in place by `degrees` (+ = left). Busy until the turn has finished."""
        if self.dry_run:
            log(f"DRY-RUN would send -> {label} (turn {degrees:+.0f} deg, Move {MOVE_ID} + StopMove {STOP_ID})")
            self.busy_until = time.monotonic() + abs(math.radians(degrees)) / TURN_MAX_RATE + COOLDOWN_S
            return
        self.turning = True
        self.busy_until = float("inf")                     # set again when the turn ends
        threading.Thread(target=self.turn_worker, args=(degrees, label), daemon=True).start()

    def turn_worker(self, degrees, label):
        target, tol = math.radians(degrees), math.radians(TURN_TOL_DEG)
        t0 = time.monotonic()
        closed = self.yaw is not None and t0 - self.yaw_time < YAW_STALE_S
        last, turned, why = self.yaw, 0.0, "done"
        timeout = abs(target) / TURN_MIN_RATE + 5.0 if closed else abs(target) / TURN_MAX_RATE
        log(f"ROBOT COMMAND -> {label}: turn {degrees:+.0f} deg "
            + ("(closed loop on IMU yaw)" if closed else "(open loop on time: no /sportmodestate)"))
        try:
            while rclpy.ok():
                now = time.monotonic()
                if self.paused:
                    why = "paused (E-STOP or manual control)"
                    break
                if closed:
                    if now - self.yaw_time > YAW_STALE_S:
                        why = "yaw data stopped"
                        break
                    y = self.yaw
                    turned += wrap(y - last)
                    last = y
                    remaining = target - turned
                    rate = max(TURN_MIN_RATE, min(TURN_MAX_RATE, TURN_KP * abs(remaining)))
                else:
                    turned = math.copysign(TURN_MAX_RATE * (now - t0), target)
                    remaining, rate = target - turned, TURN_MAX_RATE
                if abs(remaining) <= tol or (not closed and abs(turned) >= abs(target)):
                    break
                if now - t0 > timeout:
                    why = "timeout"
                    break
                self.publish(MOVE_ID, json.dumps({"x": 0.0, "y": 0.0,
                                                  "z": round(math.copysign(rate, remaining), 4)}))
                time.sleep(0.05)
        finally:
            self.publish(STOP_ID)
            log(f"turn {why}: {math.degrees(turned):+.0f} of {degrees:+.0f} deg in "
                f"{time.monotonic() - t0:.1f} s, StopMove sent")
            self.turning = False
            self.busy_until = time.monotonic() + TURN_SETTLE_S + COOLDOWN_S

    def decide(self):
        now = time.monotonic()
        if now < self.busy_until:
            # robot is moving (or settling): what the camera sees now is mostly the
            # robot's own motion, so throw it away and start the window fresh later
            with self.lock:
                self.hist.clear()
            self.window_from = now                 # (= when the robot became free, once it is)
            return
        if self.paused:                              # E-STOP / manual control: watch, never command
            self.candidate, self.follow_up = None, None
            self.status = "paused"
            return
        if self.follow_up:
            api_id, label = self.follow_up
            self.follow_up = None
            self.send(api_id, label)
            self.busy_until = now + 3.0 + COOLDOWN_S
            self.status = label
            return
        if self.shake is not None and self.shake.moving(now):
            # the camera itself is moving: its outputs mean nothing - drop them, refill the window later
            with self.lock:
                self.hist.clear()
            self.window_from = now
            self.candidate = None
            self.status = "camera moving"
            if not self.shake_logged:
                log("CAMERA MOVING (whole picture firing): gestures ignored until it is still")
                self.shake_logged = True
            return
        if self.shake_logged:
            log("camera still again - listening for gestures")
            self.shake_logged = False
        r, _ = self.window()
        if now - self.window_from < WINDOW_S:        # window not filled since the last trick yet
            return
        gestures = [i for i in range(N_OUT) if ACTIONS.get(i) is not None]
        rule = self.rule or (ROBOT_RULE if self.require_clear else RULE)
        if rule:
            thr = {i: rule["min_rate"].get(i, MIN_RATE) for i in gestures}
            quiet = {i: min(thr[i], rule["quiet"]) for i in gestures}
            need_margin, stable_s = rule["margin"], rule["stable_s"]
        else:
            thr = {i: MIN_RATE for i in gestures}
            quiet = {i: QUIET_RATE for i in gestures}
            need_margin, stable_s = MIN_MARGIN, STABLE_S
        if self.last_trick is not None:              # only the gesture that just ran must calm down
            j = self.last_trick
            calm = j not in quiet or r[j] < quiet[j]
            if calm or now - self.window_from >= REARM_MAX_S:
                self.last_trick = None
                log("calm - ready for the next gesture" if calm else
                    f"re-armed after {REARM_MAX_S:.1f} s - ready for the next gesture")
        order = sorted(gestures, key=lambda i: r[i], reverse=True)
        top = order[0]
        second = order[1] if len(order) > 1 else None
        margin = r[top] - (r[second] if second is not None else 0.0)
        winner = top if (r[top] >= thr[top] and margin >= need_margin) else None
        if (winner is not None and rule and rule.get("beat_none") is not None
                and r[top] < r[N_OUT - 1] + rule["beat_none"]):
            winner = None                            # the gesture must also beat the "none" bar
        if winner is not None and winner == self.last_trick:
            self.candidate = None                    # same trick again: wait until it is re-armed
            self.status = "waiting-for-calm"
            return
        if winner is None:
            self.candidate = None
            self.status = "listening"
            return
        if winner != self.candidate:
            self.candidate, self.cand_since = winner, now
            return
        act = ACTIONS[winner][0]
        if rule and isinstance(act, tuple) and act[0] in ("follow", "walk"):   # walking needs a longer hold
            stable_s = max(stable_s, rule.get("follow_hold_s", 0.0))
        if now - self.cand_since < stable_s:
            return
        api_id, secs, follow, label = ACTIONS[winner]
        if self.require_clear and self.clearance != self.CLEAR:
            if now - self.last_hold_log > 2.0:
                log(f"HELD {label}: obstacle check is "
                    f"{self.clear_names.get(self.clearance, 'no data yet')} - not moving")
                self.last_hold_log = now
            self.candidate = None
            return
        log(f"gesture output o{winner} won (rate {r[winner]:.2f}, next gesture "
            f"{r[second] if second is not None else 0:.2f}, none-output {r[4]:.2f})")
        self.last_trick = winner
        if isinstance(api_id, tuple) and api_id[0] == "turn":
            self.start_turn(api_id[1], label)
            self.status = f"o{winner} -> {label}"
            self.candidate = None
            return
        if isinstance(api_id, tuple) and api_id[0] == "walk":
            self.start_walk(api_id[1], label)
            self.status = f"o{winner} -> {label}"
            self.candidate = None
            return
        if isinstance(api_id, tuple) and api_id[0] == "follow":
            self.start_follow(label)
            self.status = f"o{winner} -> {label}"
            self.candidate = None
            return
        self.send(api_id, label)
        self.status = f"o{winner} -> {label}"
        if not self.require_clear:                   # RViz animator: its own (shorter) animation length
            secs = TEST_BUSY_S.get(api_id, secs)
        self.busy_until = now + secs + (0 if follow else COOLDOWN_S)
        self.follow_up = (follow, "Stand up") if follow else None
        self.candidate = None

    # ---------- RViz bars ----------
    def draw(self):
        r, _ = self.window()
        now = self.node.get_clock().now().to_msg()
        arr = MarkerArray()
        for i in range(N_OUT):
            y = 0.40 - i * 0.20
            bar = Marker()
            bar.header.frame_id, bar.header.stamp = "world", now
            bar.ns, bar.id, bar.type, bar.action = "bars", i, Marker.CUBE, Marker.ADD
            h = max(0.005, 0.6 * r[i])
            bar.pose.position.x, bar.pose.position.y, bar.pose.position.z = -0.75, y, h / 2
            bar.pose.orientation.w = 1.0
            bar.scale.x, bar.scale.y, bar.scale.z = 0.06, 0.14, h
            c = COLORS[i]
            bar.color.r, bar.color.g, bar.color.b, bar.color.a = c[0], c[1], c[2], 0.95
            arr.markers.append(bar)
            txt = Marker()
            txt.header.frame_id, txt.header.stamp = "world", now
            txt.ns, txt.id, txt.type, txt.action = "labels", i, Marker.TEXT_VIEW_FACING, Marker.ADD
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = -0.75, y, 0.72
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.032
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            a = ACTIONS.get(i)
            txt.text = f"o{i}\n{a[3] if a else 'none'}\n{r[i]:.2f}"
            arr.markers.append(txt)
        st = Marker()
        st.header.frame_id, st.header.stamp = "world", now
        st.ns, st.id, st.type, st.action = "status", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        st.pose.position.x, st.pose.position.z = -0.75, 0.88
        st.pose.orientation.w = 1.0
        st.scale.z = 0.04
        st.color.r, st.color.g, st.color.b, st.color.a = (0.4, 1.0, 0.6, 1.0) if self.connected else (1.0, 0.3, 0.3, 1.0)
        st.text = ("FPGA:" + self.status.replace(" ", "")) if self.connected else "NO-BOARD-CONNECTION"
        arr.markers.append(st)
        self.bar_pub.publish(arr)

    def report(self):
        r, n = self.window()
        now, f = time.monotonic(), self.frames
        t_last, f_last = getattr(self, "_last_report", (now, f))
        self._last_report = (now, f)
        fps = (f - f_last) / (now - t_last) if now > t_last else 0.0   # camera frames reaching the FPGA
        log("rates " + "  ".join(f"o{i}={x:.2f}" for i, x in enumerate(r))
            + f"   frames={f} ({fps:.0f}/s camera, {n} in window)")

    def run(self):
        import signal

        def _term(*_):                     # systemd / inference.sh stop: same cleanup as Ctrl-C
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _term)
        threading.Thread(target=self.reader, daemon=True).start()
        try:
            rclpy.spin(self.node)
        except (KeyboardInterrupt, Exception) as e:     # noqa: BLE001 - never leave the robot moving
            if not isinstance(e, KeyboardInterrupt):
                log(f"bridge stopped by error: {e!r}")
        finally:
            if self.turning or self.walking or (self.following and not self.dry_run):   # never leave it moving
                self.publish(STOP_ID)
                log("stopped while moving: StopMove sent")
            self.node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.77.1")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--require-clear", action="store_true",
                    help="real robot: only start a trick while /durance/clearance is CLEAR")
    ap.add_argument("--dry-run", action="store_true",
                    help="log decisions but do not publish any robot command")
    ap.add_argument("--no-viz", action="store_true", help="do not publish RViz score markers")
    ap.add_argument("--turn-rate", type=float, default=None,
                    help="max yaw rate of the 360 turns in rad/s when the settings file does not set "
                         "turn_rate (default 1.5; 0.8 = the old, slow setting)")
    ap.add_argument("--settings", default=BS.SETTINGS_FILE if BS else "",
                    help="gesture-control settings (JSON, written by the dashboard; re-read every second)")
    ap.add_argument("--status", default=BS.STATUS_FILE if BS else "",
                    help="write a JSON status snapshot here 10 times per second ('' = off)")
    ap.add_argument("--classes", choices=list(ACTION_SETS), default="own",
                    help="own = our gestures (heart / turn left 360 / turn right 360); "
                         "dataset = organisers' weights (sit / lay down / hello / dance)")
    ap.add_argument("--no-shake-guard", action="store_true",
                    help="do not ignore gestures while the whole camera picture fires (camera moving)")
    ap.add_argument("--follow", choices=["walk", "lidar"], default="walk",
                    help="'Follow me' (own classes, o3): walk = walk --walk-m metres straight forward "
                         "like the other tricks (default); lidar = LiDAR person following, which only "
                         "moves once the person walks away (lidar_follow.py)")
    ap.add_argument("--walk-m", type=float, default=1.0, help="'Follow me' walk distance in m (default 1.0)")
    ap.add_argument("--walk-speed", type=float, default=None, help=f"'Follow me' walk speed in m/s (default {WALK_VX})")
    a = ap.parse_args()
    if a.follow == "lidar":
        ACTION_SETS["own"][3] = (("follow",), 0.0, None, "Follow me")
    else:
        ACTION_SETS["own"][3] = (("walk", min(max(a.walk_m, 0.1), 3.0)), 0.0, None, "Follow me")
    if a.walk_speed:
        WALK_VX = min(max(a.walk_speed, WALK_MIN_VX), 0.5)
    ACTIONS = ACTION_SETS[a.classes]
    if a.turn_rate:
        TURN_MAX_RATE = a.turn_rate
    RULE, ROBOT_RULE = TEST_RULES.get(a.classes), ROBOT_RULES.get(a.classes)
    used = ROBOT_RULE if a.require_clear else RULE
    MIN_FRAMES = (used or {}).get("min_frames", MIN_FRAMES)
    log(f"classes: {a.classes} -> " + ", ".join(f"o{i} {v[3] if v else '-'}" for i, v in ACTIONS.items())
        + f" | {'robot' if a.require_clear else 'test'} rule: {'tuned ' + str(used) if used else 'strict'}")
    Bridge(a.host, a.port, require_clear=a.require_clear, dry_run=a.dry_run, viz=not a.no_viz, classes=a.classes,
           settings_path=a.settings or None, status_path=a.status or None, turn_rate=a.turn_rate,
           shake_guard=not a.no_shake_guard).run()
