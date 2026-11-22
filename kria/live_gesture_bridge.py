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
import socket
import struct
import threading
import time
from collections import deque

import rclpy
from rclpy.qos import qos_profile_sensor_data
from unitree_api.msg import Request
from visualization_msgs.msg import Marker, MarkerArray

N_OUT = 5
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
COOLDOWN_S = 2.0         # ignore the camera this long after a trick finishes
QUIET_RATE = 0.20        # re-arm only after every gesture output has dropped below this
                         # (breaks loops where the robot's own motion looks like a gesture)

# output index -> (action, busy seconds, follow-up api_id or None, label)
# action = sport api_id, or ("turn", degrees) = closed-loop turn (+ = left / anticlockwise)
ACTION_SETS = {
    # our own gestures (go2_viz/gestures.json; training label = id - 1):
    # gesture_01 heart, 02 turn-left-360, 03 turn-right-360, 04 namaskaram (unused), 05 no-action
    "own": {
        0: (1036, 5.0, None, "Heart"),                    # finger heart with the front paws
        1: (("turn", 360.0), 0.0, None, "Turn left 360"),  # busy time = until the turn ends
        2: (("turn", -360.0), 0.0, None, "Turn right 360"),
        3: None,                                          # namaskaram: dropped, output unused
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
    "own": {"min_rate": {0: 0.50, 1: 0.50, 2: 0.50}, "margin": 0.30, "stable_s": 0.20,
            "quiet": 0.35, "beat_none": 0.20, "min_frames": 40},
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
TURN_MAX_RATE = 0.8      # rad/s (go2_cmd.py caps at 1.0) -> a full circle takes ~9 s
TURN_MIN_RATE = 0.25     # rad/s - below this the Go2 may not rotate at all
TURN_KP = 1.5            # rad/s of turn rate per rad of remaining angle (slows down at the end)
TURN_TOL_DEG = 3.0       # stop when this close to the target
YAW_STALE_S = 0.5        # closed loop: stop the turn if /sportmodestate goes quiet this long
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


class Bridge:
    def __init__(self, host, port, require_clear=False, dry_run=False, viz=True):
        self.host, self.port = host, port
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
        self.armed = True        # False after a trick until the gesture outputs calm down
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
        try:
            from unitree_go.msg import SportModeState
            self.node.create_subscription(SportModeState, "/sportmodestate", self.on_state,
                                          qos_profile_sensor_data)
        except ImportError:
            log("unitree_go messages not found: turns will run open loop")
        self.node.create_timer(0.05, self.decide)
        if viz:
            self.node.create_timer(0.1, self.draw)
        self.node.create_timer(2.0, self.report)
        log(f"mode: {'DRY-RUN (no commands sent)' if dry_run else 'LIVE (commands sent)'}, "
            f"obstacle check {'ON' if require_clear else 'OFF'}")

    def on_clearance(self, msg):
        if msg.state != self.clearance:
            log(f"obstacle check: {self.clear_names.get(msg.state, msg.state)}"
                + (f" ({msg.range:.2f} m)" if msg.range == msg.range else ""))
        self.clearance = msg.state

    def on_state(self, msg):
        self.yaw, self.yaw_time = float(msg.imu_state.rpy[2]), time.monotonic()

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
                    else:
                        continue
                    if L != 4 * N_OUT or off + L > len(d):
                        continue
                    scores = struct.unpack(f"<{N_OUT}f", d[off:off + L])
                    with self.lock:
                        self.hist.append((time.monotonic(), [1.0 if x > 0 else 0.0 for x in scores]))
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
            self.busy_until = time.monotonic() + 1.0 + COOLDOWN_S

    def decide(self):
        now = time.monotonic()
        if now < self.busy_until:
            # robot is moving (or settling): what the camera sees now is mostly the
            # robot's own motion, so throw it away and start the window fresh later
            with self.lock:
                self.hist.clear()
            self.window_from = now
            self.armed = False
            return
        if self.follow_up:
            api_id, label = self.follow_up
            self.follow_up = None
            self.send(api_id, label)
            self.busy_until = now + 3.0 + COOLDOWN_S
            self.status = label
            return
        r, _ = self.window()
        if now - self.window_from < WINDOW_S:        # window not filled since the last trick yet
            return
        gestures = [i for i in range(N_OUT) if ACTIONS.get(i) is not None]
        rule = ROBOT_RULE if self.require_clear else RULE
        if rule:
            thr = {i: rule["min_rate"].get(i, MIN_RATE) for i in gestures}
            quiet = {i: min(thr[i], rule["quiet"]) for i in gestures}
            need_margin, stable_s = rule["margin"], rule["stable_s"]
        else:
            thr = {i: MIN_RATE for i in gestures}
            quiet = {i: QUIET_RATE for i in gestures}
            need_margin, stable_s = MIN_MARGIN, STABLE_S
        if not getattr(self, "armed", True):
            if all(r[i] < quiet[i] for i in gestures):
                self.armed = True
                log("calm - ready for the next gesture")
            else:
                self.status = "waiting-for-calm"
            return
        order = sorted(gestures, key=lambda i: r[i], reverse=True)
        top = order[0]
        second = order[1] if len(order) > 1 else None
        margin = r[top] - (r[second] if second is not None else 0.0)
        winner = top if (r[top] >= thr[top] and margin >= need_margin) else None
        if (winner is not None and rule and rule.get("beat_none") is not None
                and r[top] < r[N_OUT - 1] + rule["beat_none"]):
            winner = None                            # the gesture must also beat the "none" bar
        if winner is None:
            self.candidate = None
            self.status = "listening"
            return
        if winner != self.candidate:
            self.candidate, self.cand_since = winner, now
            return
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
        if isinstance(api_id, tuple) and api_id[0] == "turn":
            self.start_turn(api_id[1], label)
            self.status = f"o{winner} -> {label}"
            self.candidate = None
            return
        self.send(api_id, label)
        self.status = f"o{winner} -> {label}"
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
        threading.Thread(target=self.reader, daemon=True).start()
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            if self.turning:                  # never leave the robot spinning
                self.publish(STOP_ID)
                log("stopped during a turn: StopMove sent")
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
    ap.add_argument("--classes", choices=list(ACTION_SETS), default="own",
                    help="own = our gestures (heart / turn left 360 / turn right 360); "
                         "dataset = organisers' weights (sit / lay down / hello / dance)")
    a = ap.parse_args()
    ACTIONS = ACTION_SETS[a.classes]
    RULE, ROBOT_RULE = TEST_RULES.get(a.classes), ROBOT_RULES.get(a.classes)
    used = ROBOT_RULE if a.require_clear else RULE
    MIN_FRAMES = (used or {}).get("min_frames", MIN_FRAMES)
    log(f"classes: {a.classes} -> " + ", ".join(f"o{i} {v[3] if v else '-'}" for i, v in ACTIONS.items())
        + f" | {'robot' if a.require_clear else 'test'} rule: {'tuned ' + str(used) if used else 'strict'}")
    Bridge(a.host, a.port, require_clear=a.require_clear, dry_run=a.dry_run, viz=not a.no_viz).run()
