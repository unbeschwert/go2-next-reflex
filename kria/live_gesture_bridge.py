#!/usr/bin/env python3
"""Live bridge: Kria FPGA gesture model -> Go2 sport commands (RViz / robot_sim).

Connects to the SPLEAT server on the board (the camera must be feeding it, e.g.
`despot_send_and_forget --to_display 0` on the Kria), turns the model's 5 binary
outputs per 10 ms step into firing rates over a sliding window, shows them as bars
next to the 3D dog in RViz, and publishes a /api/sport/request when one output
clearly and stably wins.

    on the Kria (normally via kria/inference.sh):
        python3 /root/live_gesture_bridge.py --host 127.0.0.1 [--require-clear] [--dry-run] [--no-viz]
    on the laptop (talks to the board's SPLEAT server over the cable):
        source laptop/env.sh
        python3 kria/live_gesture_bridge.py --host 192.168.77.1

The output class order of the provided model is unknown, so each output is mapped
to a different trick (see ACTIONS) - watch which bar rises for which gesture.
Change ACTIONS once the order is known. Output 4 is treated as "no gesture"
because it fires most of the time.
"""
import argparse
import socket
import struct
import threading
import time
from collections import deque

import rclpy
from unitree_api.msg import Request
from visualization_msgs.msg import Marker, MarkerArray

N_OUT = 5
WINDOW = 50              # frames (100 Hz) -> 0.5 s sliding window
MIN_RATE = 0.50          # winner must fire in >= 50 % of the window
MIN_MARGIN = 0.20        # ...and beat the runner-up GESTURE by 20 points
                         # (the "no gesture" output is not part of this comparison:
                         #  it keeps firing in the background even during gestures)
STABLE_S = 0.40          # ...continuously for this long
COOLDOWN_S = 2.0         # ignore the camera this long after a trick finishes
QUIET_RATE = 0.20        # re-arm only after every gesture output has dropped below this
                         # (breaks loops where the robot's own motion looks like a gesture)

# output index -> (api_id, busy seconds, follow-up api_id or None, label)
ACTIONS = {
    0: (1009, 4.0, 1004, "Sit"),
    1: (1016, 4.0, None, "Hello"),
    2: (1022, 8.0, None, "Dance"),
    3: (1017, 3.0, None, "Stretch"),
    4: None,              # most likely "no gesture / background"
}
COLORS = [(0.95, 0.45, 0.2), (0.3, 0.75, 1.0), (0.85, 0.35, 0.9), (0.35, 0.9, 0.45), (0.6, 0.6, 0.6)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Bridge:
    def __init__(self, host, port, require_clear=False, dry_run=False, viz=True):
        self.host, self.port = host, port
        self.require_clear, self.dry_run = require_clear, dry_run
        self.hist = deque(maxlen=WINDOW)
        self.lock = threading.Lock()
        self.rates = [0.0] * N_OUT
        self.connected = False
        self.frames = 0
        rclpy.init()
        self.node = rclpy.create_node("live_gesture_bridge")
        self.req_pub = self.node.create_publisher(Request, "/api/sport/request", 10)
        self.bar_pub = self.node.create_publisher(MarkerArray, "/go2_viz/scores", 10)
        self.busy_until = 0.0
        self.follow_up = None
        self.candidate, self.cand_since = None, 0.0
        self.status = "waiting for board..."
        # Real robot safety: only start a trick while obstacle_detection reports CLEAR
        self.clearance, self.clear_names, self.last_hold_log = None, {}, 0.0
        if require_clear:
            from msgs.msg import Clearance
            self.CLEAR = Clearance.CLEAR
            self.clear_names = {Clearance.CLEAR: "CLEAR", Clearance.OBSTACLE: "OBSTACLE",
                                Clearance.UNKNOWN: "UNKNOWN"}
            self.node.create_subscription(Clearance, "/durance/clearance", self.on_clearance, 1)
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

    # ---------- board connection ----------
    def reader(self):
        while rclpy.ok():
            try:
                s = socket.create_connection((self.host, self.port), timeout=3)
                s.settimeout(3)
                self.connected = True
                log(f"connected to FPGA model server {self.host}:{self.port}")
                while rclpy.ok():
                    head = s.recv(4, socket.MSG_WAITALL)
                    if len(head) < 4:
                        raise OSError("server closed connection")
                    n = struct.unpack(">I", head)[0]
                    d = s.recv(n, socket.MSG_WAITALL)
                    k = d.find(b"\x67payload")
                    if k < 0:
                        continue
                    h = d[k + 8]
                    L, off = (h & 0x1F, k + 9) if h < 0x58 else (d[k + 9], k + 10)
                    if L != 4 * N_OUT:
                        continue
                    scores = struct.unpack(f"<{N_OUT}f", d[off:off + L])
                    with self.lock:
                        self.hist.append([1.0 if x > 0 else 0.0 for x in scores])
                        cnt = len(self.hist)
                        self.rates = [sum(f[i] for f in self.hist) / cnt for i in range(N_OUT)]
                        self.frames += 1
            except OSError as e:
                self.connected = False
                log(f"board connection lost ({e}); retrying in 1 s")
                time.sleep(1)

    # ---------- decision ----------
    def send(self, api_id, label):
        if self.dry_run:
            log(f"DRY-RUN would send -> {label} (api_id={api_id})")
            return
        m = Request()
        m.header.identity.api_id = api_id
        self.req_pub.publish(m)
        log(f"ROBOT COMMAND -> {label} (api_id={api_id})")

    def decide(self):
        now = time.monotonic()
        if now < self.busy_until:
            # robot is moving (or settling): what the camera sees now is mostly the
            # robot's own motion, so throw it away and start the window fresh later
            with self.lock:
                self.hist.clear()
            self.armed = False
            return
        if self.follow_up:
            api_id, label = self.follow_up
            self.follow_up = None
            self.send(api_id, label)
            self.busy_until = now + 3.0 + COOLDOWN_S
            self.status = label
            return
        with self.lock:
            r = list(self.rates)
            full = len(self.hist) == WINDOW
        if not full:
            return
        gestures = [i for i in range(N_OUT) if ACTIONS.get(i) is not None]
        if not getattr(self, "armed", True):
            if max(r[i] for i in gestures) < QUIET_RATE:
                self.armed = True
                log("calm - ready for the next gesture")
            else:
                self.status = "waiting-for-calm"
            return
        order = sorted(gestures, key=lambda i: r[i], reverse=True)
        top = order[0]
        second = order[1] if len(order) > 1 else None
        margin = r[top] - (r[second] if second is not None else 0.0)
        winner = top if (r[top] >= MIN_RATE and margin >= MIN_MARGIN) else None
        if winner is None:
            self.candidate = None
            self.status = "listening"
            return
        if winner != self.candidate:
            self.candidate, self.cand_since = winner, now
            return
        if now - self.cand_since < STABLE_S:
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
        self.send(api_id, label)
        self.status = f"o{winner} -> {label}"
        self.busy_until = now + secs + (0 if follow else COOLDOWN_S)
        self.follow_up = (follow, "Stand up") if follow else None
        self.candidate = None

    # ---------- RViz bars ----------
    def draw(self):
        with self.lock:
            r = list(self.rates)
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
        with self.lock:
            r, f = list(self.rates), self.frames
        log("rates " + "  ".join(f"o{i}={x:.2f}" for i, x in enumerate(r)) + f"   frames={f}")

    def run(self):
        threading.Thread(target=self.reader, daemon=True).start()
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
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
    a = ap.parse_args()
    Bridge(a.host, a.port, require_clear=a.require_clear, dry_run=a.dry_run, viz=not a.no_viz).run()
