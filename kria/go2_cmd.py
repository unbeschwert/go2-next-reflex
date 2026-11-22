#!/usr/bin/env python3
"""Send Unitree Go2 sport-API commands over ROS 2 (/api/sport/request) and show the reply.

Runs ON THE KRIA (it is the machine wired to the Go2):
    source /root/robot_link.sh          # ROS 2 domain 0 on the Go2 network
    python3 /root/go2_cmd.py list
    python3 /root/go2_cmd.py sit
    python3 /root/go2_cmd.py standup
    python3 /root/go2_cmd.py hello
    python3 /root/go2_cmd.py turn 90             # anticlockwise 90 deg, closed loop on IMU yaw
    python3 /root/go2_cmd.py turn -90            # clockwise 90 deg
    python3 /root/go2_cmd.py turn 360 0.8        # full circle, max 0.8 rad/s
    python3 /root/go2_cmd.py yaw                 # print the current heading
    python3 /root/go2_cmd.py move 0.3 0 0 2      # vx m/s, vy m/s, yaw rad/s, seconds
    python3 /root/go2_cmd.py stop

`move` re-sends Move (1008) at 10 Hz for the given time (the Go2 stops when the stream
stops), then sends StopMove (1003). `turn` does the same but stops when the measured yaw
(/sportmodestate imu_state.rpy[2]) has changed by the requested angle. Speeds are capped.
"""
import json
import math
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from unitree_api.msg import Request, Response
from unitree_go.msg import SportModeState

# name: (api_id, description)
COMMANDS = {
    "damp":        (1001, "motors limp - robot drops! only when it is lying down"),
    "balance":     (1002, "balance stand"),
    "stop":        (1003, "stop moving (StopMove)"),
    "standup":     (1004, "stand up (locked joints)"),
    "standdown":   (1005, "lie down"),
    "recovery":    (1006, "recovery stand (get up after falling)"),
    "sit":         (1009, "sit"),
    "risesit":     (1010, "rise from sit"),
    "hello":       (1016, "hello - wave a front paw"),
    "stretch":     (1017, "stretch"),
    "content":     (1020, "content / happy"),
    "dance1":      (1022, "dance 1"),
    "dance2":      (1023, "dance 2"),
    "scrape":      (1029, "rear up and greet"),
    "heart":       (1036, "finger heart with front paws"),
}
MOVE_ID, STOP_ID = 1008, 1003
MAX_VX, MAX_VY, MAX_YAW = 0.5, 0.3, 1.0     # indoor limits (m/s, m/s, rad/s)
MAX_MOVE_S = 5.0

# closed-loop turn
TURN_KP = 1.5               # rad/s of turn rate per rad of remaining angle
TURN_MIN_RATE = 0.25        # rad/s - below this the Go2 may not rotate at all
TURN_TOL_DEG = 2.0          # stop when within this many degrees of the target


def usage():
    print(__doc__)
    print("commands:")
    for name, (api, desc) in COMMANDS.items():
        print(f"  {name:10s} {api}  {desc}")
    print(f"  {'turn':10s} {MOVE_ID}  turn DEGREES [MAX_RATE]  (+ = anticlockwise, - = clockwise, rate <= {MAX_YAW} rad/s)")
    print(f"  {'yaw':10s}  -    print the current heading from /sportmodestate")
    print(f"  {'move':10s} {MOVE_ID}  move VX VY YAW SECONDS  (|vx|<={MAX_VX}, |vy|<={MAX_VY}, |yaw|<={MAX_YAW}, <= {MAX_MOVE_S}s)")


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Go2:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("go2_cmd")
        self.pub = self.node.create_publisher(Request, "/api/sport/request", 10)
        self.replies = []
        self.node.create_subscription(Response, "/api/sport/response", self.replies.append, 10)
        self.yaw = None
        self.node.create_subscription(SportModeState, "/sportmodestate", self.on_state,
                                      qos_profile_sensor_data)

    def on_state(self, msg):
        self.yaw = float(msg.imu_state.rpy[2])

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0.0, end - time.time())))

    def wait_ready(self, need_yaw=False):
        t0 = time.time()
        while time.time() - t0 < 5:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.pub.get_subscription_count() > 0 and (not need_yaw or self.yaw is not None):
                return
        if self.pub.get_subscription_count() == 0:
            sys.exit("no subscriber on /api/sport/request - is robot_link.sh sourced and the Go2 connected?")
        sys.exit("no /sportmodestate from the Go2 - cannot read the yaw angle")

    def send(self, api_id, parameter=""):
        m = Request()
        m.header.identity.api_id = api_id
        m.header.identity.id = int(time.time() * 1000) % (1 << 62)
        m.parameter = parameter
        self.pub.publish(m)

    def move_param(self, vx, vy, yaw_rate):
        return json.dumps({"x": round(vx, 4), "y": round(vy, 4), "z": round(yaw_rate, 4)})

    def report(self, api_ids):
        self.spin(1.0)
        codes = [(r.header.identity.api_id, r.header.status.code) for r in self.replies
                 if r.header.identity.api_id in api_ids]
        if codes:
            for a, c in codes[-2:]:
                print(f"Go2 reply: api_id={a} code={c} ({'accepted' if c == 0 else 'rejected'})")
        else:
            print("no reply from the Go2 within 1 s (it may still be executing)")

    def close(self):
        self.node.destroy_node()
        rclpy.try_shutdown()


def do_turn(go2, degrees, max_rate):
    go2.wait_ready(need_yaw=True)
    target = math.radians(degrees)
    tol = math.radians(TURN_TOL_DEG)
    max_rate = min(abs(max_rate), MAX_YAW)
    start = last = go2.yaw
    turned = 0.0                                   # unwrapped, so 360/720 deg work too
    timeout = abs(target) / TURN_MIN_RATE + 5.0
    print(f"turn {degrees:+.0f} deg ({'anticlockwise' if degrees > 0 else 'clockwise'}), "
          f"start yaw {math.degrees(start):+.1f} deg, max {max_rate:.2f} rad/s", flush=True)
    t0 = time.time()
    try:
        while True:
            go2.spin(0.05)
            turned += wrap(go2.yaw - last)
            last = go2.yaw
            remaining = target - turned
            if abs(remaining) <= tol:
                break
            if time.time() - t0 > timeout:
                print("timeout - stopping", flush=True)
                break
            rate = max(TURN_MIN_RATE, min(max_rate, TURN_KP * abs(remaining)))
            go2.send(MOVE_ID, go2.move_param(0.0, 0.0, math.copysign(rate, remaining)))
    finally:
        go2.send(STOP_ID)
    go2.spin(1.0)                                   # let it settle, keep integrating
    turned += wrap(go2.yaw - last)
    print(f"StopMove sent. turned {math.degrees(turned):+.1f} deg (target {degrees:+.0f}, "
          f"error {math.degrees(turned - target):+.1f}), now yaw {math.degrees(go2.yaw):+.1f} deg, "
          f"{time.time() - t0:.1f}s", flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("list", "-h", "--help"):
        usage()
        return
    cmd = sys.argv[1].lower()
    if cmd not in COMMANDS and cmd not in ("move", "turn", "yaw"):
        sys.exit(f"unknown command {cmd!r} - try: list")

    go2 = Go2()
    try:
        if cmd == "yaw":
            go2.wait_ready(need_yaw=True)
            print(f"yaw {math.degrees(go2.yaw):+.1f} deg ({go2.yaw:+.3f} rad)")
        elif cmd == "turn":
            try:
                degrees = float(sys.argv[2])
                max_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
            except (IndexError, ValueError):
                sys.exit("usage: turn DEGREES [MAX_RATE]   e.g. turn 90  /  turn -90  /  turn 360 0.8")
            do_turn(go2, degrees, max_rate)
            go2.report((MOVE_ID, STOP_ID))
        elif cmd == "move":
            go2.wait_ready()
            try:
                vx, vy, yaw, secs = (float(x) for x in sys.argv[2:6])
            except ValueError:
                sys.exit("usage: move VX VY YAW SECONDS   e.g. move 0.3 0 0 2")
            clip = lambda v, lim: max(-lim, min(lim, v))
            vx, vy, yaw = clip(vx, MAX_VX), clip(vy, MAX_VY), clip(yaw, MAX_YAW)
            secs = min(max(secs, 0), MAX_MOVE_S)
            param = go2.move_param(vx, vy, yaw)
            print(f"Move {param} for {secs:.1f}s ...", flush=True)
            end = time.time() + secs
            try:
                while time.time() < end:
                    go2.send(MOVE_ID, param)
                    go2.spin(0.1)
            finally:
                go2.send(STOP_ID)
                print("StopMove sent", flush=True)
            go2.report((MOVE_ID, STOP_ID))
        else:
            go2.wait_ready()
            api_id, desc = COMMANDS[cmd]
            print(f"{cmd} (api_id={api_id}): {desc}", flush=True)
            go2.send(api_id)
            go2.report((api_id,))
    except KeyboardInterrupt:
        go2.send(STOP_ID)
        print("interrupted - StopMove sent")
    finally:
        go2.close()


if __name__ == "__main__":
    main()
