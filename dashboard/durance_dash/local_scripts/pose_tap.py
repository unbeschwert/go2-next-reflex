"""Prints the simulated Go2's pose for the dashboard's 3D view (run with the ROS 2 environment of the
domain to watch: the local simulator or the board's test mode).

    python3 pose_tap.py SOURCE_NAME        -> JSON lines, 20 per second

Subscribes /tf (world -> base), /joint_states, /go2_viz/label (current action text) and
/api/sport/request (last Move command), exactly what RViz shows.
"""
import json
import math
import sys
import threading
import time

import rclpy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

try:
    from unitree_api.msg import Request
except ImportError:
    Request = None

source = sys.argv[1] if len(sys.argv) > 1 else "sim"
rclpy.init()
node = rclpy.create_node("durance_pose_tap")
s = {"pose": None, "joints": {}, "action": "", "cmd": {"vx": 0.0, "vy": 0.0, "wz": 0.0}, "t_pose": 0.0, "t_cmd": 0.0}


def on_tf(msg):
    for t in msg.transforms:
        if t.child_frame_id == "base" and t.header.frame_id in ("world", "odom", "map"):
            tr, r = t.transform.translation, t.transform.rotation
            s["pose"] = (tr.x, tr.y, tr.z, r.x, r.y, r.z, r.w)
            s["t_pose"] = time.monotonic()


def on_js(msg):
    s["joints"] = {n: float(p) for n, p in zip(msg.name, msg.position)}


def on_label(msg):
    s["action"] = msg.text.replace("-", " ")


def on_req(msg):
    if msg.header.identity.api_id == 1008:
        try:
            p = json.loads(msg.parameter or "{}")
            s["cmd"] = {"vx": float(p.get("x", 0)), "vy": float(p.get("y", 0)), "wz": float(p.get("z", 0))}
            s["t_cmd"] = time.monotonic()
        except ValueError:
            pass
    elif msg.header.identity.api_id == 1003:
        s["cmd"] = {"vx": 0.0, "vy": 0.0, "wz": 0.0}


node.create_subscription(TFMessage, "/tf", on_tf, 20)
node.create_subscription(JointState, "/joint_states", on_js, 20)
node.create_subscription(Marker, "/go2_viz/label", on_label, 5)
if Request is not None:
    node.create_subscription(Request, "/api/sport/request", on_req, 20)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
try:
    while rclpy.ok():
        time.sleep(0.05)
        now = time.monotonic()
        if s["pose"] is None or now - s["t_pose"] > 2.0:
            out = {"source": source, "stale": True}
        else:
            x, y, z, qx, qy, qz, qw = s["pose"]
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            cmd = s["cmd"] if now - s["t_cmd"] < 0.6 else {"vx": 0.0, "vy": 0.0, "wz": 0.0}
            out = {"source": source, "x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw, "yaw": yaw,
                   "joints": s["joints"], "action": s["action"], "cmd": cmd}
        out["t"] = time.time()
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
except (KeyboardInterrupt, BrokenPipeError):
    pass
finally:
    rclpy.try_shutdown()
