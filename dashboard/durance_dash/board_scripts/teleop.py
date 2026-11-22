"""Dead-man teleoperation of the Go2 over ROS 2 (/api/sport/request), driven by JSON lines on stdin.

    python3 -c <this> MAX_VX MAX_VY MAX_WZ DEADMAN_S      (run after sourcing the ROS environment of
                                                          the link: robot = domain 0, test = 42, sim)
stdin, one JSON object per line:
    {"op": "move", "vx": 0.2, "vy": 0.0, "wz": 0.0}   Move (1008); must repeat faster than DEADMAN_S
    {"op": "stop"}                                     StopMove (1003)
    {"op": "estop"}                                    StopMove burst (5x over 0.5 s)
    {"op": "api", "id": 1004}                          one whitelisted sport command
    {"op": "ping"}
stdout: {"type": "ready"}, {"type": "ack", ...}, {"type": "state", ...} (5 Hz from /sportmodestate),
        {"type": "deadman"} when the robot was stopped because the moves stopped, {"type": "error", ...}.
Safety: speeds are clamped here as well as in the dashboard; no move for DEADMAN_S -> StopMove;
stdin closed (SSH session dropped, dashboard stopped) -> StopMove x3 and exit.
"""
import json
import math
import sys
import threading
import time

import rclpy
from unitree_api.msg import Request

try:
    from rclpy.qos import qos_profile_sensor_data
    from unitree_go.msg import SportModeState
except ImportError:
    SportModeState = None

MOVE, STOP = 1008, 1003
ALLOWED = {1001, 1002, 1003, 1004, 1005, 1006, 1009, 1010, 1016, 1017, 1020, 1022, 1023, 1029, 1036}
MAX = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]
DEADMAN = float(sys.argv[4])

out_lock = threading.Lock()


def emit(obj):
    with out_lock:
        try:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass


rclpy.init()
node = rclpy.create_node("durance_teleop")
pub = node.create_publisher(Request, "/api/sport/request", 10)
state = {"moving": False, "last_move": 0.0, "yaw": None, "pos": None, "mode": None, "t_state": 0.0,
         "body_height": None, "vel": None}


def publish(api_id, param=""):
    m = Request()
    m.header.identity.api_id = api_id
    m.header.identity.id = int(time.time() * 1e6) % (1 << 62)
    m.parameter = param
    pub.publish(m)


def clamp(v, lim):
    v = float(v)
    if not math.isfinite(v):
        return 0.0
    return max(-lim, min(lim, v))


if SportModeState is not None:
    def on_state(msg):
        state.update(yaw=float(msg.imu_state.rpy[2]), pos=[float(x) for x in msg.position],
                     mode=int(msg.mode), body_height=float(msg.body_height),
                     vel=[float(x) for x in msg.velocity], t_state=time.monotonic())
    node.create_subscription(SportModeState, "/sportmodestate", on_state, qos_profile_sensor_data)

threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


def watchdog():
    last_report = 0.0
    while rclpy.ok():
        now = time.monotonic()
        if state["moving"] and now - state["last_move"] > DEADMAN:
            publish(STOP)
            state["moving"] = False
            emit({"type": "deadman"})
        if now - last_report >= 0.2:
            last_report = now
            fresh = now - state["t_state"] < 1.0
            emit({"type": "state", "moving": state["moving"], "yaw": state["yaw"] if fresh else None,
                  "pos": state["pos"] if fresh else None, "mode": state["mode"] if fresh else None,
                  "body_height": state["body_height"] if fresh else None, "vel": state["vel"] if fresh else None})
        time.sleep(0.02)


threading.Thread(target=watchdog, daemon=True).start()
emit({"type": "ready", "limits": MAX, "deadman_s": DEADMAN})
try:
    for raw in sys.stdin:
        try:
            msg = json.loads(raw)
            op = msg.get("op")
            if op == "move":
                v = [clamp(msg.get("vx", 0), MAX[0]), clamp(msg.get("vy", 0), MAX[1]), clamp(msg.get("wz", 0), MAX[2])]
                if any(abs(x) > 1e-3 for x in v):
                    publish(MOVE, json.dumps({"x": round(v[0], 3), "y": round(v[1], 3), "z": round(v[2], 3)}))
                    state["moving"], state["last_move"] = True, time.monotonic()
                elif state["moving"]:
                    publish(STOP)
                    state["moving"] = False
                emit({"type": "ack", "op": "move", "v": v})
            elif op == "stop":
                publish(STOP)
                state["moving"] = False
                emit({"type": "ack", "op": "stop"})
            elif op == "estop":
                state["moving"] = False
                for _ in range(5):
                    publish(STOP)
                    time.sleep(0.1)
                emit({"type": "ack", "op": "estop"})
            elif op == "api":
                api_id = int(msg.get("id"))
                if api_id not in ALLOWED:
                    emit({"type": "error", "text": "api_id %d not allowed" % api_id})
                    continue
                state["moving"] = False
                publish(api_id)
                emit({"type": "ack", "op": "api", "id": api_id})
            elif op == "ping":
                emit({"type": "ack", "op": "ping"})
            else:
                emit({"type": "error", "text": "unknown op %r" % op})
        except (ValueError, TypeError) as e:
            emit({"type": "error", "text": "bad message: %s" % e})
finally:
    for _ in range(3):                       # stdin closed: the link to the dashboard is gone
        publish(STOP)
        time.sleep(0.05)
    rclpy.try_shutdown()
