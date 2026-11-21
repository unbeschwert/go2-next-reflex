#!/usr/bin/env python3
"""Animates the Unitree Go2 model in RViz from the sport commands the pipeline sends.

Subscribes to /api/sport/request (the same commands robot_sim / the real Go2 get)
and plays a pose animation for each api_id: Sit, StandUp, StandDown, Hello,
Stretch, Content, Dance, FingerHeart, ... It publishes /joint_states for
robot_state_publisher, the world->base transform (body height and pitch are
solved so the supporting feet touch the floor), and a text label in RViz.

This is a visual stand-in for Unitree's proprietary sport controller, not a
physics simulation: it shows WHICH trick the gesture pipeline commanded.
"""
import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from unitree_api.msg import Request
from visualization_msgs.msg import Marker

L1 = L2 = 0.213          # thigh / calf length (m)
HIP_X = 0.1934           # hip joint x offset from body centre (m)
FOOT_R = 0.022           # foot sphere radius (m)
LEGS = ("FR", "FL", "RR", "RL")
BLEND = 0.7              # seconds to blend into a new pose

STAND = (0.0, 0.8, -1.5)
LIE = (0.0, 1.25, -2.70)
SIT_FRONT = (0.0, 0.10, -0.90)
SIT_REAR = (0.0, 1.90, -2.60)
NAMES = {1001: "Damp", 1003: "Stop move", 1004: "Stand up", 1005: "Lie down",
         1006: "Recovery stand", 1009: "Sit", 1010: "Rise from sit", 1016: "Hello (wave paw)",
         1017: "Stretch", 1020: "Content", 1022: "Dance 1", 1023: "Dance 2",
         1029: "Rear up + greet", 1036: "Finger heart"}


def legs(fr=STAND, fl=STAND, rr=STAND, rl=STAND):
    return {"FR": fr, "FL": fl, "RR": rr, "RL": rl}


def sit_pose():
    return legs(SIT_FRONT, SIT_FRONT, SIT_REAR, SIT_REAR)


def foot_xz(leg, q):
    """Foot position in the body's sagittal plane (x forward, z up)."""
    _, th, ca = q
    hx = HIP_X if leg[0] == "F" else -HIP_X
    return (hx - L1 * math.sin(th) - L2 * math.sin(th + ca),
            -L1 * math.cos(th) - L2 * math.cos(th + ca))


def body_pose(pose, grounded):
    """Pitch and height so the grounded front and rear feet rest on the floor."""
    front = [foot_xz(l, pose[l]) for l in grounded if l[0] == "F"] or [foot_xz("FL", STAND)]
    rear = [foot_xz(l, pose[l]) for l in grounded if l[0] == "R"] or [foot_xz("RL", STAND)]
    xf = sum(p[0] for p in front) / len(front)
    zf = sum(p[1] for p in front) / len(front)
    xr = sum(p[0] for p in rear) / len(rear)
    zr = sum(p[1] for p in rear) / len(rear)
    pitch = math.atan2(zf - zr, xf - xr)
    height = FOOT_R + xf * math.sin(pitch) - zf * math.cos(pitch)
    return pitch, height


class Action:
    """An animation: pose(t) -> (leg angles, grounded legs, roll, yaw), plus a length."""

    def __init__(self, name, fn, seconds=None, then=None):
        self.name, self.fn, self.seconds, self.then = name, fn, seconds, then


def hold(pose):
    return lambda t: (pose, LEGS, 0.0, 0.0)


def hello(t):
    p = sit_pose()
    wave = 0.35 * math.sin(t * 7.0)
    p["FR"] = (-0.25 + wave, -1.10, -1.20)
    return p, ("FL", "RR", "RL"), 0.0, 0.0


def heart(t):
    p = sit_pose()
    bob = 0.15 * math.sin(t * 4.0)
    p["FR"] = (0.15, -1.30 + bob, -1.40)
    p["FL"] = (-0.15, -1.30 + bob, -1.40)
    return p, ("RR", "RL"), 0.0, 0.0


def stretch(t):
    k = min(1.0, t / 0.8)
    front = (0.0, 0.8 - 0.9 * k, -1.5 - 0.9 * k)
    return legs(front, front, STAND, STAND), LEGS, 0.0, 0.0


def content(t):
    return legs(), LEGS, 0.12 * math.sin(t * 6.0), 0.10 * math.sin(t * 3.0)


def dance(t):
    phase = int(t * 2.5) % 4
    lifted = LEGS[phase]
    p = legs()
    p[lifted] = (0.0, 0.40, -2.10)
    grounded = tuple(l for l in LEGS if l != lifted)
    return p, grounded, 0.15 * math.sin(t * 5.0), 0.25 * math.sin(t * 2.5)


def rear_up(t):
    p = legs((0.0, -0.6, -1.6), (0.0, -0.6, -1.6), (0.0, 1.6, -2.4), (0.0, 1.6, -2.4))
    p["FR"] = (0.0, -0.6 + 0.3 * math.sin(t * 6.0), -1.6)
    return p, ("RR", "RL"), 0.0, 0.0


def make_action(api_id):
    name = NAMES.get(api_id, f"api_id {api_id}")
    stand = Action("Standing", hold(legs()))
    table = {
        1004: Action(name, hold(legs())),
        1006: Action(name, hold(legs())),
        1010: Action(name, hold(legs())),
        1005: Action(name, hold(legs(LIE, LIE, LIE, LIE))),
        1001: Action(name, hold(legs(LIE, LIE, LIE, LIE))),
        1009: Action(name, hold(sit_pose())),
        1016: Action(name, hello, 3.5, stand),
        1036: Action(name, heart, 3.5, stand),
        1017: Action(name, stretch, 2.5, stand),
        1020: Action(name, content, 3.0, stand),
        1022: Action(name, dance, 10.0, stand),
        1023: Action(name, dance, 10.0, stand),
        1029: Action(name, rear_up, 3.0, stand),
    }
    return table.get(api_id)


class Go2Animator(Node):
    def __init__(self):
        super().__init__("go2_animator")
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.label_pub = self.create_publisher(Marker, "/go2_viz/label", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Request, "/api/sport/request", self.on_request, 10)
        self.action = Action("Standing", hold(legs()))
        self.started = time.monotonic()
        self.from_state = self.sample(self.action, 0.0)
        self.state = self.from_state
        self.create_timer(1.0 / 50, self.tick)
        self.get_logger().info("Go2 animator ready, listening on /api/sport/request")

    def sample(self, action, t):
        pose, grounded, roll, yaw = action.fn(t)
        pitch, height = body_pose(pose, grounded)
        return {"legs": pose, "roll": roll, "yaw": yaw, "pitch": pitch, "height": height}

    def start(self, action):
        self.from_state = self.state
        self.action = action
        self.started = time.monotonic()

    def on_request(self, msg):
        api_id = msg.header.identity.api_id
        action = make_action(api_id)
        self.get_logger().info(f"command api_id={api_id} -> {NAMES.get(api_id, 'unknown')}")
        if action:
            self.start(action)

    def tick(self):
        t = time.monotonic() - self.started
        a = self.action
        if a.seconds is not None and t > a.seconds and a.then is not None:
            self.start(a.then)
            t, a = 0.0, self.action
        target = self.sample(a, t)
        s = min(1.0, t / BLEND)
        s = s * s * (3 - 2 * s)
        f = self.from_state
        mix = lambda x, y: x + (y - x) * s
        cur = {
            "legs": {l: tuple(mix(x, y) for x, y in zip(f["legs"][l], target["legs"][l])) for l in LEGS},
            "roll": mix(f["roll"], target["roll"]), "yaw": mix(f["yaw"], target["yaw"]),
            "pitch": mix(f["pitch"], target["pitch"]), "height": mix(f["height"], target["height"]),
        }
        self.state = cur
        self.publish(cur, a.name)

    def publish(self, st, label):
        now = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        for l in LEGS:
            for part, v in zip(("hip", "thigh", "calf"), st["legs"][l]):
                js.name.append(f"{l}_{part}_joint")
                js.position.append(v)
        self.joint_pub.publish(js)

        r, p, y = st["roll"], st["pitch"], st["yaw"]
        cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                                  math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "world"
        tf.child_frame_id = "base"
        tf.transform.translation.z = st["height"]
        tf.transform.rotation.w = cr * cp * cy + sr * sp * sy
        tf.transform.rotation.x = sr * cp * cy - cr * sp * sy
        tf.transform.rotation.y = cr * sp * cy + sr * cp * sy
        tf.transform.rotation.z = cr * cp * sy - sr * sp * cy
        self.tf.sendTransform(tf)

        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = now
        m.ns, m.id, m.type, m.action = "label", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.z = 0.62
        m.pose.orientation.w = 1.0
        m.scale.z = 0.055
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.1, 1.0
        m.text = label.split(' (')[0].replace(' ', '-')   # RViz spreads words apart on spaces
        self.label_pub.publish(m)


def main():
    rclpy.init()
    node = Go2Animator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
