#!/usr/bin/env python3
"""Interactive laptop test bench for the gesture -> Go2 pipeline (no board, no robot).

Starts the simulated Go2 (robot_sim) and the real pipeline (gesture_node,
action_bridge, obstacle_detection, profiling_node), plays the role of the FPGA
SPLEAT server, and fakes the robot's LiDAR so the obstacle check passes.

Usage:
    source ros2/env.sh
    python3 ros2/sim_playground.py

Then type a command and press Enter:
    0..N-1   perform that gesture for 1.5 s (model output = 1.0 for that class)
    n        nobody in front of the camera (all scores 0)
    o        toggle an obstacle 0.4 m in front of the robot
    q        quit
"""
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import PoseStamped
from msgs.msg import Clearance, Gestures
from unitree_api.msg import Request, Response
from unitree_go.msg import HeightMap

N = len(Gestures().scores)
DVS_LABELS = ["hand clap", "right hand wave", "left hand wave", "right arm cw",
              "right arm ccw", "left arm cw", "left arm ccw", "arm roll",
              "air drums", "air guitar", "other"]
LABELS = DVS_LABELS if N == len(DVS_LABELS) else [f"class {i}" for i in range(N)]
ACTIONS = {1001: "Damp", 1003: "StopMove", 1004: "StandUp", 1005: "StandDown (lie down)",
           1006: "RecoveryStand", 1009: "Sit", 1010: "RiseSit", 1016: "Hello (wave paw)",
           1017: "Stretch", 1020: "Content", 1022: "Dance1", 1023: "Dance2",
           1029: "Scrape (rear up + greet)", 1036: "FingerHeart"}
CLEARANCE = {Clearance.CLEAR: "CLEAR", Clearance.OBSTACLE: "BLOCKED (obstacle)",
             Clearance.UNKNOWN: "UNKNOWN"}
GESTURE_SECONDS = 1.5
FRAME_HZ = 100

state = {"scores": [0.0] * N, "until": 0.0, "obstacle": False, "stop": False}
lock = threading.Lock()


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cbor_frame(scores):
    """Same framing the SPLEAT server uses: 4-byte BE size + CBOR {"payload": <floats>}."""
    payload = struct.pack(f"<{len(scores)}f", *scores)
    n = len(payload)
    head = bytes([0x40 | n]) if n < 24 else b"\x58" + bytes([n])
    body = b"\xa1\x67payload" + head + payload
    return struct.pack(">I", len(body)) + body


def fake_spleat(server):
    """Stream model output frames at 100 Hz to gesture_node, like the FPGA server."""
    while not state["stop"]:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        say("gesture_node connected to the fake FPGA model server")
        try:
            while not state["stop"]:
                with lock:
                    if state["until"] and time.monotonic() > state["until"]:
                        state["scores"], state["until"] = [0.0] * N, 0.0
                        say("gesture finished -> nobody in view")
                    scores = list(state["scores"])
                conn.sendall(cbor_frame(scores))
                time.sleep(1.0 / FRAME_HZ)
        except OSError:
            pass
        finally:
            conn.close()


def main():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(0.5)
    port = server.getsockname()[1]

    log = open("/tmp/sim_playground_pipeline.log", "w")
    sim = subprocess.Popen([get_package_prefix("robot_sim") + "/lib/robot_sim/robot_sim"],
                           start_new_session=True)
    pipe = subprocess.Popen(["ros2", "launch", "bringup", "pipeline.launch.py",
                             "despot:=false", f"port:={port}"],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    threading.Thread(target=fake_spleat, args=(server,), daemon=True).start()

    rclpy.init()
    node = rclpy.create_node("sim_playground")
    last_clear = [None]

    def on_request(m):
        a = m.header.identity.api_id
        say(f"ROBOT COMMAND -> api_id={a}  {ACTIONS.get(a, '?')}")

    def on_response(m):
        a, c = m.header.identity.api_id, m.header.status.code
        say(f"robot ack     <- api_id={a} code={c} ({'accepted' if c == 0 else 'rejected'})")

    def on_clearance(m):
        if m.state != last_clear[0]:
            last_clear[0] = m.state
            say(f"obstacle check: {CLEARANCE.get(m.state, m.state)}")

    node.create_subscription(Request, "/api/sport/request", on_request, 10)
    node.create_subscription(Response, "/api/sport/response", on_response, 10)
    node.create_subscription(Clearance, "/durance/clearance", on_clearance, 10)
    poses = node.create_publisher(PoseStamped, "/utlidar/robot_pose", 1)
    maps = node.create_publisher(HeightMap, "/utlidar/height_map_array", 1)

    def publish_lidar():
        grid = HeightMap(width=15, height=15, resolution=0.1, origin=[-0.7, -0.7], data=[0.0] * 225)
        if state["obstacle"]:
            grid.data[7 * 15 + 11] = 0.5   # 0.5 m tall box 0.4 m ahead
        poses.publish(PoseStamped())
        maps.publish(grid)

    node.create_timer(0.1, publish_lidar)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    print(f"\nModel outputs: {N} classes")
    for i, name in enumerate(LABELS):
        print(f"  {i:2d}  {name}")
    print("Commands: <number> = do gesture, n = nobody, o = toggle obstacle, q = quit")
    print("(pipeline log: /tmp/sim_playground_pipeline.log)\n", flush=True)

    try:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "q":
                break
            if cmd == "n":
                with lock:
                    state["scores"], state["until"] = [0.0] * N, 0.0
                say("nobody in view (all scores 0)")
            elif cmd == "o":
                state["obstacle"] = not state["obstacle"]
                say(f"obstacle {'PLACED' if state['obstacle'] else 'REMOVED'}")
            elif cmd.isdigit() and int(cmd) < N:
                i = int(cmd)
                scores = [0.0] * N
                scores[i] = 1.0
                with lock:
                    state["scores"], state["until"] = scores, time.monotonic() + GESTURE_SECONDS
                say(f"performing gesture {i} ({LABELS[i]}) for {GESTURE_SECONDS}s")
            elif cmd:
                print(f"unknown command {cmd!r}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        state["stop"] = True
        # SIGINT first (clean launch shutdown); SIGINT can be ignored when started
        # from a non-interactive shell, so escalate to SIGTERM, then SIGKILL.
        for sig, wait in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            alive = [p for p in (pipe, sim) if p.poll() is None]
            if not alive:
                break
            for p in alive:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
            for p in alive:
                try:
                    p.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    pass
        node.destroy_node()
        rclpy.try_shutdown()
        say("stopped")


if __name__ == "__main__":
    main()
