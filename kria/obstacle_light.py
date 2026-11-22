#!/usr/bin/env python3
"""Light signal for the obstacle check: blinks while /durance/clearance reports OBSTACLE.

Runs ON THE KRIA next to obstacle_detection (robot network, ROS domain 0):
    source /root/robot_link.sh
    python3 /root/obstacle_light.py                    # Go2 head light (default)
    python3 /root/obstacle_light.py --led NAME         # also a board LED from /sys/class/leds/NAME
    python3 /root/obstacle_light.py --no-go2 --led NAME

  OBSTACLE  -> blinks fast (2.5 Hz)
  no data   -> dim, steady (obstacle check not running or not reporting)
  CLEAR     -> off

Go2 head light: sport-style request on /api/vui/request, api_id 1005 (SetBrightness),
parameter {"brightness": 0..10}. Read-only for the robot otherwise: it never commands motion.
On exit (Ctrl-C / SIGTERM) the light is switched off.
"""
import argparse
import json
import signal
import time

import rclpy
from msgs.msg import Clearance
from unitree_api.msg import Request

VUI_SET_BRIGHTNESS = 1005
BLINK_HZ = 2.5
STALE_S = 2.0          # no clearance message for this long = no data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--led", default="", help="board LED name in /sys/class/leds (optional)")
    ap.add_argument("--no-go2", action="store_true", help="do not use the Go2 head light")
    ap.add_argument("--bright", type=int, default=10, help="Go2 head light level when on (1-10)")
    a = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("obstacle_light")
    pub = None if a.no_go2 else node.create_publisher(Request, "/api/vui/request", 10)
    st = {"state": None, "time": 0.0}

    def on_clear(msg):
        st["state"], st["time"] = msg.state, time.monotonic()
    node.create_subscription(Clearance, "/durance/clearance", on_clear, 1)

    sent = {"go2": None, "led": None}

    def go2(level):
        if pub is None or sent["go2"] == level:
            return
        m = Request()
        m.header.identity.api_id = VUI_SET_BRIGHTNESS
        m.header.identity.id = int(time.time() * 1000) % (1 << 62)
        m.parameter = json.dumps({"brightness": int(level)})
        pub.publish(m)
        sent["go2"] = level

    def led(on):
        if not a.led or sent["led"] == on:
            return
        try:
            with open(f"/sys/class/leds/{a.led}/brightness", "w") as f:
                f.write("1" if on else "0")
            sent["led"] = on
        except OSError as e:
            node.get_logger().warning(f"LED {a.led}: {e}")
            a.led = ""

    last_mode = None

    def tick():
        nonlocal last_mode
        now = time.monotonic()
        fresh = now - st["time"] < STALE_S
        if fresh and st["state"] == Clearance.OBSTACLE:
            mode = "OBSTACLE: blinking"
            on = int(now * BLINK_HZ * 2) % 2 == 0
            go2(a.bright if on else 0)
            led(on)
        elif fresh and st["state"] == Clearance.CLEAR:
            mode = "CLEAR: off"
            go2(0)
            led(False)
        else:
            mode = "no data: dim"
            go2(2)
            led(True)
        if mode != last_mode:
            print(time.strftime("[%H:%M:%S]"), "light", mode, flush=True)
            last_mode = mode

    node.create_timer(0.05, tick)

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        sent["go2"] = sent["led"] = None
        go2(0)
        led(False)
        time.sleep(0.2)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
