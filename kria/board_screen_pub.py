#!/usr/bin/env python3
"""Runs ON THE KRIA: captures the board's X screen (:0), where
`despot_send_and_forget --to_display 1` draws the live camera view, and publishes
it as a ROS 2 image so RViz on the laptop can show it next to the 3D Go2.

    source /root/laptop_link.sh
    python3 /root/board_screen_pub.py            # publishes /kria/camera_view
    python3 /root/board_screen_pub.py --snapshot /tmp/screen.jpg   # one frame, no ROS
"""
import ctypes
import sys
import time

import cv2
import numpy as np


class XImage(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int),
                ("format", ctypes.c_int), ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
                ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int),
                ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int),
                ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int)]


class Screen:
    def __init__(self, display=b":0"):
        x = ctypes.CDLL("libX11.so.6")
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XDefaultRootWindow.restype = ctypes.c_ulong
        x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x.XGetImage.restype = ctypes.POINTER(XImage)
        x.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                                ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
        x.XDestroyImage.argtypes = [ctypes.POINTER(XImage)]
        x.XDisplayWidth.argtypes = x.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.x = x
        self.d = x.XOpenDisplay(display)
        if not self.d:
            raise RuntimeError("cannot open X display :0")
        self.root = x.XDefaultRootWindow(self.d)
        self.w, self.h = x.XDisplayWidth(self.d, 0), x.XDisplayHeight(self.d, 0)

    def grab(self):
        img = self.x.XGetImage(self.d, self.root, 0, 0, self.w, self.h, 0xFFFFFFFF, 2)  # ZPixmap
        im = img.contents
        buf = ctypes.string_at(im.data, im.bytes_per_line * im.height)
        a = np.frombuffer(buf, np.uint8).reshape(im.height, im.bytes_per_line // 4, 4)
        frame = a[:, :im.width, :3].copy()          # BGRX -> BGR
        self.x.XDestroyImage(img)
        return frame


def main():
    scr = Screen()
    if len(sys.argv) > 2 and sys.argv[1] == "--snapshot":
        cv2.imwrite(sys.argv[2], scr.grab())
        print(f"saved {sys.argv[2]} ({scr.w}x{scr.h})")
        return

    import rclpy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("kria_screen_pub")
    pub = node.create_publisher(Image, "/kria/camera_view", 1)
    out_w = 320
    print(f"publishing board screen {scr.w}x{scr.h} -> /kria/camera_view ({out_w} px wide, 10 Hz)", flush=True)

    def tick():
        f = scr.grab()
        f = cv2.resize(f, (out_w, int(f.shape[0] * out_w / f.shape[1])), interpolation=cv2.INTER_AREA)
        m = Image()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = "kria_screen"
        m.height, m.width = f.shape[:2]
        m.encoding, m.is_bigendian, m.step = "bgr8", 0, f.shape[1] * 3
        m.data = f.tobytes()
        pub.publish(m)

    node.create_timer(0.1, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
