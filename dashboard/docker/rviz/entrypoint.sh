#!/bin/bash
# Virtual screen + window manager (keeps RViz full screen) + VNC on 127.0.0.1 + RViz.
set -euo pipefail
: "${VNC_PASSWORD:?VNC_PASSWORD must be set}"
VNC_PORT=${VNC_PORT:-5901}
WIDTH=${WIDTH:-1600}
HEIGHT=${HEIGHT:-900}
export DISPLAY=:77

Xvfb :77 -screen 0 "${WIDTH}x${HEIGHT}x24" -nolisten tcp -nolisten local >/tmp/xvfb.log 2>&1 &
for _ in $(seq 100); do [ -e /tmp/.X11-unix/X77 ] && break; sleep 0.1; done
[ -e /tmp/.X11-unix/X77 ] || { echo "Xvfb did not start:"; cat /tmp/xvfb.log; exit 1; }
matchbox-window-manager -use_titlebar no -use_cursor no >/tmp/wm.log 2>&1 &

mkdir -p /tmp/vnc
x11vnc -storepasswd "$VNC_PASSWORD" /tmp/vnc/passwd >/dev/null 2>&1
unset VNC_PASSWORD
x11vnc -display :77 -rfbport "$VNC_PORT" -localhost -rfbauth /tmp/vnc/passwd -forever -shared \
       -noxdamage -nowf -quiet -o /tmp/x11vnc.log &

set +u                                  # ROS setup scripts read unset variables
source /opt/ros/jazzy/setup.bash
exec ros2 launch /opt/durance/view_rviz.launch.py
