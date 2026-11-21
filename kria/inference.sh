#!/bin/bash
# Gesture inference on the Kria KV260: camera -> SPLEAT SNN on the FPGA -> gesture
# decisions -> /api/sport/request (+ score bars and camera view for RViz on the laptop).
#
#   /root/inference.sh start [--headless]   start everything (after boot too)
#   /root/inference.sh stop                 stop camera feed, bridge, screen publisher
#   /root/inference.sh status               show what is running + latest results
#   /root/inference.sh robot --dry-run      real Go2: decide + log, never command
#   /root/inference.sh robot                real Go2: gestures move the robot
#
# --headless : camera feeds the FPGA without drawing on the board's screen
#              (no camera view in RViz, but less CPU).
# ROS 2 settings come from /root/laptop_link.sh (test domain 42, eth0 -> laptop).

source /root/laptop_link.sh

killp() {  # BusyBox has no pkill
  for p in $(ps | grep "$1" | grep -v grep | awk '{print $1}'); do kill "$p" 2>/dev/null; done
}

stop_all() {
  killp live_gesture_bridge.py
  killp board_screen_pub.py
  killp event_web_viewer.py
  killall despot_send_and_forget despot_debug_camera v4l2-ctl 2>/dev/null
  sleep 1
}

status() {
  echo "FPGA app : $(xmutil listapps 2>/dev/null | grep -q '0->0' && echo 'spleat loaded' || echo 'NOT loaded')"
  echo "SNN srv  : $(systemctl is-active spleat-server.service)"
  for p in despot_send_and_forget live_gesture_bridge.py board_screen_pub.py; do
    printf '%-9s: %s\n' "${p%%.*}" "$(ps | grep "$p" | grep -v grep >/dev/null && echo running || echo stopped)" | sed 's/despot_send_and_forget/camera  /;s/live_gesture_bridge/bridge  /;s/board_screen_pub/screen  /'
  done
  echo "--- latest inference:"
  grep rates /tmp/bridge.log 2>/dev/null | tail -n 1
  grep "ROBOT COMMAND" /tmp/bridge.log 2>/dev/null | tail -n 3
}

case "${1:-status}" in
  start)
    stop_all
    if ! xmutil listapps 2>/dev/null | grep -q '0->0'; then
      echo "loading FPGA app spleat..."
      xmutil unloadapp >/dev/null 2>&1; xmutil loadapp spleat
    fi
    systemctl is-active --quiet spleat-server.service || systemctl start spleat-server.service
    echo "camera warm-up (despot_test_camera)..."
    if ! despot_test_camera > /tmp/camtest.log 2>&1; then
      echo "camera test FAILED - see /tmp/camtest.log (reseat ribbon cable / reboot)"; exit 1
    fi
    if [ "${2:-}" = "--headless" ]; then
      setsid nohup despot_send_and_forget --to_display 0 > /tmp/despot.log 2>&1 < /dev/null &
    else
      DISPLAY=:0 SDL_VIDEODRIVER=x11 setsid nohup despot_send_and_forget --to_display 1 \
        > /tmp/despot.log 2>&1 < /dev/null &
      setsid nohup python3 /root/board_screen_pub.py > /tmp/screenpub.log 2>&1 < /dev/null &
    fi
    sleep 3
    setsid nohup python3 /root/live_gesture_bridge.py --host 127.0.0.1 > /tmp/bridge.log 2>&1 < /dev/null &
    # browser view of SNN input + outputs: http://192.168.77.1:8080 (passive, keeps running)
    ps | grep snn_input_viewer.py | grep -v grep >/dev/null || \
      setsid nohup python3 /root/snn_input_viewer.py 8080 > /tmp/snnview.log 2>&1 < /dev/null &
    sleep 4
    status
    echo "camera view: http://192.168.77.1:8080"
    ;;
  robot)
    # REAL Go2: keep camera -> FPGA running, swap the bridge onto the robot network
    # (domain 0, USB-Ethernet adapter) with the LiDAR obstacle check enabled.
    #   /root/inference.sh robot --dry-run   decide + log only, robot never commanded
    #   /root/inference.sh robot             live: gestures move the robot
    if ! ping -c 1 -W 1 192.168.123.161 >/dev/null 2>&1; then
      echo "Go2 not reachable at 192.168.123.161 - check USB-Ethernet adapter + cable"; exit 1
    fi
    ps | grep despot_send_and_forget | grep -v grep >/dev/null || { echo "camera feed not running - run '$0 start' first"; exit 1; }
    killp live_gesture_bridge.py
    killp board_screen_pub.py
    killp "obstacle_dete"; killp "ros2 run obstacle"
    sleep 1
    (
      source /root/robot_link.sh
      setsid nohup ros2 run obstacle_detection obstacle_detection > /tmp/obstacle.log 2>&1 < /dev/null &
      sleep 2
      setsid nohup python3 /root/live_gesture_bridge.py --host 127.0.0.1 --require-clear --no-viz \
        ${2:+$2} > /tmp/bridge.log 2>&1 < /dev/null &
    )
    sleep 6
    status
    echo "--- obstacle check:"; grep -E "Clear|Blocked|ready" /tmp/obstacle.log | tail -n 2
    grep -E "mode:|obstacle check" /tmp/bridge.log | tail -n 2
    ;;
  stop)   stop_all; killp "obstacle_dete"; killp "ros2 run obstacle"; status ;;
  status) status ;;
  *) echo "usage: $0 {start [--headless]|robot [--dry-run]|stop|status}"; exit 1 ;;
esac
