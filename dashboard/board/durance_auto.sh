#!/bin/bash
# Durance AI "SD card mode": at power-on, start camera -> FPGA SNN -> gesture bridge without the laptop.
# Installed by the dashboard (Board & Auto mode page) together with durance-auto.service and
# /root/durance_auto.conf (KEY=value lines, read here, never sourced).
#
# Switch it off without the laptop: create /root/DURANCE_AUTO_DISABLE (e.g. with the SD card in a
# card reader), or run `systemctl disable durance-auto.service` on the board.
# Log: /root/durance_auto.log
set -u
LOG=/root/durance_auto.log
CONF=/root/durance_auto.conf
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 1000000 ]; then mv -f "$LOG" "$LOG.1"; fi
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') durance_auto ${1:-start}"

get() { grep -E "^$1=[A-Za-z0-9_.-]*$" "$CONF" 2>/dev/null | tail -n 1 | cut -d= -f2; }

case "${1:-start}" in
  stop) bash /root/inference.sh stop; exit 0 ;;
  start) ;;
  *) echo "usage: $0 start|stop"; exit 2 ;;
esac
if [ -e /root/DURANCE_AUTO_DISABLE ]; then echo "disabled by /root/DURANCE_AUTO_DISABLE"; exit 0; fi

MODE=$(get MODE); MODE=${MODE:-robot}
CLASSES=$(get CLASSES)
START_DELAY_S=$(get START_DELAY_S)
WAIT_GO2_S=$(get WAIT_GO2_S)
KEEP_PAUSE=$(get KEEP_PAUSE)
case "$MODE" in robot|robot-dry|test) ;; *) echo "bad MODE '$MODE' in $CONF"; exit 1 ;; esac
case "$CLASSES" in own|dataset|"") ;; *) echo "bad CLASSES '$CLASSES' in $CONF"; exit 1 ;; esac
[[ "$START_DELAY_S" =~ ^[0-9]+$ ]] || START_DELAY_S=20
[[ "$WAIT_GO2_S" =~ ^[0-9]+$ ]] || WAIT_GO2_S=120
if [ -n "$CLASSES" ]; then echo "$CLASSES" > /root/model_classes; fi
echo "mode $MODE, classes $(cat /root/model_classes 2>/dev/null), settling for ${START_DELAY_S} s"
sleep "$START_DELAY_S"

# A pause left over from the last session (E-STOP, manual mode) would make the dog ignore every
# gesture, and without the laptop nobody could lift it. Keep it only when KEEP_PAUSE=1.
if [ "${KEEP_PAUSE:-0}" != 1 ] && [ -f /root/bridge_settings.json ]; then
  python3 - <<'PY'
import json, os
p = "/root/bridge_settings.json"
try:
    with open(p) as f:
        d = json.load(f)
    if d.pop("paused", None):
        with open(p + ".tmp", "w") as f:
            json.dump(d, f)
        os.replace(p + ".tmp", p)
        print("cleared 'paused' from the last session")
except Exception as e:
    print("bridge settings not changed:", e)
PY
fi

ok=0
for i in 1 2 3; do
  if bash /root/inference.sh start --headless; then ok=1; break; fi
  echo "inference start failed (try $i/3), retrying in 10 s"
  sleep 10
done
if [ $ok != 1 ]; then echo "giving up: camera or FPGA did not start"; exit 1; fi

# The camera client sometimes runs but sends nothing after a start (seen 2026-11-22: 0 frames, the
# bridge reconnecting every 4 s; 'Error TimeHigh discrepancy' in /tmp/despot.log). A still scene still
# gives ~10 frames/s, so 0 frames in 5 s means a dead feed: start once or twice more.
frames() { python3 /root/snn_record.py 5 /tmp/durance_auto_frames.npz 2>/dev/null | sed -n 's/.*: \([0-9]*\) input frames.*/\1/p' | tail -n 1; }
for i in 1 2; do
  n=$(frames); n=${n:-0}
  echo "camera check: $n frames in 5 s"
  [ "$n" -gt 0 ] && break
  echo "no camera frames: restarting the camera feed (try $i/2)"
  bash /root/inference.sh start --headless || true
done
if [ "$MODE" = test ]; then echo "=== running in test mode"; exit 0; fi

if [ ! -f /root/robot_link.sh ]; then
  echo "/root/robot_link.sh missing: staying in test mode (no robot commands)"; exit 0
fi
echo "waiting up to ${WAIT_GO2_S} s for the Go2 at 192.168.123.161"
t=0
until ping -c 1 -W 1 192.168.123.161 >/dev/null 2>&1; do
  t=$((t + 2))
  if [ $t -ge "$WAIT_GO2_S" ]; then echo "Go2 not reachable: staying in test mode (no robot commands)"; exit 0; fi
  sleep 1
done
if [ "$MODE" = robot-dry ]; then bash /root/inference.sh robot --dry-run; else bash /root/inference.sh robot; fi
echo "=== started ($MODE)"
