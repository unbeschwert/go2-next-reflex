#!/bin/sh
# Load a complete board setup and leave camera inference stopped.
set -eu

name=${1:-}
case "$name" in
  ''|[!a-zA-Z0-9]*|*[!a-zA-Z0-9_-]*) echo "usage: $0 SETUP_NAME" >&2; exit 2 ;;
esac

setup=/root/setups/$name
firmware=/usr/lib/firmware/xilinx/spleat
for file in spleat.bit.bin spleat.dtbo config.yaml metavision_camera.json weights.txt model_classes; do
  [ -s "$setup/$file" ] || { echo "$setup/$file is missing or empty" >&2; exit 1; }
done
[ -d "$firmware" ] && [ -d /root/.config/spleat-server ] ||
  { echo "SPLEAT firmware or server configuration directory is missing" >&2; exit 1; }

/root/inference.sh stop
systemctl stop spleat-server
apps=$(xmutil listapps)
if printf '%s\n' "$apps" | grep -q '0->[0-9]'; then
  xmutil unloadapp
  sleep 1
fi
cp "$setup/spleat.bit.bin" "$setup/spleat.dtbo" "$firmware/"
cp "$setup/config.yaml" /root/.config/spleat-server/config.yaml
cp "$setup/metavision_camera.json" /root/metavision_camera.json
cp "$setup/weights.txt" /root/weights.txt
cp "$setup/model_classes" /root/model_classes
xmutil loadapp spleat
sleep 2
apps=$(xmutil listapps)
printf '%s\n' "$apps" | awk '$1 == "spleat" && /0->[0-9]/ { loaded=1 } END { exit !loaded }' ||
  { echo "SPLEAT app did not load" >&2; exit 1; }
systemctl start spleat-server
sleep 3
systemctl is-active --quiet spleat-server
printf 'setup %s | server active | app 1\n' "$name"
