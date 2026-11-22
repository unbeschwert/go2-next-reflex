#!/bin/bash
# Runs ON THE KRIA (sent by the dashboard over SSH: bash -s -- <folder>). Installs a complete SPLEAT setup
# from <folder>: firmware (spleat.bit.bin, spleat.dtbo), server config (config.yaml), camera config
# (metavision_camera.json), weights.txt and model_classes. Same steps as
# firmware_performance_testing/switch_setup.sh. Leaves inference stopped; prints the md5 of every installed
# file and "SETUP INSTALLED" at the end.
set -eu

src=${1:?usage: install_setup.sh <folder>}
fw=/usr/lib/firmware/xilinx/spleat
for f in spleat.bit.bin spleat.dtbo config.yaml metavision_camera.json weights.txt model_classes; do
  [ -s "$src/$f" ] || { echo "$src/$f is missing or empty" >&2; exit 1; }
done
[ -d "$fw" ] && [ -d /root/.config/spleat-server ] ||
  { echo "SPLEAT firmware or server configuration folder is missing" >&2; exit 1; }

bash /root/inference.sh stop >/dev/null 2>&1 || true
systemctl stop spleat-server.service || true
if xmutil listapps 2>/dev/null | grep -q '0->[0-9]'; then
  xmutil unloadapp
  sleep 1
fi
cp "$src/spleat.bit.bin" "$src/spleat.dtbo" "$fw/"
cp "$src/config.yaml" /root/.config/spleat-server/config.yaml
cp "$src/metavision_camera.json" /root/metavision_camera.json
cp "$src/weights.txt" /root/weights.txt
cp "$src/model_classes" /root/model_classes
sync
xmutil loadapp spleat
sleep 2
xmutil listapps | awk '$1 == "spleat" && /0->[0-9]/ { loaded = 1 } END { exit !loaded }' ||
  { echo "the SPLEAT app did not load" >&2; exit 1; }
systemctl reset-failed spleat-server.service 2>/dev/null || true
systemctl start spleat-server.service
sleep 3
systemctl is-active --quiet spleat-server.service || { echo "spleat-server did not start" >&2; exit 1; }
md5sum "$fw/spleat.bit.bin" "$fw/spleat.dtbo" /root/.config/spleat-server/config.yaml \
  /root/metavision_camera.json /root/weights.txt /root/model_classes
echo "SETUP INSTALLED"
