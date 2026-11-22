#!/bin/sh
# /root/switch_setup.sh NAME  - load firmware + server config + camera setting + weights of
# /root/setups/NAME (no arguments: list the setups). Inference is stopped first; start it again
# with /root/inference.sh start [--headless].
D=/root/setups/$1
if [ -z "$1" ] || [ ! -d "$D" ]; then echo "setups: $(ls /root/setups 2>/dev/null | tr '\n' ' ')"; exit 1; fi
for f in spleat.bit.bin spleat.dtbo config.yaml metavision_camera.json weights.txt model_classes; do
  [ -f "$D/$f" ] || { echo "$D/$f missing - nothing changed"; exit 1; }
done
/root/inference.sh stop >/dev/null 2>&1
systemctl stop spleat-server
cp "$D/spleat.bit.bin" "$D/spleat.dtbo" /usr/lib/firmware/xilinx/spleat/
cp "$D/config.yaml" /root/.config/spleat-server/config.yaml
cp "$D/metavision_camera.json" /root/metavision_camera.json
cp "$D/weights.txt" /root/weights.txt
cp "$D/model_classes" /root/model_classes
xmutil unloadapp >/dev/null 2>&1
sleep 1
xmutil loadapp spleat
sleep 2
systemctl start spleat-server
sleep 3
echo "setup $1 | input $(grep -E 'height|scaling' /root/.config/spleat-server/config.yaml | tr -s ' ' | tr '\n' ' ')| camera $(grep downsampling /root/metavision_camera.json | tr -d ' ,') | weights $(md5sum /root/weights.txt | cut -c1-12) | server $(systemctl is-active spleat-server) | app $(xmutil listapps 2>/dev/null | grep -c '0->0')"
