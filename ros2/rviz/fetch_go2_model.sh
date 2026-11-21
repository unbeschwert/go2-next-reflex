#!/bin/bash
# Downloads the official Unitree Go2 model (URDF + meshes, BSD-3-Clause, from
# github.com/unitreerobotics/unitree_ros) next to this script and writes go2.urdf
# with absolute mesh paths, as used by view_go2.launch.py. Not committed (~25 MB).
set -euo pipefail
cd "$(dirname "$0")"
B=https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/go2_description
mkdir -p meshes
curl -sfL -o go2_description.urdf "$B/urdf/go2_description.urdf"
for m in base calf calf_mirror foot hip thigh thigh_mirror; do
  curl -sfL -o "meshes/$m.dae" "$B/meshes/$m.dae"
done
sed "s#package://go2_description/dae/#file://$PWD/meshes/#g" go2_description.urdf > go2.urdf
echo "Go2 model ready: $PWD/go2.urdf"
