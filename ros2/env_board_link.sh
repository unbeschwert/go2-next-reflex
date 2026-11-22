# Source on the LAPTOP to talk ROS 2 with the Kria board over the Ethernet cable.
#   source ros2/env_board_link.sh
#   KRIA_IFACE=enp3s0 source ros2/env_board_link.sh   # if your wired port is not eno1
# Find your port with `ip -br link` (the one the cable to the Kria is plugged into).
# Board side:  ssh root@192.168.77.1  then  source /root/laptop_link.sh   (in bash, kria/laptop_link.sh)
# Test domain 42, that interface only, unicast peers = board + laptop. Never reaches a robot.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || return 1   # Jazzy or Humble + local CycloneDDS + workspace

KRIA_IFACE="${KRIA_IFACE:-eno1}"
if [ ! -e "/sys/class/net/$KRIA_IFACE" ]; then
  echo "env_board_link.sh: no network interface '$KRIA_IFACE' - set KRIA_IFACE (see: ip -br link)" >&2
fi

export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET     # Jazzy (ignored by Humble)
unset ROS_LOCALHOST_ONLY                        # Humble: allow the network
export CYCLONEDDS_URI="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"$KRIA_IFACE\"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>30</MaxAutoParticipantIndex><Peers><Peer Address=\"192.168.77.1\"/><Peer Address=\"192.168.77.2\"/></Peers></Discovery></Domain></CycloneDDS>"
