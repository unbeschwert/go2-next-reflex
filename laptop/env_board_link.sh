# Source on the LAPTOP to talk ROS 2 with the Kria board over the Ethernet cable.
#   source laptop/env_board_link.sh
# Board side:  ssh root@192.168.77.1  then  source /root/laptop_link.sh   (in bash, kria/laptop_link.sh)
# Test domain 42, eno1 only, unicast peers = board + laptop. Never reaches a robot.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # Jazzy + local CycloneDDS + workspace

export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
unset ROS_LOCALHOST_ONLY
export CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>30</MaxAutoParticipantIndex><Peers><Peer Address="192.168.77.1"/><Peer Address="192.168.77.2"/></Peers></Discovery></Domain></CycloneDDS>'
