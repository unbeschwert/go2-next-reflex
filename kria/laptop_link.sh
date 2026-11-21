# Source on the Kria (bash) to talk ROS 2 with the laptop over eth0 (test domain 42).
export COLCON_PYTHON_EXECUTABLE=/usr/bin/python3
source /opt/ros/humble/setup.bash
source /opt/go2-current/local_setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"eth0\"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>30</MaxAutoParticipantIndex><Peers><Peer Address=\"192.168.77.1\"/><Peer Address=\"192.168.77.2\"/></Peers></Discovery></Domain></CycloneDDS>"
