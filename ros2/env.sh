# Source on the LAPTOP before building or running the Go2 pipeline locally:
#   source ros2/env.sh
#
# ROS 2 Jazzy + CycloneDDS packages unpacked without sudo (see doc/gesture-pipeline.md),
# and all DDS traffic kept on loopback in domain 73, so laptop-only tests can never
# reach a real robot on the network.
#
#   GO2_EXTRA  folder with the unpacked .deb packages   (default ~/go2-hackathon/ros_extra/root)
#   GO2_WS     colcon workspace with the go2-kria src    (default ~/go2-hackathon/laptop_ws)

source /opt/ros/jazzy/setup.bash

X="${GO2_EXTRA:-$HOME/go2-hackathon/ros_extra/root}"
export CMAKE_PREFIX_PATH="$X/opt/ros/jazzy:$X/usr:${CMAKE_PREFIX_PATH}"
export AMENT_PREFIX_PATH="$X/opt/ros/jazzy:${AMENT_PREFIX_PATH}"
export LD_LIBRARY_PATH="$X/opt/ros/jazzy/lib:$X/opt/ros/jazzy/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export PATH="$X/opt/ros/jazzy/bin:${PATH}"
export PYTHONPATH="$X/opt/ros/jazzy/lib/python3.12/site-packages:${PYTHONPATH}"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export CYCLONEDDS_URI='<CycloneDDS><Domain Id="73"><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'

# Workspace overlay (exists after the first colcon build)
WS="${GO2_WS:-$HOME/go2-hackathon/laptop_ws}"
if [ -f "$WS/install/local_setup.bash" ]; then
  source "$WS/install/local_setup.bash"
fi
