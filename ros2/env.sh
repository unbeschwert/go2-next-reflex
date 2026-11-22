# Source on the LAPTOP before building or running the Go2 pipeline locally:
#   source ros2/env.sh
#
# Works with ROS 2 Jazzy (Ubuntu 24.04) and ROS 2 Humble (Ubuntu 22.04); force one with ROS2_DISTRO=humble.
# All DDS traffic stays on loopback in domain 73, so laptop-only tests can never reach a real robot.
#
#   GO2_EXTRA  folder with CycloneDDS .deb packages unpacked without sudo (see doc/gesture-pipeline.md;
#              default ~/go2-hackathon/ros_extra/root, used only when it exists)
#   GO2_WS     colcon workspace with the go2-kria src    (default ~/go2-hackathon/laptop_ws)

if [ -z "${ROS2_DISTRO:-}" ]; then
  for d in jazzy humble; do
    if [ -f "/opt/ros/$d/setup.bash" ]; then ROS2_DISTRO=$d; break; fi
  done
fi
if [ -z "${ROS2_DISTRO:-}" ] || [ ! -f "/opt/ros/$ROS2_DISTRO/setup.bash" ]; then
  echo "env.sh: no ROS 2 Jazzy or Humble found in /opt/ros" >&2
  return 1 2>/dev/null || exit 1
fi
export ROS2_DISTRO
source "/opt/ros/$ROS2_DISTRO/setup.bash"

X="${GO2_EXTRA:-$HOME/go2-hackathon/ros_extra/root}"
if [ -d "$X/opt/ros/$ROS2_DISTRO" ]; then
  PYV=$(python3 -c 'import sys; print("python%d.%d" % sys.version_info[:2])')
  export CMAKE_PREFIX_PATH="$X/opt/ros/$ROS2_DISTRO:$X/usr:${CMAKE_PREFIX_PATH}"
  export AMENT_PREFIX_PATH="$X/opt/ros/$ROS2_DISTRO:${AMENT_PREFIX_PATH}"
  export LD_LIBRARY_PATH="$X/opt/ros/$ROS2_DISTRO/lib:$X/opt/ros/$ROS2_DISTRO/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
  export PATH="$X/opt/ros/$ROS2_DISTRO/bin:${PATH}"
  export PYTHONPATH="$X/opt/ros/$ROS2_DISTRO/lib/$PYV/site-packages:${PYTHONPATH}"
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=73
if [ "$ROS2_DISTRO" = "humble" ]; then
  export ROS_LOCALHOST_ONLY=1                      # Humble
else
  export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST   # Jazzy and newer
fi
export CYCLONEDDS_URI='<CycloneDDS><Domain Id="73"><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'

# Workspace overlay (exists after the first colcon build)
WS="${GO2_WS:-$HOME/go2-hackathon/laptop_ws}"
if [ -f "$WS/install/local_setup.bash" ]; then
  source "$WS/install/local_setup.bash"
fi
