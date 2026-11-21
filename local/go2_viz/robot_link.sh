# Source on the Kria (bash) to talk ROS 2 with the REAL Go2 over the USB-Ethernet
# adapter (192.168.123.99 -> Go2 at 192.168.123.161), as in the hackathon guide.
export COLCON_PYTHON_EXECUTABLE=/usr/bin/python3
source /opt/ros/humble/setup.bash
source /opt/go2-current/local_setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///etc/cyclonedds/cyclonedds-go2.xml
