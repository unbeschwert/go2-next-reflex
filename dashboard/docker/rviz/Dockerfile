# durance-rviz: RViz 2 (Jazzy) on a virtual screen, served over VNC on 127.0.0.1 only.
# The dashboard starts it with host networking (ROS 2 DDS on the simulator's loopback domain or the board
# link) and shows it in the Simulation page through its own authenticated WebSocket.
#   docker build -t durance-rviz:1 docker/rviz          (the dashboard's "Build RViz image" does this)
FROM ros:jazzy-ros-base-noble

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ros-jazzy-rviz2 ros-jazzy-robot-state-publisher ros-jazzy-rmw-cyclonedds-cpp \
      xvfb x11vnc matchbox-window-manager libgl1-mesa-dri fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh view_rviz.launch.py /opt/durance/
RUN chmod 755 /opt/durance/entrypoint.sh

ENV LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp HOME=/tmp GO2_VIZ=/go2_viz
ENTRYPOINT ["/opt/durance/entrypoint.sh"]
