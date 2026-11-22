"""robot_state_publisher + RViz for the Go2 (no animator: the simulator / board test mode provides
joint states and TF). Used by the durance-rviz container and by the dashboard's desktop RViz button.

    GO2_VIZ=<folder with go2.urdf, meshes/, go2.rviz> ros2 launch view_rviz.launch.py
"""
import os
import re
import tempfile

from launch import LaunchDescription
from launch_ros.actions import Node

GO2 = os.path.expanduser(os.environ.get("GO2_VIZ", "/go2_viz"))
EMBED = os.environ.get("RVIZ_EMBED", "1") == "1"      # hide the side panels: the dashboard shows only the 3D view


def generate_launch_description():
    with open(os.path.join(GO2, "go2.urdf")) as f:
        urdf = f.read().replace("@MESHES@", os.path.join(GO2, "meshes"))
    # ros2/rviz/fetch_go2_model.sh writes absolute host paths: point them at the meshes next to this URDF
    urdf = re.sub(r'file://[^"]*?/meshes/', "file://" + os.path.join(GO2, "meshes") + "/", urdf)
    with open(os.path.join(GO2, "go2.rviz")) as f:
        cfg = f.read()
    if EMBED:
        # no side panels in the dashboard (RViz ignores "Hide Left Dock" at start-up); Panels menu re-adds them
        cfg = re.sub(r"^Panels:\n(?:[ -].*\n)*", "Panels: []\n", cfg, count=1, flags=re.M)
        cfg = re.sub(r"Hide Left Dock: \w+", "Hide Left Dock: true", cfg)
        # camera close to the dog (it turns in place at the origin); the mouse still orbits and zooms
        cfg = re.sub(r"(Class: rviz_default_plugins/Orbit\n\s+Distance: )[-\d.e]+", r"\g<1>3.2", cfg, count=1)
        cfg = re.sub(r"(Focal Point:\n\s+X: )[-\d.e]+(\n\s+Y: )[-\d.e]+(\n\s+Z: )[-\d.e]+", r"\g<1>0\g<2>0\g<3>0.15",
                     cfg, count=1)
        cfg = re.sub(r"Hide Right Dock: \w+", "Hide Right Dock: true", cfg)
        cfg = re.sub(r"(Window Geometry:\n(?:.*\n)*?\s+X: )\d+", r"\g<1>0", cfg)
        cfg = re.sub(r"(Window Geometry:\n(?:.*\n)*?\s+Y: )\d+", r"\g<1>0", cfg)
    out = os.path.join(tempfile.gettempdir(), "durance_go2.rviz")
    with open(out, "w") as f:
        f.write(cfg)
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}], output="log"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", out], output="log"),
    ])
