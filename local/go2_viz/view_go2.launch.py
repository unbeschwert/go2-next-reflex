"""3D view of the Go2 reacting to pipeline commands.

    source ~/go2-hackathon/laptop_ws/env.sh
    ros2 launch ~/go2-hackathon/go2_viz/view_go2.launch.py
"""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    with open(os.path.join(HERE, "go2.urdf")) as f:
        urdf = f.read()
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}], output="log"),
        ExecuteProcess(cmd=["python3", os.path.join(HERE, "go2_animator.py")], output="screen"),
        Node(package="rviz2", executable="rviz2",
             arguments=["-d", os.path.join(HERE, "go2.rviz")], output="log"),
    ])
