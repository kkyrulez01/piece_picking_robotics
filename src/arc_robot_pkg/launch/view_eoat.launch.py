import os

from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import xacro


def generate_launch_description():

    package_name = "arc_robot_pkg"
    pkg_path = get_package_share_directory(package_name)

    # Main ARC S6 Xacro
    xacro_file = os.path.join(
        pkg_path,
        "ARC_S6",
        "ARC.urdf.xacro"
    )

    # Existing RViz configuration
    rviz_config = os.path.join(
        pkg_path,
        "config",
        "robot_model.rviz"
    )

    # Generate complete robot description
    robot_description = xacro.process_file(
        xacro_file
    ).toxml()

    # Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True
        }]
    )

    # Joint State Publisher GUI
    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True
        }]
    )

    # RViz2
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            rviz_config
        ],
        output="screen",
        parameters=[{
            "use_sim_time": False
        }]
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz_node
    ])