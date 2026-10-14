import os

from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import (
    get_package_share_directory,
)

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder(
            "ARC",
            package_name="arc_robot_moveit_config",
        )
        .to_moveit_configs()
    )

    package_share = get_package_share_directory(
        "unseen_sku_motion_planning"
    )

    config_file = os.path.join(
        package_share,
        "config",
        "motion_planner_core.yaml",
    )

    motion_planner_core = Node(
        package="unseen_sku_motion_planning",
        executable="motion_planner_node",
        name="unseen_sku_motion_planner_core",
        output="screen",

        parameters=[
            config_file,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([
        motion_planner_core,
    ])