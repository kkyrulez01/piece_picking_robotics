"""Launch the standalone Perception Coordinator."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    package_share = Path(
        get_package_share_directory("unseen_sku_coordinator")
    )

    config_file = (
        package_share
        / "config"
        / "perception_coordinator.yaml"
    )

    coordinator_node = Node(
        package="unseen_sku_coordinator",
        executable="perception_coordinator",
        name="perception_coordinator",
        output="screen",
        parameters=[str(config_file)],
    )

    return LaunchDescription([
        coordinator_node,
    ])