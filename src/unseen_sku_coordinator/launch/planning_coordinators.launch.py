"""Start the two independent planning-only coordinator action servers.

Requires a separate C++ /unseen_sku/plan_candidate service (MoveIt2).
Does not start perception, MoveIt2, controllers, hardware, or execution.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    base_frame = LaunchConfiguration("base_frame")
    max_candidate_attempts = LaunchConfiguration("max_candidate_attempts")
    max_candidate_age_sec = LaunchConfiguration("max_candidate_age_sec")

    return LaunchDescription([
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("max_candidate_attempts", default_value="5"),
        DeclareLaunchArgument("max_candidate_age_sec", default_value="30.0"),
        Node(
            package="unseen_sku_coordinator",
            executable="motion_coordinator_node",
            name="motion_coordinator_node",
            output="screen",
            parameters=[{
                "base_frame": base_frame,
                "max_candidate_age_sec": ParameterValue(
                    max_candidate_age_sec, value_type=float),
            }],
        ),
        Node(
            package="unseen_sku_coordinator",
            executable="candidate_planning_coordinator_node",
            name="candidate_planning_coordinator_node",
            output="screen",
            parameters=[{
                "max_candidate_attempts": ParameterValue(
                    max_candidate_attempts, value_type=int),
            }],
        ),
    ])
