from launch import LaunchDescription

from launch_ros.actions import Node

from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # Moveit Config
    moveit_config = (
        MoveItConfigsBuilder(
            "ARC_S6",
            package_name="arc_robot_moveit_config"
        )
        .to_moveit_configs()
    )

    # Path to joint conversion converter config file
    converter_config = PathJoinSubstitution([
        FindPackageShare("unseen_sku_motion_planning"),
        "config",
        "joint_convention_converter_config.yaml",
    ])

    joint_convention_converter_node = Node(
        package="unseen_sku_motion_planning",
        executable="joint_convention_converter_node.py",
        name="joint_convention_converter",
        output="screen",
        parameters=[
            converter_config,
        ],
    )

    # Publish MOMA TF tree using converted joint states
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="moma_robot_state_publisher",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            {"ignore_timestamp": True},
        ],
        remappings=[("joint_states", "/unseen_sku/joint_states")],
    )

    return LaunchDescription([
        joint_convention_converter_node,
        robot_state_publisher,
    ])