from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package directories
    perception_share = FindPackageShare("unseen_sku_perception")
    camera_share = FindPackageShare("unseen_sku_camera")

    # Config files
    realsense_config = PathJoinSubstitution([
        FindPackageShare("unseen_sku_camera"),
        "config",
        "realsense.yaml",
    ])
    helios_config = LaunchConfiguration("helios_config")
    sam2_config = LaunchConfiguration("sam2_config")
    fusion_config = LaunchConfiguration("fusion_config")

    # Launch RealSense
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch",
                "rs_launch.py",
            ])
        ),

        launch_arguments={
            "config_file": realsense_config,
        }.items()
    )

    # Launch Helios2 node
    helios_node = Node(
        package="unseen_sku_camera",
        executable="helios2_node",
        name="helios2_node",
        output="screen",

        parameters=[{
            "config_path": helios_config,
        }],
    )

    camera_tf_node = Node(
        package="unseen_sku_camera",
        executable="camera_tf_node",
        name="camera_tf_node",
        output="screen",
    )

    # Launch SAM2 node
    sam2_node = Node(
        package="unseen_sku_perception",
        executable="sam2_node",
        name="sam2_node",
        output="screen",

        parameters=[{
            "config_file": sam2_config,
            "initial_scene_index": ParameterValue(
                LaunchConfiguration("initial_scene_index"),
                value_type=int,
            ),
        }],
    )

    # Launch Mask Fusion node
    mask_fusion_node = Node(
        package="unseen_sku_perception",
        executable="mask_fusion_node",
        name="mask_fusion_node",
        output="screen",

        parameters=[fusion_config],
    )

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            "helios_config",
            default_value=PathJoinSubstitution([
                camera_share,
                "config",
                "helios2.yaml",
            ])
        ),
        DeclareLaunchArgument(
            "sam2_config",
            default_value=PathJoinSubstitution([
                perception_share,
                "config",
                "sam2.yaml",
            ])
        ),
        DeclareLaunchArgument(
            "fusion_config",
            default_value=PathJoinSubstitution([
                perception_share,
                "config",
                "mask_fusion.yaml",
            ])
        ),
        DeclareLaunchArgument(
            "initial_scene_index",
            default_value="1",
        ),

        # Nodes
        realsense_launch,
        helios_node,
        camera_tf_node,
        sam2_node,
        mask_fusion_node,
    ])