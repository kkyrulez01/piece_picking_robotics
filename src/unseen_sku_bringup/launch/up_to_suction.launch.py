from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    EnvironmentVariable,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package directories
    perception_share = FindPackageShare("unseen_sku_perception")
    camera_share = FindPackageShare("unseen_sku_camera")
    suction_share = FindPackageShare("unseen_sku_suction_scoring")

    # Config files
    realsense_config = PathJoinSubstitution([
        FindPackageShare("unseen_sku_camera"),
        "config",
        "realsense.yaml",
    ])
    helios_config = LaunchConfiguration("helios_config")
    sam2_config = LaunchConfiguration("sam2_config")
    fusion_config = LaunchConfiguration("fusion_config")
    suction_config = LaunchConfiguration("suction_config")
    candidate_transform_config = LaunchConfiguration("candidate_transform_config")
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
        # NEW: Optional SAM2 virtual environment
        prefix=LaunchConfiguration("sam2_python"),
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

        parameters=[
            fusion_config,
        ],
    )

    # NEW: Launch Sim-Suction node
    suction_node = Node(
        package="unseen_sku_suction_scoring",
        executable="suction_node",
        name="suction_node",
        output="screen",

        # Optional Sim-Suction virtual environment
        prefix=LaunchConfiguration("suction_python"),

        parameters=[{
            "config_path": suction_config,
        }],
    )

    # Candidate transform node
    candidate_transform_node = Node(
        package="unseen_sku_suction_scoring",
        executable="candidate_transform_node",
        name="candidate_transform_node",
        output="screen",

        parameters=[
            candidate_transform_config,
        ],
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
        # NEW: Sim-Suction configuration
        DeclareLaunchArgument(
            "suction_config",
            default_value=PathJoinSubstitution([
                suction_share,
                "config",
                "sim_suction_config.yaml",
            ])
        ),
        # Candidate Transform configuration
        DeclareLaunchArgument(
            "candidate_transform_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("unseen_sku_suction_scoring"),
                "config",
                "candidate_transform.yaml",
            ]),
        ),
        # Python environment for SAM2
        DeclareLaunchArgument(
            "sam2_python",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "unseen_sku_ws",
                "venv",
                "sam2",
                "bin",
                "python",
            ]),
        ),
        # Python environment for Sim-Suction
        DeclareLaunchArgument(
            "suction_python",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "unseen_sku_ws",
                "venv",
                "sim_suction_venv",
                "bin",
                "python",
            ]),
        ),

        # Nodes
        realsense_launch,
        helios_node,
        camera_tf_node,
        sam2_node,
        mask_fusion_node,
        suction_node,
        candidate_transform_node
    ])