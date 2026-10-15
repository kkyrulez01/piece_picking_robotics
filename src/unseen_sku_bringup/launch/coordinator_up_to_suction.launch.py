"""Standalone development bringup for coordinator-controlled perception.

Starts RealSense, Helios2, camera TF, SAM2, Mask Fusion, Sim-Suction,
and the Perception Coordinator. Does NOT include up_to_suction.launch.py.

Each Python executable can be overridden with a ROS launch argument or its
UNSEEN_SKU_*_PYTHON environment variable. This lets each developer keep
private virtual-environment names without editing shared launch files.

Perception remains coordinator-controlled. This launch does not start
MoveIt2, the Motion Coordinator, or the Candidate Planning Coordinator.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _default_python(workspace, venv_name, env_var):
    """Environment override with a backwards-compatible workspace default."""
    return EnvironmentVariable(
        env_var,
        default_value=PathJoinSubstitution([
            workspace, "venv", venv_name, "bin", "python"
        ]),
    )


def generate_launch_description():
    workspace = PathJoinSubstitution([
        EnvironmentVariable("HOME"), "unseen_sku_ws"
    ])

    camera_share = FindPackageShare("unseen_sku_camera")
    perception_share = FindPackageShare("unseen_sku_perception")
    coordinator_share = FindPackageShare("unseen_sku_coordinator")
    suction_share = FindPackageShare("unseen_sku_suction_scoring")

    launch_arguments = [
        DeclareLaunchArgument(
            "helios_config",
            default_value=PathJoinSubstitution([camera_share, "config", "helios2.yaml"]),
        ),
        DeclareLaunchArgument(
            "sam2_config",
            default_value=PathJoinSubstitution([perception_share, "config", "sam2.yaml"]),
        ),
        DeclareLaunchArgument(
            "fusion_config",
            default_value=PathJoinSubstitution([perception_share, "config", "mask_fusion.yaml"]),
        ),
        DeclareLaunchArgument(
            "suction_config",
            default_value=PathJoinSubstitution([
                workspace, "local_config", "sim_suction_config_local.yaml"
            ]),
        ),
        DeclareLaunchArgument(
            "candidate_transform_config",
            default_value=PathJoinSubstitution([
                suction_share, "config", "candidate_transform.yaml"
            ]),
        ),
        DeclareLaunchArgument(
            "coordinator_config",
            default_value=PathJoinSubstitution([
                coordinator_share, "config", "perception_coordinator.yaml"
            ]),
        ),
        DeclareLaunchArgument("initial_scene_index", default_value="1"),
        DeclareLaunchArgument(
            "helios_python",
            default_value=_default_python(workspace, "arena", "UNSEEN_SKU_HELIOS_PYTHON"),
        ),
        DeclareLaunchArgument(
            "sam2_python",
            default_value=_default_python(workspace, "sam2", "UNSEEN_SKU_SAM2_PYTHON"),
        ),
        DeclareLaunchArgument(
            "fusion_python",
            default_value=_default_python(workspace, "mask_fusion", "UNSEEN_SKU_FUSION_PYTHON"),
        ),
        DeclareLaunchArgument(
            "suction_python",
            default_value=_default_python(workspace, "sim_suction", "UNSEEN_SKU_SUCTION_PYTHON"),
        ),
    ]

    # The RealSense vendor launch remains included; the *other project*
    # up_to_suction.launch.py is no longer used.
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"
        ])),
        launch_arguments={
            "config_file": PathJoinSubstitution([camera_share, "config", "realsense.yaml"]),
        }.items(),
    )

    helios_node = Node(
        package="unseen_sku_camera",
        executable="helios2_node",
        name="helios2_node",
        output="screen",
        prefix=LaunchConfiguration("helios_python"),
        parameters=[{"config_path": LaunchConfiguration("helios_config")}],
    )

    camera_tf_node = Node(
        package="unseen_sku_camera",
        executable="camera_tf_node",
        name="camera_tf_node",
        output="screen",
    )

    sam2_node = Node(
        package="unseen_sku_perception",
        executable="sam2_node",
        name="sam2_node",
        output="screen",
        prefix=LaunchConfiguration("sam2_python"),
        parameters=[{
            "config_file": LaunchConfiguration("sam2_config"),
            "initial_scene_index": ParameterValue(
                LaunchConfiguration("initial_scene_index"), value_type=int
            ),
        }],
    )

    fusion_node = Node(
        package="unseen_sku_perception",
        executable="mask_fusion_node",
        name="mask_fusion_node",
        output="screen",
        prefix=LaunchConfiguration("fusion_python"),
        parameters=[
            LaunchConfiguration("fusion_config"),
            {"input_mode": "controlled"},
        ],
    )

    suction_node = Node(
        package="unseen_sku_suction_scoring",
        executable="suction_node",
        name="suction_node",
        output="screen",
        prefix=LaunchConfiguration("suction_python"),
        parameters=[{
            "config_path": LaunchConfiguration("suction_config"),
            "input_mode": "controlled",
        }],
    )

    # The upstream transformation operates on SuctionResult, not the raw
    # candidate topic. This preserves scene_id for Perception Coordinator.
    candidate_transform_node = Node(
        package="unseen_sku_suction_scoring",
        executable="candidate_transform_node",
        name="candidate_transform_node",
        output="screen",
        parameters=[
            LaunchConfiguration("candidate_transform_config"),
            {
                "message_type": "suction_result",
                "input_topic": "/sim_suction/result",
                "output_topic": "/sim_suction/result_base",
                "source_frame": "helios2_optical_frame",
                "target_frame": "base_link",
                "require_nonzero_stamp": True,
            },
        ],
    )

    coordinator_node = Node(
        package="unseen_sku_coordinator",
        executable="perception_coordinator",
        name="perception_coordinator",
        output="screen",
        parameters=[
            LaunchConfiguration("coordinator_config"),
            {"enable_suction_scoring": True},
        ],
    )

    return LaunchDescription(launch_arguments + [
        realsense_launch,
        helios_node,
        camera_tf_node,
        sam2_node,
        fusion_node,
        suction_node,
        candidate_transform_node,
        coordinator_node,
    ])
