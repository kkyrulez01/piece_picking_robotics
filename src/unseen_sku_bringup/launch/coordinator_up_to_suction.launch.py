"""Launch the complete Coordinator-controlled perception pipeline."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch.substitutions import (
    EnvironmentVariable,
    PathJoinSubstitution,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    workspace = PathJoinSubstitution([
        EnvironmentVariable("HOME"),
        "unseen_sku_ws",
    ])

    # ------------------------------------------
    # Cameras, SAM2, Mask Fusion, Sim-Suction
    # ------------------------------------------

    perception_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("unseen_sku_bringup"),
                "launch",
                "up_to_suction.launch.py",
            ])
        ),
        launch_arguments={
            "fusion_input_mode": "controlled",
            "suction_input_mode": "controlled",

            "suction_config": PathJoinSubstitution([
                workspace,
                "local_config",
                "sim_suction_config_local.yaml",
            ]),

            "suction_python": PathJoinSubstitution([
                workspace,
                "venv",
                "sim_suction",
                "bin",
                "python",
            ]),
        }.items(),
    )

    # ------------------------------------------
    # Perception Coordinator
    # ------------------------------------------

    coordinator_config = PathJoinSubstitution([
        FindPackageShare("unseen_sku_coordinator"),
        "config",
        "perception_coordinator.yaml",
    ])

    coordinator_node = Node(
        package="unseen_sku_coordinator",
        executable="perception_coordinator",
        name="perception_coordinator",
        output="screen",

        parameters=[
            coordinator_config,

            # Enable suction scoring only for this bringup.
            # The base YAML remains fusion-only.
            {
                "enable_suction_scoring": True,
            },
        ],
    )

    return LaunchDescription([
        perception_pipeline,
        coordinator_node,
    ])