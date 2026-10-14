from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time") # False when testing in real life

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock. False for the real ARC S6.",
    )

    # ---------------------------------------------------------
    # MoveIt configuration
    # ---------------------------------------------------------
    moveit_config = (MoveItConfigsBuilder(
        "ARC_S6",
        package_name="arc_robot_moveit_config"
    )
    .planning_pipelines(
        default_planning_pipeline="pilz",
        pipelines=["pilz", "ompl"]
    )
    .sensors_3d(file_path="config/sensors_real.yaml")
    .planning_scene_monitor(
        publish_planning_scene=True,
        publish_geometry_updates=True,
        publish_state_updates=True,
        publish_transforms_updates=True,
        publish_robot_description=True,
        publish_robot_description_semantic=True
    )
    .to_moveit_configs())

    # ---------------------------------------------------------
    # OctoMap configuration
    # ---------------------------------------------------------
    octomap_config = {
        "octomap_frame": "base_link",

        # 2cm voxels
        "octomap_resolution": 0.02
    }

    # ---------------------------------------------------------
    # Move Group
    # ---------------------------------------------------------
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            octomap_config,
            {"use_sim_time": use_sim_time},
        ],
        remappings=[("joint_states", "/moma/joint_states")]
    )
    
    # World -> base_link transform
    world_to_base_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_tf",
        output="log",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "world",
            "--child-frame-id", "base_link",
        ]
    )

    # Pick motion planner node
    pick_motion_planner_config = PathJoinSubstitution([
        FindPackageShare("arc_robot_moveit_config"),
        "config",
        "pick_motion_planner.yaml"
    ])

    pick_motion_planner_node = Node(
        package="moma_motion_planning",
        executable="pick_motion_planner",
        name="pick_motion_planner_node",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            pick_motion_planner_config,
            {
                "use_sim_time": use_sim_time,
            }
        ],
        remappings = [("joint_states", "/moma/joint_states")]
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare("arc_robot_moveit_config"),
        "config",
        "moma_moveit.rviz",
    ])

    moma_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            rviz_config,
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {
                "use_sim_time": use_sim_time,
            },
        ],
    )

    arc_trajectory_executor_node = Node(
        package="moma_motion_planning",
        executable="arc_trajectory_executor.py",
        name="arc_trajectory_executor",
        output="screen"
    )

    return LaunchDescription([
        declare_use_sim_time,
        world_to_base_tf,
        move_group_node,
        pick_motion_planner_node,
        moma_rviz_node,
        arc_trajectory_executor_node
    ])
