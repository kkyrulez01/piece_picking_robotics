"""Isolated, non-actuating ARC S6 / Pilz planning smoke test.

Starts only synthetic joint feedback, TF, MoveIt move_group and the active
PlanCandidate service. No Gazebo, ros2_control controller manager, robot
hardware driver, trajectory executor, or vacuum controller is started.

Run ONLY in an isolated ROS_DOMAIN_ID with the physical robot disconnected.
Joint and Cartesian limits below are demonstration-only, not hardware ratings.
"""

import os
import xml.etree.ElementTree as ET

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


JOINTS = [f"Revolute_{i}" for i in range(1, 7)]
HOME = [1.5707, 0.6, -1.0, -1.5707, -1.5707, 0.0]


def _planning_only_urdf() -> str:
    """Use the real robot geometry without either hardware plugin."""
    config_share = get_package_share_directory("arc_robot_moveit_config")
    urdf_path = os.path.join(config_share, "config", "ARC.urdf.xacro")
    root = ET.fromstring(xacro.process_file(urdf_path).toxml())

    for element in list(root):
        if element.tag in ("ros2_control", "gazebo"):
            root.remove(element)

    return ET.tostring(root, encoding="unicode")


def generate_launch_description():
    config = (
        MoveItConfigsBuilder("ARC", package_name="arc_robot_moveit_config")
        .planning_pipelines(
            default_planning_pipeline="pilz",
            pipelines=["pilz"],
        )
        .pilz_cartesian_limits()
        .to_moveit_configs()
    )

    # Identical, non-hardware robot model for all four test nodes.
    config.robot_description = {"robot_description": _planning_only_urdf()}

    # No point-cloud monitor in this first smoke test. Collision checking
    # therefore only includes the robot model, NOT the picking bin or parcels.
    config.sensors_3d = {}

    # Disable controller management. move_group execution is also disabled.
    config.trajectory_execution = {"moveit_manage_controllers": False}

    # TEST-ONLY joint dynamics: deliberately override source limits of
    # 100 rad/s and missing accelerations. These are NOT hardware limits.
    joint_limits = config.joint_limits["robot_description_planning"]["joint_limits"]
    for joint in JOINTS:
        joint_limits[joint].update(
            has_velocity_limits=True,
            max_velocity=0.5,
            has_acceleration_limits=True,
            max_acceleration=0.5,
            has_deceleration_limits=True,
            max_deceleration=-0.5,
        )

    # TEST-ONLY Cartesian limits for the Pilz LIN stages.
    cartesian_limits = config.pilz_cartesian_limits["robot_description_planning"]["cartesian_limits"]
    cartesian_limits.update(
        max_trans_vel=0.1,
        max_trans_acc=0.3,
        max_trans_dec=-0.3,
        max_rot_vel=0.5,
    )

    planner_share = get_package_share_directory("unseen_sku_motion_planning")
    planner_config = os.path.join(
        planner_share, "config", "motion_planner_core.yaml"
    )

    joint_feedback = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="planning_only_joint_states",
        output="screen",
        parameters=[
            config.robot_description,
            {
                "rate": 20,
                "publish_default_velocities": True,
                "use_sim_time": False,
                **{f"zeros.{joint}": pose for joint, pose in zip(JOINTS, HOME)},
            },
        ],
    )

    tf_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="planning_only_robot_state_publisher",
        output="screen",
        parameters=[config.robot_description, {"use_sim_time": False}],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            config.to_dict(),
            {
                "allow_trajectory_execution": False,
                "moveit_manage_controllers": False,
                "use_sim_time": False,
            },
        ],
    )

    candidate_planner = Node(
        package="unseen_sku_motion_planning",
        executable="motion_planner_node",
        name="unseen_sku_motion_planner_core",
        output="screen",
        parameters=[
            planner_config,
            config.robot_description,
            config.robot_description_semantic,
            config.robot_description_kinematics,
            config.joint_limits,
            {"use_sim_time": False},
        ],
    )

    return LaunchDescription([
        joint_feedback,
        tf_publisher,
        move_group,
        TimerAction(period=5.0, actions=[candidate_planner]),
    ])
