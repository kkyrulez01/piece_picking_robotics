#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from joint_conversion import (
    arc_to_moma_position, 
    moma_to_arc_position,
    arc_to_moma_derivative,
    moma_to_arc_derivative)

class JointConventionConverter(Node):
    def __init__(self):
        super().__init__("joint_convention_converter")

        # Parameters
        self.declare_parameter("input_joint_states_topic", "/joint_states")
        self.declare_parameter("output_joint_states_topic", "/moma/joint_states")

        self.declare_parameter("input_trajectory_topic", "/moma/planned_joint_trajectory")
        self.declare_parameter("output_trajectory_topic", "/moma/ARC_joint_trajectory")

        self.declare_parameter("arc_joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ])

        self.declare_parameter("moma_joint_names",
            [
                "Revolute_1",
                "Revolute_2",
                "Revolute_3",
                "Revolute_4",
                "Revolute_5",
                "Revolute_6",
            ])

        self.declare_parameter("signs",
            [
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
            ])

        self.declare_parameter("offsets",
            [
                0.0,
                math.pi / 2.0,
                0.0,
                math.pi / 2.0,
                0.0,
                0.0
            ])

        self.declare_parameter("wrap_moma_positions", True)

        self.declare_parameter("max_state_age_sec", 0.5)

        # Read parameters
        self.input_joint_states_topic = self.get_parameter("input_joint_states_topic").value
        self.output_joint_states_topic = self.get_parameter("output_joint_states_topic").value

        self.input_trajectory_topic = self.get_parameter("input_trajectory_topic").value
        self.output_trajectory_topic = self.get_parameter("output_trajectory_topic").value

        self.arc_joint_names = self.get_parameter("arc_joint_names").value
        self.moma_joint_names = self.get_parameter("moma_joint_names").value

        self.signs = list(self.get_parameter("signs").value)

        self.offsets = list(self.get_parameter("offsets").value)

        self.wrap_moma_positions = self.get_parameter("wrap_moma_positions").value
        self.max_state_age_sec = float(self.get_parameter("max_state_age_sec").value)

        # Validate configuration
        num_joints = len(self.arc_joint_names)

        if not (
            len(self.moma_joint_names) == num_joints
            and len(self.signs) == num_joints
            and len(self.offsets) == num_joints
        ):
            raise RuntimeError(
                "Joint converter parameter arrays must all "
                "have the same length."
            )

        if self.input_joint_states_topic == self.output_joint_states_topic:
            raise RuntimeError(
                "Input and output joint-state topics must be different "
                "to avoid a feedback loop."
            )

        # Current physical ARC robot state
        self.current_arc_positions = {}

        self.last_arc_state_rx_ns = None

        # ROS Publishers and Subscribers
        # ARC -> MOMA
        self.moma_joint_state_publisher = self.create_publisher(
            JointState,
            self.output_joint_states_topic,
            10
        )

        self.arc_joint_state_subscriber = self.create_subscription(
            JointState,
            self.input_joint_states_topic,
            self.arc_joint_state_callback,
            10
        )

        # MOMA -> ARC
        self.arc_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            self.output_trajectory_topic,
            10
        )

        self.moma_trajectory_subscription = self.create_subscription(
            JointTrajectory,
            self.input_trajectory_topic,
            self.moma_trajectory_callback,
            10
        )

        # =====================================================
        # Startup
        # =====================================================

        self.get_logger().info(
            "Joint convention converter started."
        )

        self.get_logger().info(
            "ARC -> MOMA joint states: "
            f"{self.input_joint_states_topic} "
            f"-> {self.output_joint_states_topic}"
        )

        self.get_logger().info(
            "MOMA -> ARC trajectory: "
            f"{self.input_trajectory_topic} "
            f"-> {self.output_trajectory_topic}"
        )

        for i in range(num_joints):
            self.get_logger().info(
                f"{self.arc_joint_names[i]} "
                f"<-> {self.moma_joint_names[i]}: "
                f"MOMA = "
                f"{self.signs[i]:+.1f} * ARC "
                f"+ {self.offsets[i]:+.6f}"
            )

    # =========================================================
    # ARC state checks
    # =========================================================

    def arc_state_is_ready(self) -> bool:

        return all(
            joint_name in self.current_arc_positions
            for joint_name in self.arc_joint_names
        )

    def arc_state_is_fresh(self) -> bool:

        if self.last_arc_state_rx_ns is None:
            return False

        now_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        age_sec = (
            now_ns - self.last_arc_state_rx_ns
        ) / 1.0e9

        return age_sec <= self.max_state_age_sec
    
    # =========================================================
    # ARC -> MOMA joint-state conversion
    # =========================================================
    def arc_joint_state_callback(self, msg: JointState):

        joint_indices = {name: i for i, name in enumerate(msg.name)}

        missing_joints = [name for name in self.arc_joint_names if name not in joint_indices]

        if missing_joints:
            self.get_logger().warning(
                "Ignoring ARC JointState. Missing joints: "
                + ", ".join(missing_joints)
            )

            return
    
        output = JointState()

        output.header = msg.header
        output.name = list(self.moma_joint_names)

        output.position = []
        output.velocity = []
        output.effort = []

        # Convert joints
        for i, arc_name in enumerate(self.arc_joint_names):
            # Match naming in ARC robot with URDF file
            input_index = joint_indices[arc_name] 

            # Extract signs and offset
            sign = self.signs[i]
            offset = self.offsets[i]

            # Extract ARC position
            arc_position = msg.position[input_index]

            # Stores actual unwrapped ARC position
            self.current_arc_positions[arc_name] = arc_position

            # Calculate ARC -> MOMA position
            moma_position = arc_to_moma_position(
                arc_position=arc_position,
                sign=sign,
                offset=offset,
                wrap=self.wrap_moma_positions
            )

            output.position.append(moma_position)

            # Velocity
            output.velocity.append(arc_to_moma_derivative(
                value=msg.velocity[input_index],
                sign=sign
            ))

            # Effort
            output.effort.append(arc_to_moma_derivative(
                value=msg.effort[input_index],
                sign=sign
            ))

        self.last_arc_state_rx_ns = self.get_clock().now().nanoseconds

        # Publish converted joint states to MOMA/MoveIt
        self.moma_joint_state_publisher.publish(output) 

    # =========================================================
    # MOMA -> ARC trajectory conversion
    # =========================================================
    def moma_trajectory_callback(self, msg: JointTrajectory):
        if not self.arc_state_is_ready():
            self.get_logger().error(
                "Trajectory rejected: current ARC joint state is unavailable."
            )

            return

        if not self.arc_state_is_fresh():

            self.get_logger().error(
                "Trajectory rejected: current ARC joint state is stale."
            )

            return

        if len(msg.points) == 0:
            self.get_logger().warning(
                "Received empty MOMA trajectory."
            )

            return

        if len(msg.joint_names) == 0:
            self.get_logger().error(
                "Trajectory rejected: joint_names is empty."
            )

            return

        if len(set(msg.joint_names)) != len(msg.joint_names):
            self.get_logger().error(
                "Trajectory rejected: duplicate joint names."
            )

            return

        # Find MOMA joint indices from Joint Trajectory message from MoveIt
        moma_indices = {name: index for index, name in enumerate(msg.joint_names)}

        missing_joints = [name for name in self.moma_joint_names if name not in moma_indices]

        if missing_joints:
            self.get_logger().error(
                "Trajectory rejected. Missing MOMA joints: "
                + ", ".join(missing_joints)
            )
            return
        
        # Create ARC trajectory
        arc_trajectory = JointTrajectory()

        arc_trajectory.header = msg.header
        arc_trajectory.joint_names = list(self.arc_joint_names)

        # Reads current joint positions of ARC robot stored earlier
        reference_positions = [self.current_arc_positions[name] for name in self.arc_joint_names]

        # -----------------------------------------------------
        # Convert each trajectory point
        # -----------------------------------------------------

        # For each waypoint in MoveIt Trajectory
        for point_index, moma_point in enumerate(msg.points):
            num_input_joints = len(msg.joint_names)

            if len(moma_point.positions) != num_input_joints:

                self.get_logger().error(
                    "Trajectory rejected: "
                    f"point {point_index} has "
                    "incorrect positions array length."
                )

                return

            # Velocity, acceleration and effort are sometimes optional
            has_velocity = len(moma_point.velocities) > 0
            has_acceleration = len(moma_point.accelerations) > 0
            has_effort = len(moma_point.effort) > 0

            arc_point = JointTrajectoryPoint()

            arc_point.positions = []
            if has_velocity:
                arc_point.velocities = []
            if has_acceleration:
                arc_point.accelerations = []
            if has_effort:
                arc_point.effort = []

            # Conversion for each joint
            for i, moma_name in enumerate(self.moma_joint_names):
                input_index = moma_indices[moma_name]

                sign = self.signs[i]
                offset = self.offsets[i]

                # Position
                moma_position = moma_point.positions[input_index]

                arc_position = moma_to_arc_position(
                    moma_position=moma_position,
                    sign=sign,
                    offset=offset,
                    reference=(reference_positions[i])
                )

                arc_point.positions.append(arc_position)

                # Velocity
                if has_velocity:
                    arc_point.velocities.append(moma_to_arc_derivative(
                        value=moma_point.velocities[input_index],
                        sign=sign
                    ))

                # Acceleration
                if has_acceleration:
                    arc_point.accelerations.append(moma_to_arc_derivative(
                        value=moma_point.accelerations[input_index],
                        sign=sign
                    ))

                # Effort
                if has_effort:
                    arc_point.effort.append(moma_to_arc_derivative(
                        value=moma_point.effort[input_index],
                        sign=sign
                    ))

            arc_point.time_from_start = moma_point.time_from_start

            arc_trajectory.points.append(arc_point)

            # Next point is unwrapped relative to this point
            reference_positions = list(arc_point.positions)

        self.arc_trajectory_publisher.publish(arc_trajectory)
        self.get_logger().info(
            "Converted MOMA trajectory -> ARC: "
            f"{len(arc_trajectory.points)} points."
        )

def main(args=None):
    rclpy.init(args=args)
    node = JointConventionConverter()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
