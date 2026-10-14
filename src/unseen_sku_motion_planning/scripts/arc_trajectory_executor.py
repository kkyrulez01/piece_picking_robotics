#!/usr/bin/env python3

import copy
from std_msgs.msg import Bool, Empty

import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from std_srvs.srv import Trigger

class ArcTrajectoryExecutor(Node):

    def __init__(self):
        super().__init__("arc_trajectory_executor")

        # Execution mode
        self.execution_mode = self.declare_parameter(
            "execution_mode",
            "manual"
        ).get_parameter_value().string_value.lower()

        if self.execution_mode not in ("manual", "auto"):
            raise ValueError("execution_mode must be either 'manual' or 'auto'.")

        # Expected physical ARC joint order
        self.arc_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Stored states
        self.latest_trajectory = None
        self.current_joint_positions = {}

        self.execution_in_progress = False

        # # Safety limits
        # self.max_start_error_rad = 1.57
        # self.max_first_test_motion_rad = 1.57

        # Publish execution result
        self.execution_result_publisher = self.create_publisher(
            Bool,
            "/moma/arc_trajectory_execution_result",
            10
        )

        # Subscribers
        # Subscribe to converted ARC trajectory
        self.trajectory_subscriber = self.create_subscription(
            JointTrajectory,
            "/moma/ARC_joint_trajectory",
            self.trajectory_callback,
            10
        )

        # Subscribe to live physical robot state
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        # Manual execution trigger
        self.execute_service = self.create_service(
            Trigger,
            "/moma/execute_latest_arc_trajectory",
            self.execute_callback
        )

        # Real ARC trajectory controller
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )

        # Discard path
        self.discard_service = self.create_service(
            Trigger,
            "/moma/discard_latest_arc_trajectory",
            self.discard_callback,
        )

        self.get_logger().info(
            "ARC trajectory executor ready."
        )
        self.get_logger().warn(
            f"Execution mode: {self.execution_mode.upper()}"
        )

        if self.execution_mode == "auto":
            self.get_logger().warn(
                "AUTO mode: incoming ARC trajectories "
                "will execute automatically."
            )

        else:
            self.get_logger().warn(
                "MANUAL mode: incoming ARC trajectories "
                "will be stored but NOT executed."
            )

            self.get_logger().warn(
                "Inspect the trajectory in RViz, then call "
                "/moma/execute_latest_arc_trajectory."
            )

    # ==============================================================
    # JOINT STATES
    # ==============================================================
    def joint_state_callback(self, message):
        self.current_joint_positions = {name: position for name, position in zip(message.name, message.position)}

    # ==============================================================
    # TRAJECTORY RECEIVED
    # ==============================================================
    def trajectory_callback(self, message):
        if self.execution_in_progress:
            self.get_logger().warn(
                "Execution currently in progress. Ignoring incoming trajectory."
            )
            return

        if not message.points:
            self.get_logger().error("Received empty trajectory. Not storing it.")
            return

        # Store latest converted ARC trajectory
        self.latest_trajectory = copy.deepcopy(message)

        self.get_logger().info(
            f"Stored ARC trajectory containing "
            f"{len(message.points)} points."
        )

        # AUTO MODE
        if self.execution_mode == "auto":
            self.get_logger().warn("AUTO mode: executing stored trajectory.")

            success, message_text = self.start_execution(self.latest_trajectory)

            if not success:
                self.get_logger().error(
                    f"Unable to start trajectory execution: "
                    f"{message_text}"
                )

                self.publish_execution_result(False)

            return

        # MANUAL MODE
        self.get_logger().warn(
            "MANUAL mode: trajectory stored."
        )

        self.get_logger().warn(
            "Robot will NOT move until execution is triggered."
        )

        self.get_logger().warn(
            "Inspect trajectory in RViz, then execute:"
        )

        self.get_logger().warn(
            "ros2 service call "
            "/moma/execute_latest_arc_trajectory "
            "std_srvs/srv/Trigger \"{}\""
        )

    # ==============================================================
    # MANUAL EXECUTION TRIGGER
    # ==============================================================
    def execute_callback(self, request, response):
        if self.execution_in_progress:
            response.success = False
            response.message = "Execution already in progress."
            return response

        if self.latest_trajectory is None:
            response.success = False
            response.message = "No stored ARC trajectory."
            return response

        self.get_logger().warn("Manual execution trigger received.")

        success, message = self.start_execution(self.latest_trajectory)

        response.success = success
        response.message = message

        return response

    # ==============================================================
    # START PHYSICAL EXECUTION
    # ==============================================================
    def start_execution(self, trajectory):
        # Prevent overlapping goals
        if self.execution_in_progress:
            self.get_logger().error(
            "Cannot execute trajectory: another trajectory is already running."
        )
            return False, "Execution already in progress."

        # Validate trajectory
        if trajectory is None:
            return False, "No trajectory provided."
        
        if not trajectory.points:
            return False, "Empty trajectory."

        # Check first trajectory input
        first_point = trajectory.points[0]
        
        if len(first_point.positions) != len(self.arc_joint_names):
            return (False, 
                    f"First trajectory point does not contain "
                    f"{len(self.arc_joint_names)} positions.")

        max_start_error = 0.0
        for index, joint_name in enumerate(self.arc_joint_names):
            current_position = self.current_joint_positions[joint_name]
            trajectory_start = first_point.positions[index]

            start_error = abs(
                trajectory_start - current_position
            )

            max_start_error = max(
                max_start_error,
                start_error,
            )

            self.get_logger().warn(
                f"{joint_name}: "
                f"current={current_position:.4f}, "
                f"start={trajectory_start:.4f}, "
                f"error={start_error:.4f} rad"
            )

        self.get_logger().warn(
            f"Maximum start error = "
            f"{max_start_error:.4f} rad"
        )

        # Check final trajectory point
        final_point = trajectory.points[-1]

        if len(final_point.positions) != len(self.arc_joint_names):
            return (
                False,
                f"Final trajectory point does not contain "
                f"{len(self.arc_joint_names)} positions."
            )

        max_motion = 0.0

        for index, joint_name in enumerate(self.arc_joint_names):
            current_position = self.current_joint_positions[joint_name]
            final_position = final_point.positions[index]

            motion = abs(final_position - current_position)

            max_motion = max(max_motion, motion)

        # Check if Robot Controller exists
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            return (
                False,
                "scaled_joint_trajectory_controller "
                "FollowJointTrajectory server unavailable"
            )

        # Build FollowJointTrajectory goal
        goal = FollowJointTrajectory.Goal()

        goal.trajectory = copy.deepcopy(trajectory)

        # Physical execution
        self.execution_in_progress = True

        self.get_logger().warn("SENDING TRAJECTORY TO PHYSICAL ARC ROBOT")

        future = self.action_client.send_goal_async(goal)

        future.add_done_callback(self.goal_response_callback)

        return (
            True,
            f"Trajectory submitted. "
            f"Max motion={max_motion:.4f} rad."
        )

    # ==============================================================
    # GOAL RESPONSE
    # ==============================================================
    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()

        except Exception as exception:
            self.get_logger().error(f"Failed to send trajectory: {exception}")
            self.execution_in_progress = False
            self.publish_execution_result(False)

            return

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal REJECTED by controller.")

            self.execution_in_progress = False
            self.publish_execution_result(False)

            return

        self.get_logger().warn("Trajectory goal ACCEPTED. Robot may now move.")

        result_future = goal_handle.get_result_async()

        # Add a callback to check if goal is reached
        result_future.add_done_callback(self.result_callback)

    # ==============================================================
    # EXECUTION RESULT
    # ==============================================================
    def result_callback(self, future):
        try:
            result_response = future.result()
            result = result_response.result

        except Exception as exception:
            self.get_logger().error(
                f"Error waiting for trajectory result: {exception}"
            )
            self.execution_in_progress = False
            self.publish_execution_result(False)

            return

        self.execution_in_progress = False

        if result.error_code == 0:
            self.get_logger().info("Trajectory execution SUCCESSFUL.")

            # Prevent accidental execution of the same trajectory again.
            self.latest_trajectory = None

            self.publish_execution_result(True)

        else:
            self.get_logger().error(
                f"Trajectory execution FAILED. "
                f"error_code={result.error_code}, "
                f"error_string='{result.error_string}'"
            )

            # Require a fresh trajectory after failure.
            self.latest_trajectory = None

            self.publish_execution_result(False)

    def discard_callback(self, request, response):
        if self.execution_in_progress:
            response.success = False
            response.message = "Cannot discard while execution is in progress."
            return response

        if self.latest_trajectory is None:
            response.success = False
            response.message = "No stored trajectory to discard."
            return response

        self.latest_trajectory = None

        response.success = True
        response.message = "Stored ARC trajectory discarded."

        return response

    def publish_execution_result(self, success):

        message = Bool()
        message.data = success

        self.execution_result_publisher.publish(message)

def main(args=None):
    rclpy.init(args=args)

    node = ArcTrajectoryExecutor()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()