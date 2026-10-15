"""Planning-only ROS 2 action wrapper around the existing MotionCoordinator.

This is a ROS node, but the MotionCoordinator core remains a plain Python class.
It never sends FollowJointTrajectory/ExecuteTrajectory or vacuum commands.
"""

import threading

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from unseen_sku_interfaces.action import PlanSuctionCandidate

from .action_bridge import to_motion_action_result
from .motion_coordinator import MotionCoordinator
from .motion_types import MotionResult, MotionStatus


ACTION_NAME = "/unseen_sku/plan_suction_candidate"


class MotionCoordinatorNode(Node):
    """Serve one pre-transformed base_link planning goal at a time."""

    def __init__(self):
        super().__init__("motion_coordinator_node")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("planner_service", "/unseen_sku/plan_candidate")
        self.declare_parameter("max_candidate_age_sec", 30.0)
        self.declare_parameter("service_wait_timeout_sec", 5.0)
        self.declare_parameter("planning_timeout_sec", 60.0)
        self._planner = MotionCoordinator(
            self,
            base_frame=self.get_parameter("base_frame").value,
            service_name=self.get_parameter("planner_service").value,
            max_candidate_age_sec=float(
                self.get_parameter("max_candidate_age_sec").value),
            service_wait_timeout_sec=float(
                self.get_parameter("service_wait_timeout_sec").value),
            planning_timeout_sec=float(
                self.get_parameter("planning_timeout_sec").value),
        )
        self._goal_lock = threading.Lock()
        self._reserved = False
        self._active_goal = None
        self._cancel_forwarded = False
        self._cancel_timer = self.create_timer(0.05, self._poll_cancellation)
        self._action = ActionServer(
            self, PlanSuctionCandidate, ACTION_NAME,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f"Planning-only motion action ready: {ACTION_NAME}")

    def _on_goal(self, request):
        with self._goal_lock:
            if self._reserved:
                self.get_logger().warning("Motion action busy; rejecting new goal")
                return GoalResponse.REJECT
            self._reserved = True
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        # Return first so rclpy marks the goal CANCELING; the timer then
        # invalidates the core Future. This avoids racing goal_handle.canceled().
        return CancelResponse.ACCEPT

    def _poll_cancellation(self):
        with self._goal_lock:
            goal = self._active_goal
            if (goal is None or not goal.is_cancel_requested
                    or self._cancel_forwarded):
                return
            self._cancel_forwarded = True
        # The C++ PlanCandidate service cannot be preempted. Only the local
        # result is cancelled; the late C++ response will be ignored.
        self._planner.cancel_scene(goal.request.scene_id)

    async def _execute(self, goal_handle):
        goal = goal_handle.request
        with self._goal_lock:
            self._active_goal = goal_handle
            self._cancel_forwarded = False
        try:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return PlanSuctionCandidate.Result()

            feedback = PlanSuctionCandidate.Feedback()
            feedback.phase = "planning"
            goal_handle.publish_feedback(feedback)
            pending = self._planner.plan_candidate_async(
                goal.candidate, goal.candidate_header, goal.scene_id)
            # rclpy Future is awaitable: the executor continues spinning TF,
            # the planning service, timers and cancellation callbacks.
            outcome = await pending
            wire = to_motion_action_result(outcome)
            if goal_handle.is_cancel_requested or outcome.status == MotionStatus.CANCELLED:
                goal_handle.canceled()
            elif outcome.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return wire
        except Exception as exc:
            self.get_logger().error(f"Motion action wrapper error: {exc}")
            if goal_handle.is_active:
                goal_handle.abort()
            return to_motion_action_result(MotionResult(
                MotionStatus.PLANNER_ERROR, f"motion action wrapper: {exc}",
                goal.scene_id, int(goal.candidate.object_id), int(goal.candidate.rank)))
        finally:
            with self._goal_lock:
                self._active_goal = None
                self._reserved = False

    def destroy_node(self):
        self.destroy_timer(self._cancel_timer)
        self._planner.close()
        self._action.destroy()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotionCoordinatorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
