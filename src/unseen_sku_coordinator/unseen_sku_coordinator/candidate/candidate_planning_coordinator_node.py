"""Planning-only ROS action wrapper for ranked suction candidates.

The BT supplies a SuctionResult already validated by Perception Coordinator.
This node delegates every rank to the separate Motion Coordinator action node.
No trajectory execution, vacuum operation, or object reselection occurs here.
"""

import threading

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from unseen_sku_interfaces.action import PlanSuctionObject

from .action_bridge import to_candidate_action_result
from .candidate_planning_coordinator import (
    CandidatePlanningCoordinator, CandidatePlanningResult,
    CandidatePlanningStatus,
)
from .motion_action_adapter import MotionActionAdapter


ACTION_NAME = "/unseen_sku/plan_object"


class CandidatePlanningCoordinatorNode(Node):
    """Expose ranked planning as an independent BT-friendly ROS action."""

    def __init__(self):
        super().__init__("candidate_planning_coordinator_node")
        self.declare_parameter("max_candidate_attempts", 5)
        self.declare_parameter("motion_action_timeout_sec", 95.0)
        self.declare_parameter(
            "motion_action", "/unseen_sku/plan_suction_candidate")
        self._lock = threading.Lock()
        self._reserved = False
        self._active_goal = None
        self._cancel_forwarded = False
        self._attempt_count = 0
        self._cancel_timer = self.create_timer(0.05, self._poll_cancellation)
        self._motion = MotionActionAdapter(
            self, action_name=self.get_parameter("motion_action").value,
            on_rank_started=self._rank_started,
            goal_timeout_sec=float(
                self.get_parameter("motion_action_timeout_sec").value))
        self._candidate = CandidatePlanningCoordinator(
            self._motion,
            max_candidate_attempts=int(
                self.get_parameter("max_candidate_attempts").value))
        self._action = ActionServer(
            self, PlanSuctionObject, ACTION_NAME,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f"Planning-only ranked candidate action ready: {ACTION_NAME}")

    def _on_goal(self, request):
        with self._lock:
            if self._reserved:
                self.get_logger().warning("Candidate action busy; rejecting goal")
                return GoalResponse.REJECT
            self._reserved = True
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        # Allow rclpy to transition into CANCELING before completing the
        # coordinator Future; see _poll_cancellation below.
        return CancelResponse.ACCEPT

    def _poll_cancellation(self):
        with self._lock:
            goal = self._active_goal
            if (goal is None or not goal.is_cancel_requested
                    or self._cancel_forwarded):
                return
            self._cancel_forwarded = True
        self._candidate.cancel_scene(goal.request.expected_scene_id)

    def _rank_started(self, scene_id, rank):
        with self._lock:
            goal = self._active_goal
            if goal is None or not goal.is_active or (
                    goal.request.expected_scene_id != scene_id):
                return
            self._attempt_count += 1
            completed = self._attempt_count - 1
        feedback = PlanSuctionObject.Feedback()
        feedback.phase = "planning_rank"
        feedback.current_rank = rank
        feedback.attempts_completed = completed
        goal.publish_feedback(feedback)

    async def _execute(self, goal_handle):
        request = goal_handle.request
        with self._lock:
            self._active_goal = goal_handle
            self._cancel_forwarded = False
            self._attempt_count = 0
        try:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return PlanSuctionObject.Result()

            pending = self._candidate.plan_object_async(
                request.suction_result,
                request.target_object_id,
                expected_scene_id=request.expected_scene_id,
            )
            outcome = await pending
            # Pass the original header, which is preserved for all ranked
            # candidates. Never relabel camera coordinates as base_link here.
            wire = to_candidate_action_result(
                outcome, request.suction_result.candidates.header)
            if (goal_handle.is_cancel_requested
                    or outcome.status == CandidatePlanningStatus.CANCELLED):
                goal_handle.canceled()
            elif outcome.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return wire
        except Exception as exc:
            self.get_logger().error(f"Candidate action wrapper error: {exc}")
            if goal_handle.is_active:
                goal_handle.abort()
            return to_candidate_action_result(CandidatePlanningResult(
                CandidatePlanningStatus.SYSTEM_ERROR,
                request.expected_scene_id, 0,
                int(request.target_object_id),
                f"candidate action wrapper: {exc}"))
        finally:
            with self._lock:
                self._active_goal = None
                self._reserved = False

    def destroy_node(self):
        self.destroy_timer(self._cancel_timer)
        self._candidate.close()
        self._motion.close()
        self._action.destroy()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CandidatePlanningCoordinatorNode()
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
