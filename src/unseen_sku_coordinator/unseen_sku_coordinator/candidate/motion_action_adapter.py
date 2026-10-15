"""MotionCoordinator-compatible client adapter for a separate ROS 2 action node.

CandidatePlanningCoordinator can keep its existing Future-based API and
rank-fallback logic. This adapter replaces its in-process MotionCoordinator.
"""

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Optional

from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future

from unseen_sku_interfaces.action import PlanSuctionCandidate
from unseen_sku_coordinator.motion.action_bridge import from_motion_action_result
from unseen_sku_coordinator.motion.motion_types import MotionResult, MotionStatus


MOTION_ACTION = "/unseen_sku/plan_suction_candidate"


@dataclass
class _Pending:
    scene_id: str
    object_id: int
    rank: int
    output: Future
    goal_handle: Optional[Any] = None
    cancelled: bool = False
    started_at: float = 0.0


class MotionActionAdapter:
    """One in-flight motion goal at a time, including cancellation latency.

    Cancelling a scene resolves its local Future immediately, but retains the
    in-flight reservation until the remote action terminates. Subsequent
    attempts fail BUSY rather than overlapping a still-running C++ service.
    """

    def __init__(self, node, *, action_name=MOTION_ACTION,
                 on_rank_started: Optional[Callable[[str, int], None]] = None,
                 goal_timeout_sec: float = 95.0):
        self._client = ActionClient(
            node, PlanSuctionCandidate, action_name,
            callback_group=ReentrantCallbackGroup())
        self._lock = threading.RLock()
        self._pending: Optional[_Pending] = None
        self._closed = False
        if goal_timeout_sec <= 0:
            raise ValueError("goal_timeout_sec must be positive")
        self._goal_timeout_sec = goal_timeout_sec
        self._watchdog_timer = node.create_timer(0.25, self._watchdog)
        self.on_rank_started = on_rank_started

    @staticmethod
    def _local_result(scene, object_id, rank, status, detail):
        return MotionResult(status, detail, scene, object_id, rank)

    def plan_candidate_async(self, candidate, header, scene_id):
        output = Future()
        object_id, rank = int(candidate.object_id), int(candidate.rank)
        with self._lock:
            if self._closed:
                output.set_result(self._local_result(
                    scene_id, object_id, rank, MotionStatus.INVALID_REQUEST,
                    "motion action adapter is closed"))
                return output
            if self._pending is not None:
                output.set_result(self._local_result(
                    scene_id, object_id, rank, MotionStatus.BUSY,
                    "previous motion action is still running"))
                return output
            if not self._client.server_is_ready():
                output.set_result(self._local_result(
                    scene_id, object_id, rank, MotionStatus.SERVICE_UNAVAILABLE,
                    "Motion Coordinator action server is not ready"))
                return output

            item = _Pending(scene_id, object_id, rank, output,
                            started_at=time.monotonic())
            self._pending = item
            request = PlanSuctionCandidate.Goal()
            request.scene_id = scene_id
            request.candidate_header = header
            request.candidate = candidate
            try:
                future = self._client.send_goal_async(request)
                future.add_done_callback(
                    lambda completed: self._goal_response(item, completed))
            except Exception as exc:
                self._finish_locked(item, self._local_result(
                    scene_id, object_id, rank, MotionStatus.PLANNER_ERROR,
                    f"failed to submit motion action: {exc}"))
                return output
            if self.on_rank_started is not None:
                try:
                    self.on_rank_started(scene_id, rank)
                except Exception:
                    pass  # Feedback is best-effort, not part of planning policy.
            return output

    def _finish_locked(self, item, result):
        if self._pending is not item:
            return
        self._pending = None
        if not item.output.done():
            item.output.set_result(result)

    def _goal_response(self, item, pending):
        with self._lock:
            if self._pending is not item:
                return
            try:
                goal_handle = pending.result()
                if goal_handle is None or not goal_handle.accepted:
                    self._finish_locked(item, self._local_result(
                        item.scene_id, item.object_id, item.rank,
                        MotionStatus.BUSY, "motion action goal was rejected"))
                    return
                item.goal_handle = goal_handle
                reply = goal_handle.get_result_async()
                reply.add_done_callback(
                    lambda completed: self._result_response(item, completed))
                if item.cancelled:
                    goal_handle.cancel_goal_async()
            except Exception as exc:
                self._finish_locked(item, self._local_result(
                    item.scene_id, item.object_id, item.rank,
                    MotionStatus.PLANNER_ERROR,
                    f"motion action goal handling error: {exc}"))

    def _result_response(self, item, future):
        with self._lock:
            if self._pending is not item:
                return
            try:
                completed = future.result()
                decoded = from_motion_action_result(
                    completed.result, completed.status,
                    item.scene_id, item.object_id, item.rank)
            except Exception as exc:
                decoded = self._local_result(
                    item.scene_id, item.object_id, item.rank,
                    MotionStatus.PLANNER_ERROR,
                    f"motion action result handling error: {exc}")
            self._finish_locked(item, decoded)

    def _watchdog(self):
        with self._lock:
            item = self._pending
            if (item is None or item.output.done()
                    or time.monotonic() - item.started_at < self._goal_timeout_sec):
                return
            # Do not release the reservation yet: a remote C++ plan may still
            # be running. Only the late action result clears _pending.
            item.cancelled = True
            item.output.set_result(self._local_result(
                item.scene_id, item.object_id, item.rank,
                MotionStatus.SERVICE_TIMEOUT,
                "remote motion action exceeded client timeout; state uncertain"))
            if item.goal_handle is not None:
                item.goal_handle.cancel_goal_async()

    def cancel_scene(self, scene_id):
        with self._lock:
            item = self._pending
            if item is None or item.scene_id != scene_id:
                return
            item.cancelled = True
            # Inform CandidatePlanningCoordinator immediately so it can cancel
            # the scene; reserve the action slot until the server finishes.
            if not item.output.done():
                item.output.set_result(self._local_result(
                    item.scene_id, item.object_id, item.rank,
                    MotionStatus.CANCELLED, "scene cancelled; late result ignored"))
            if item.goal_handle is not None:
                item.goal_handle.cancel_goal_async()

    def close(self):
        with self._lock:
            self._closed = True
            if self._pending is not None:
                self.cancel_scene(self._pending.scene_id)
        self._client.destroy()
        self._watchdog_timer.cancel()
