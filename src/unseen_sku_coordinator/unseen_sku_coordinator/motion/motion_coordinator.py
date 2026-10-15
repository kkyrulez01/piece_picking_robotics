"""Plan ONE suction candidate via the C++ PlanCandidate service.

This is a helper owned/spun by the future main coordinator node, not a new
ROS node. It NEVER publishes trajectories or sends execution goals.

The caller must provide an upstream-transformed base_link SuctionCandidate,
its parent array's Header (base_link + capture stamp), and matching scene_id.
Ranked fallback is intentionally NOT implemented here.

Usage (from an already-spinning rclpy Node)::

    motion = MotionCoordinator(node)
    future = motion.plan_candidate_async(candidate, candidate_array.header, scene_id)
    # In a BT tick: if future.done(): result = future.result()

Keep spinning the parent node while waiting. Do NOT spin_until_future_complete
from inside another ROS callback. Call cancel_scene(scene_id) when the BT
abandons or replaces a scene; late service replies are ignored.

Only MotionCoordinator is defined in this module. Planning result types and
validation helpers remain re-exported for existing callers.
"""

import threading
import time
from typing import Any, Optional

from rclpy.task import Future

from unseen_sku_motion_planning.srv import PlanCandidate

from .candidate_geometry import _normalized, _rotate, _xyz, prepare_base_candidate
from .motion_types import _Attempt, MotionResult, MotionStatus
from .planner_failure import _classify_planner_rejection
from .trajectory_validation import JOINT_NAMES, _seconds, validate_plan_response


class MotionCoordinator:
    """Asynchronous, planning-only ROS adapter for *one candidate at a time*.

    Receives only pre-transformed base-frame candidates. Never calls TF2.
    Call from a normally spun rclpy Node; the node's executor must keep running.
    The coordinator uses short nonblocking timer ticks rather than waiting for
    TF or ROS services inside a callback. Only one pending PlanCandidate request
    is allowed; a timed-out/cancelled service request must finish server-side
    before another request can start (it cannot be cancelled by this interface).
    """

    def __init__(self, node: Any, *, base_frame: str = "base_link",
                 service_name: str = "/unseen_sku/plan_candidate",
                 max_candidate_age_sec: Optional[float] = 30.0,
                 service_wait_timeout_sec: float = 5.0,
                 planning_timeout_sec: float = 60.0):
        if base_frame != "base_link":
            raise ValueError("C++ PlanCandidate requires base_link input")
        self.node = node
        self.base_frame = base_frame
        self.max_candidate_age_sec = max_candidate_age_sec
        self.service_wait_timeout_sec = service_wait_timeout_sec
        self.planning_timeout_sec = planning_timeout_sec
        self._client = node.create_client(PlanCandidate, service_name)
        self._lock = threading.RLock()
        self._active = None
        self._inflight = None
        self._timer = node.create_timer(0.05, self._tick)

    def _capture_age_error(self, header: Any) -> Optional[str]:
        """Return an error if the candidate exceeds its capture-age policy."""
        if self.max_candidate_age_sec is None:
            return None
        age = self.node.get_clock().now().nanoseconds * 1e-9 - _seconds(header.stamp)
        if age > self.max_candidate_age_sec or age < -2.0:
            return f"capture age outside allowed range ({age:.2f}s)"
        return None

    def plan_candidate_async(self, candidate: Any, header: Any,
                             scene_id: str) -> Future:
        """Return Future[MotionResult] immediately; never execute trajectories."""
        result_future = Future()
        try:
            object_id = int(candidate.object_id)
            rank = int(candidate.rank)
        except (TypeError, ValueError, AttributeError) as exc:
            result_future.set_result(MotionResult(
                status=MotionStatus.INVALID_REQUEST,
                detail=f"invalid candidate identifiers: {exc}",
                scene_id=scene_id, object_id=0, rank=0,
            ))
            return result_future

        def complete(status: MotionStatus, detail: str) -> Future:
            result_future.set_result(MotionResult(
                status=status, detail=detail, scene_id=scene_id,
                object_id=object_id, rank=rank,
            ))
            return result_future

        with self._lock:
            if self._inflight is not None and self._inflight.done():
                self._inflight = None
            if self._active is not None or self._inflight is not None:
                return complete(MotionStatus.BUSY, "previous planning request still active")
            if not isinstance(scene_id, str) or not scene_id.strip():
                return complete(MotionStatus.INVALID_REQUEST, "scene_id is required")
            if object_id <= 0 or rank <= 0:
                return complete(MotionStatus.INVALID_REQUEST, "invalid object ID or rank")
            try:
                if not header.frame_id or not header.frame_id.strip():
                    raise ValueError("candidate frame_id is empty")
                if header.frame_id != self.base_frame:
                    raise ValueError(
                        f"expected pre-transformed {self.base_frame} candidates; "
                        f"received {header.frame_id!r}"
                    )
                if _seconds(header.stamp) <= 0:
                    raise ValueError("missing capture timestamp")
            except (ValueError, TypeError, AttributeError) as exc:
                return complete(MotionStatus.INVALID_REQUEST, str(exc))
            try:
                point, normal = prepare_base_candidate(
                    candidate, header, self.base_frame
                )
            except (ValueError, TypeError, AttributeError) as exc:
                return complete(MotionStatus.INVALID_CANDIDATE, str(exc))
            age_error = self._capture_age_error(header)
            if age_error:
                return complete(MotionStatus.STALE_CANDIDATE, age_error)

            self._active = _Attempt(
                candidate=candidate, header=header, scene_id=scene_id,
                output=result_future, phase="wait_service",
                deadline=time.monotonic() + self.service_wait_timeout_sec,
                point_base=point, normal_base=normal,
            )
            # Try immediately. Pending work advances on subsequent timer ticks.
            self._tick()
            return result_future

    def cancel_scene(self, scene_id: str) -> None:
        """Invalidate a scene's active result; cannot cancel C++ service work."""
        with self._lock:
            attempt = self._active
            if attempt is not None and attempt.scene_id == scene_id:
                self._finish_locked(attempt, MotionStatus.CANCELLED,
                                    "scene cancelled/replaced; late plan ignored")

    def _finish_locked(self, attempt: _Attempt, status: MotionStatus,
                       detail: str, response: Optional[Any] = None,
                       *, failure_code: int = 0, failed_stage: str = "",
                       moveit_error_code: int = 0) -> None:
        if self._active is not attempt:
            return
        self._active = None
        if self._inflight is not None and self._inflight.done():
            self._inflight = None
        if not attempt.output.done():
            attempt.output.set_result(MotionResult(
                status=status, detail=detail, scene_id=attempt.scene_id,
                object_id=int(attempt.candidate.object_id),
                rank=int(attempt.candidate.rank),
                suction_point_base=attempt.point_base,
                surface_normal_base=attempt.normal_base,
                planning_response=response if status == MotionStatus.PLANNED else None,
                failure_code=failure_code,
                failed_stage=failed_stage,
                moveit_error_code=moveit_error_code,
            ))

    def _tick(self) -> None:
        with self._lock:
            try:
                self._tick_locked()
            except Exception as exc:  # Fail closed instead of leaving a stuck BT.
                if self._active is not None:
                    self._finish_locked(self._active, MotionStatus.PLANNER_ERROR,
                                        f"motion adapter exception: {exc}")

    def _tick_locked(self) -> None:
        if self._inflight is not None and self._inflight.done() and self._active is None:
            self._inflight = None
        attempt = self._active
        if attempt is None:
            return
        now = time.monotonic()
        if attempt.phase == "wait_service":
            age_error = self._capture_age_error(attempt.header)
            if age_error:
                self._finish_locked(attempt, MotionStatus.STALE_CANDIDATE, age_error)
                return
            if not self._client.service_is_ready():
                if now >= attempt.deadline:
                    self._finish_locked(attempt, MotionStatus.SERVICE_UNAVAILABLE,
                                        "PlanCandidate service is not ready")
                return
            request = PlanCandidate.Request()
            request.suction_point = attempt.point_base
            request.surface_normal = attempt.normal_base
            attempt.request_future = self._client.call_async(request)
            self._inflight = attempt.request_future
            attempt.phase = "wait_result"
            attempt.deadline = now + self.planning_timeout_sec

        if attempt.phase == "wait_result":
            if attempt.request_future.done():
                try:
                    response = attempt.request_future.result()
                    if response is None:
                        raise ValueError("planning service returned no response")
                    if not response.success:
                        status = _classify_planner_rejection(response)
                        self._finish_locked(
                            attempt, status, response.message,
                            failure_code=response.failure_code,
                            failed_stage=response.failed_stage,
                            moveit_error_code=response.moveit_error_code,
                        )
                        return
                    if response.failure_code != PlanCandidate.Response.FAILURE_NONE:
                        raise ValueError("successful plan carried a nonzero failure_code")
                    validate_plan_response(
                        response, attempt.point_base, attempt.normal_base,
                        self.base_frame,
                    )
                    age_error = self._capture_age_error(attempt.header)
                    if age_error:
                        self._finish_locked(attempt, MotionStatus.STALE_CANDIDATE, age_error)
                        return
                    self._finish_locked(attempt, MotionStatus.PLANNED,
                                        response.message, response)
                except ValueError as exc:
                    self._finish_locked(attempt, MotionStatus.INVALID_RESPONSE, str(exc))
                except Exception as exc:
                    self._finish_locked(attempt, MotionStatus.PLANNER_ERROR,
                                        f"planning service exception: {exc}")
                return
            if now >= attempt.deadline:
                # Keep _inflight populated; C++ cannot cancel this service call.
                self._finish_locked(attempt, MotionStatus.SERVICE_TIMEOUT,
                                    "PlanCandidate reply timed out; backend may still be planning")

    def close(self) -> None:
        """Stop local polling. Call when destroying the owning main node."""
        with self._lock:
            if self._active is not None:
                self._finish_locked(self._active, MotionStatus.CANCELLED,
                                    "motion coordinator closed")
            self.node.destroy_timer(self._timer)
