"""Plan ONE suction candidate through the existing C++ PlanCandidate service.

This is a helper owned/spun by the future main coordinator node, not a new
ROS node. It NEVER publishes trajectories or sends execution goals.

The caller must provide a validated SuctionCandidate, its parent array's
Header (frame + capture stamp), and the matching perception scene_id. Ranked
fallback is intentionally NOT implemented here.

Usage (from an already-spinning rclpy Node)::

    motion = MotionCoordinator(node)
    future = motion.plan_candidate_async(candidate, candidate_array.header, scene_id)
    # In a BT tick: if future.done(): result = future.result()

Keep spinning the parent node while waiting. Do NOT spin_until_future_complete
from inside another ROS callback. Call cancel_scene(scene_id) when the BT
abandons or replaces a scene; late service replies are ignored.
"""

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Any, Optional

from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.task import Future
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from unseen_sku_motion_planning.srv import PlanCandidate


JOINT_NAMES = tuple(f"Revolute_{index}" for index in range(1, 7))


class MotionStatus(str, Enum):
    """Classifies a *single* candidate-planning attempt."""

    PLANNED = "planned"
    INVALID_CANDIDATE = "invalid_candidate"
    INVALID_REQUEST = "invalid_request"  # Invalid scene metadata/request, not rank-retryable.
    STALE_CANDIDATE = "stale_candidate"
    TF_UNAVAILABLE = "tf_unavailable"
    SERVICE_UNAVAILABLE = "service_unavailable"
    SERVICE_TIMEOUT = "service_timeout"
    ROBOT_STATE_UNAVAILABLE = "robot_state_unavailable"
    CANDIDATE_REJECTED = "candidate_rejected"
    PLANNING_FAILED = "planning_failed"  # MoveIt cause unknown: do NOT auto-retry.
    PLANNER_ERROR = "planner_error"
    INVALID_RESPONSE = "invalid_response"
    BUSY = "busy"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class MotionResult:
    """Return value for a single request; no robot movement is implied."""

    status: MotionStatus
    detail: str
    scene_id: str
    object_id: int
    rank: int
    suction_point_base: Optional[Any] = None
    surface_normal_base: Optional[Any] = None
    planning_response: Optional[Any] = None
    failure_code: int = 0
    failed_stage: str = ""
    moveit_error_code: int = 0

    @property
    def success(self) -> bool:
        return self.status == MotionStatus.PLANNED

    @property
    def retry_next_candidate(self) -> bool:
        """Only a clearly candidate-specific rejection permits rank fallback.

        An unclassified MoveIt PLANNING_FAILED result is non-retryable.
        A specific NO_IK_SOLUTION / GOAL_IN_COLLISION result may be retryable,
        but only when explicitly categorized by the C++ planner.
        """
        return self.status in (
            MotionStatus.INVALID_CANDIDATE,
            MotionStatus.CANDIDATE_REJECTED,
        )


@dataclass
class _Attempt:
    candidate: Any
    header: Any
    scene_id: str
    output: Future
    phase: str
    deadline: float
    point_base: Optional[Any] = None
    normal_base: Optional[Any] = None
    request_future: Optional[Any] = None
    last_tf_error: str = ""


def _xyz(value: Any) -> tuple:
    return (float(value.x), float(value.y), float(value.z))


def _normalized(vector: tuple) -> tuple:
    if not all(math.isfinite(v) for v in vector):
        raise ValueError("nonfinite surface normal")
    length = math.sqrt(sum(v * v for v in vector))
    if length < 1e-6:
        raise ValueError("surface normal has near-zero magnitude")
    return tuple(v / length for v in vector)


def _rotate(vector: tuple, quaternion: Any) -> tuple:
    """Rotate a 3-vector by a normalized TF quaternion (xyzw)."""
    x, y, z, w = (
        float(quaternion.x), float(quaternion.y),
        float(quaternion.z), float(quaternion.w),
    )
    if not all(math.isfinite(v) for v in (x, y, z, w)):
        raise ValueError("TF quaternion contains nonfinite values")
    length = math.sqrt(x*x + y*y + z*z + w*w)
    if length < 1e-9:
        raise ValueError("TF quaternion has zero magnitude")
    x, y, z, w = (x/length, y/length, z/length, w/length)
    vx, vy, vz = vector
    # v' = v + w*t + cross(q.xyz, t), t = 2*cross(q.xyz, v).
    tx, ty, tz = (
        2.0 * (y*vz - z*vy),
        2.0 * (z*vx - x*vz),
        2.0 * (x*vy - y*vx),
    )
    return (
        vx + w*tx + y*tz - z*ty,
        vy + w*ty + z*tx - x*tz,
        vz + w*tz + x*ty - y*tx,
    )


def transform_candidate(candidate: Any, header: Any, target_frame: str,
                        tf_transform: Optional[Any] = None) -> tuple:
    """Build PointStamped and Vector3Stamped in target_frame.

    tf_transform is a geometry_msgs/Transform (or None for identity). Only the
    POINT is translated. The surface NORMAL is rotated and renormalized.
    Capture timestamp is preserved for the downstream planning request.
    """
    position = _xyz(candidate.position)
    normal = _normalized(_xyz(candidate.normal))
    if not all(math.isfinite(v) for v in position):
        raise ValueError("nonfinite candidate position")
    if tf_transform is None:
        transformed_position, transformed_normal = position, normal
    else:
        rotated_point = _rotate(position, tf_transform.rotation)
        transformed_position = tuple(
            a + b for a, b in zip(rotated_point, _xyz(tf_transform.translation))
        )
        transformed_normal = _normalized(_rotate(normal, tf_transform.rotation))
    if not all(math.isfinite(v) for v in transformed_position):
        raise ValueError("transformed candidate position is nonfinite")

    point = PointStamped()
    point.header.frame_id = target_frame
    point.header.stamp = header.stamp
    point.point.x, point.point.y, point.point.z = transformed_position
    vector = Vector3Stamped()
    vector.header.frame_id = target_frame
    vector.header.stamp = header.stamp
    vector.vector.x, vector.vector.y, vector.vector.z = transformed_normal
    return point, vector


def _seconds(duration: Any) -> float:
    if duration.sec < 0 or duration.nanosec < 0 or duration.nanosec >= 1_000_000_000:
        raise ValueError("invalid trajectory timestamp")
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def validate_plan_response(response: Any, point: Any, normal: Any,
                           target_frame: str = "base_link") -> None:
    """Raise ValueError if a successful C++ response is structurally invalid.

    Checks nonempty 6-joint trajectories, finite arrays, increasing timestamps,
    zero terminal velocities, segment continuity, and matching target poses.
    This does not establish obstacle clearance or physical execution safety.
    """
    trajectories = (
        response.pre_pick_trajectory,
        response.pick_trajectory,
        response.retract_trajectory,
    )
    canonical_names = None
    for label, trajectory in zip(("PRE_PICK", "PICK", "RETRACT"), trajectories):
        names = list(trajectory.joint_names)
        if len(names) != 6 or len(set(names)) != 6 or set(names) != set(JOINT_NAMES):
            raise ValueError(f"{label}: unexpected joint names")
        if canonical_names is None:
            canonical_names = names
        elif names != canonical_names:
            raise ValueError(f"{label}: joint order differs between stages")
        if not trajectory.points:
            raise ValueError(f"{label}: empty trajectory")
        last_t = -1.0
        for waypoint in trajectory.points:
            t = _seconds(waypoint.time_from_start)
            if t <= last_t:
                raise ValueError(f"{label}: timestamps not strictly increasing")
            last_t = t
            if len(waypoint.positions) != 6 or not all(
                math.isfinite(v) for v in waypoint.positions
            ):
                raise ValueError(f"{label}: invalid joint positions")
            for field in ("velocities", "accelerations", "effort"):
                values = getattr(waypoint, field)
                if values and (len(values) != 6 or not all(math.isfinite(v) for v in values)):
                    raise ValueError(f"{label}: invalid {field}")
        last_velocity = trajectory.points[-1].velocities
        if last_velocity and any(abs(v) > 1e-3 for v in last_velocity):
            raise ValueError(f"{label}: nonzero terminal velocity")

    for previous, following in zip(trajectories, trajectories[1:]):
        end = previous.points[-1].positions
        start = following.points[0].positions
        if max(abs(a-b) for a, b in zip(end, start)) > 1e-3:
            raise ValueError("trajectory stages are not joint-continuous")

    poses = (response.pre_pick_pose, response.pick_pose, response.retract_pose)
    for label, pose in zip(("PRE_PICK", "PICK", "RETRACT"), poses):
        if pose.header.frame_id != target_frame:
            raise ValueError(f"{label}: unexpected pose frame")
        if not all(math.isfinite(v) for v in _xyz(pose.pose.position)):
            raise ValueError(f"{label}: nonfinite target position")
        quaternion = pose.pose.orientation
        qvalues = (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        if not all(math.isfinite(v) for v in qvalues) or abs(
            math.sqrt(sum(v*v for v in qvalues)) - 1.0
        ) > 1e-3:
            raise ValueError(f"{label}: invalid target orientation")

    pick_position = _xyz(response.pick_pose.pose.position)
    if max(abs(a-b) for a, b in zip(pick_position, _xyz(point.point))) > 1e-4:
        raise ValueError("pick pose does not match requested contact point")
    outward = _xyz(normal.vector)
    tool_z = _rotate((0.0, 0.0, 1.0), response.pick_pose.pose.orientation)
    if sum(a*b for a, b in zip(tool_z, outward)) > -0.999:
        raise ValueError("pick TCP +Z is not opposite the outward normal")
    for label, pose in (("PRE_PICK", response.pre_pick_pose),
                        ("RETRACT", response.retract_pose)):
        offset = tuple(a-b for a, b in zip(_xyz(pose.pose.position), pick_position))
        along = sum(a*b for a, b in zip(offset, outward))
        lateral = math.sqrt(sum((a - along*b)**2 for a, b in zip(offset, outward)))
        if along < 1e-4 or lateral > 1e-3:
            raise ValueError(f"{label}: incorrect surface-normal offset")


def _classify_planner_rejection(response: Any) -> MotionStatus:
    """Map structured ROS failure_code to MotionStatus; never parse message text.

    Unknown codes, missing fields, or FAILURE_NONE with success=False fail
    closed as PLANNER_ERROR (never cause automatic rank fallback).
    """
    codes = PlanCandidate.Response
    status_by_code = {
        codes.FAILURE_INVALID_REQUEST: MotionStatus.INVALID_REQUEST,
        codes.FAILURE_INVALID_CANDIDATE: MotionStatus.INVALID_CANDIDATE,
        codes.FAILURE_CANDIDATE_INFEASIBLE: MotionStatus.CANDIDATE_REJECTED,
        codes.FAILURE_ROBOT_STATE_UNAVAILABLE: MotionStatus.ROBOT_STATE_UNAVAILABLE,
        codes.FAILURE_PLANNING_FAILED_UNKNOWN: MotionStatus.PLANNING_FAILED,
        codes.FAILURE_PLANNER_ERROR: MotionStatus.PLANNER_ERROR,
        codes.FAILURE_INVALID_TRAJECTORY: MotionStatus.INVALID_RESPONSE,
    }
    return status_by_code.get(getattr(response, "failure_code", None),
                              MotionStatus.PLANNER_ERROR)


class MotionCoordinator:
    """Asynchronous, planning-only ROS adapter for *one candidate at a time*.

    Call from a normally spun rclpy Node; the node's executor must keep running.
    The coordinator uses short nonblocking timer ticks rather than waiting for
    TF or ROS services inside a callback. Only one pending PlanCandidate request
    is allowed; a timed-out/cancelled service request must finish server-side
    before another request can start (it cannot be cancelled by this interface).
    """

    def __init__(self, node: Any, *, tf_buffer: Optional[Any] = None,
                 base_frame: str = "base_link",
                 service_name: str = "/unseen_sku/plan_candidate",
                 max_candidate_age_sec: Optional[float] = 30.0,
                 tf_wait_timeout_sec: float = 3.0,
                 service_wait_timeout_sec: float = 5.0,
                 planning_timeout_sec: float = 60.0):
        self.node = node
        self.base_frame = base_frame
        self.max_candidate_age_sec = max_candidate_age_sec
        self.tf_wait_timeout_sec = tf_wait_timeout_sec
        self.service_wait_timeout_sec = service_wait_timeout_sec
        self.planning_timeout_sec = planning_timeout_sec
        self.tf_buffer = tf_buffer if tf_buffer is not None else Buffer()
        self._tf_listener = (
            TransformListener(self.tf_buffer, node, spin_thread=False)
            if tf_buffer is None else None
        )
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
                if _seconds(header.stamp) <= 0:
                    raise ValueError("missing capture timestamp")
            except (ValueError, TypeError, AttributeError) as exc:
                return complete(MotionStatus.INVALID_REQUEST, str(exc))
            try:
                position = _xyz(candidate.position)
                _normalized(_xyz(candidate.normal))
                if not all(math.isfinite(v) for v in position):
                    raise ValueError("nonfinite suction position")
            except (ValueError, TypeError, AttributeError) as exc:
                return complete(MotionStatus.INVALID_CANDIDATE, str(exc))
            age_error = self._capture_age_error(header)
            if age_error:
                return complete(MotionStatus.STALE_CANDIDATE, age_error)

            self._active = _Attempt(
                candidate=candidate, header=header, scene_id=scene_id,
                output=result_future, phase="wait_tf",
                deadline=time.monotonic() + self.tf_wait_timeout_sec,
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
        if attempt.phase == "wait_tf":
            try:
                if attempt.header.frame_id == self.base_frame:
                    tf = None
                else:
                    tf = self.tf_buffer.lookup_transform(
                        self.base_frame,
                        attempt.header.frame_id,
                        Time.from_msg(attempt.header.stamp),
                    ).transform
                attempt.point_base, attempt.normal_base = transform_candidate(
                    attempt.candidate, attempt.header, self.base_frame, tf
                )
                attempt.phase = "wait_service"
                attempt.deadline = now + self.service_wait_timeout_sec
            except TransformException as exc:
                attempt.last_tf_error = str(exc)
                if now >= attempt.deadline:
                    self._finish_locked(attempt, MotionStatus.TF_UNAVAILABLE,
                                        f"TF at capture stamp unavailable: {exc}")
                return
            except ValueError as exc:
                # Geometry was validated before entering this state. An invalid
                # TF quaternion/translation is a frame-system problem, not an
                # unreachable suction candidate.
                self._finish_locked(attempt, MotionStatus.TF_UNAVAILABLE, str(exc))
                return

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
