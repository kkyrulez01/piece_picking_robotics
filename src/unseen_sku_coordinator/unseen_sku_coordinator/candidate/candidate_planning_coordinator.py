"""Select the first feasible ranked suction candidate for ONE chosen object.

This is a planning-only helper, NOT a ROS node or an execution controller.
The owner must keep spinning the rclpy node used by MotionCoordinator.

Expected input is the *validated* SuctionResult returned when
GetPerceptionSceneResult.has_validated_suction_result is true. Always pass the
scene_id originally returned by StartPerceptionScene as expected_scene_id.
The caller/main BT decides which object to consider and whether to rescan.

Example::

    coordinator = CandidatePlanningCoordinator(motion, max_candidate_attempts=5)
    future = coordinator.plan_object_async(
        suction_result, target_object_id=2, expected_scene_id=scene_id
    )
    # Poll future.done() from a BT tick; do not block a ROS callback.
    # On abandonment, coordinator.cancel_scene(scene_id).

Only MotionCoordinator calls the C++ PlanCandidate service. This module NEVER
publishes trajectories, moves the robot, or controls vacuum.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import math
import threading
import time
from typing import Any, Optional

from rclpy.task import Future

from unseen_sku_coordinator.motion.motion_coordinator import (
    MotionResult,
    MotionStatus,
)


class CandidatePlanningStatus(str, Enum):
    """Outcome of ranked planning for a selected object in one scene."""

    PICK_READY = "pick_ready"
    NO_CANDIDATES = "no_candidates"
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    ATTEMPT_LIMIT_REACHED = "attempt_limit_reached"
    INVALID_REQUEST = "invalid_request"
    SYSTEM_ERROR = "system_error"
    BUSY = "busy"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CandidateAttempt:
    """Diagnostic record for a single rank, never an executed robot move."""

    object_id: int
    rank: int
    source_index: int
    ranking_score: float
    motion_status: MotionStatus
    detail: str
    duration_ms: float
    failure_code: int = 0
    failed_stage: str = ""
    moveit_error_code: int = 0


@dataclass(frozen=True)
class CandidatePlanningResult:
    """One object/scene result with a selected candidate and its validated plan."""

    status: CandidatePlanningStatus
    scene_id: str
    scene_index: int
    target_object_id: int
    detail: str
    attempts: tuple[CandidateAttempt, ...] = ()
    selected_candidate: Optional[Any] = None
    motion_result: Optional[MotionResult] = None

    @property
    def success(self) -> bool:
        return self.status == CandidatePlanningStatus.PICK_READY

    @property
    def selected_rank(self) -> Optional[int]:
        return (int(self.selected_candidate.rank)
                if self.selected_candidate is not None else None)

    @property
    def planning_response(self) -> Optional[Any]:
        """Contains pre-pick/pick/retract trajectories only on PICK_READY."""
        return (self.motion_result.planning_response
                if self.success and self.motion_result is not None else None)


@dataclass
class _Session:
    scene_id: str
    scene_index: int
    object_id: int
    header: Any
    candidates: tuple[Any, ...]
    output: Future
    attempts: list[CandidateAttempt] = field(default_factory=list)
    next_index: int = 0
    pending_future: Optional[Any] = None
    pending_candidate: Optional[Any] = None
    pending_started_at: float = 0.0


def _read_candidates(suction_result: Any, expected_scene_id: str,
                     object_id: int) -> tuple[int, Any, tuple[Any, ...]]:
    """Validate correlation/shape; return (scene_index, header, ranked items).

    Caller must already have checked the perception coordinator's
    has_validated_suction_result flag. Rechecking a few invariants here is
    defense in depth, not a replacement for perception validation.
    """
    if not isinstance(expected_scene_id, str) or not expected_scene_id.strip():
        raise ValueError("expected_scene_id must be nonempty")
    if getattr(suction_result, "scene_id", None) != expected_scene_id:
        raise ValueError("suction result scene_id does not match expected_scene_id")

    array = suction_result.candidates
    scene_index = int(array.scene_index)
    if scene_index < 0:
        raise ValueError("invalid scene_index")
    status = str(array.status)
    if status == "ERROR":
        raise ValueError(f"perception suction scoring reported ERROR: {array.error_message}")
    if status not in ("OK", "NO_CANDIDATES"):
        raise ValueError(f"unexpected suction scoring status {status!r}")
    if status == "NO_CANDIDATES" and array.candidates:
        raise ValueError("NO_CANDIDATES response unexpectedly contains candidates")
    if status == "OK" and not array.candidates:
        raise ValueError("OK response contains no candidates")

    header = array.header
    if not isinstance(header.frame_id, str) or not header.frame_id.strip():
        raise ValueError("candidate array frame_id is empty")
    if header.frame_id != "base_link":
        raise ValueError(
            f"candidate array must be upstream-transformed to base_link; "
            f"received {header.frame_id!r}"
        )
    stamp = header.stamp
    if (int(stamp.sec) < 0 or int(stamp.nanosec) < 0
            or int(stamp.nanosec) >= 1_000_000_000
            or (int(stamp.sec) == 0 and int(stamp.nanosec) == 0)):
        raise ValueError("candidate array capture timestamp is invalid")

    selected = []
    seen_ranks = set()
    for candidate in array.candidates:
        if int(candidate.object_id) != object_id:
            continue
        rank = int(candidate.rank)
        if rank < 1 or rank in seen_ranks:
            raise ValueError(f"invalid or duplicate rank {rank} for object {object_id}")
        if not math.isfinite(float(candidate.ranking_score)):
            raise ValueError(f"nonfinite ranking_score for object {object_id}, rank {rank}")
        seen_ranks.add(rank)
        selected.append(deepcopy(candidate))

    selected.sort(key=lambda item: int(item.rank))
    return scene_index, deepcopy(header), tuple(selected)


def _complete_plan_present(motion_result: MotionResult) -> bool:
    """Fail closed if a broken adapter claims success without all 3 segments.

    Geometric and temporal trajectory validation remains MotionCoordinator's
    responsibility; do not duplicate MoveIt/trajectory checking here.
    """
    response = motion_result.planning_response
    if response is None:
        return False
    return all(
        getattr(response, pose_name, None) is not None
        and bool(getattr(getattr(response, traj_name, None), "points", ()))
        for pose_name, traj_name in (
            ("pre_pick_pose", "pre_pick_trajectory"),
            ("pick_pose", "pick_trajectory"),
            ("retract_pose", "retract_trajectory"),
        )
    )


class CandidatePlanningCoordinator:
    """Evaluate only one selected object's Sim-Suction ranks in ascending order.

    Nonblocking: at most one outstanding MotionCoordinator request is pending.
    Rank fallback occurs *only* for MotionResult.retry_next_candidate and
    known candidate-local statuses. A generic MoveIt failure always stops.
    """

    def __init__(self, motion_coordinator: Any, *, max_candidate_attempts: int = 5):
        if (isinstance(max_candidate_attempts, bool)
                or not isinstance(max_candidate_attempts, int)
                or max_candidate_attempts < 1):
            raise ValueError("max_candidate_attempts must be a positive integer")
        self._motion = motion_coordinator
        self.max_candidate_attempts = max_candidate_attempts
        self._lock = threading.RLock()
        self._active: Optional[_Session] = None
        self._closed = False

    def plan_object_async(self, suction_result: Any, target_object_id: int,
                          *, expected_scene_id: str) -> Future:
        """Return Future[CandidatePlanningResult], never command robot motion.

        The BT must select target_object_id and verify that the perception
        service returned has_validated_suction_result=True before calling.
        """
        output = Future()
        scene_id = expected_scene_id if isinstance(expected_scene_id, str) else ""
        object_id = 0
        scene_index = 0

        with self._lock:
            if self._closed:
                output.set_result(CandidatePlanningResult(
                    CandidatePlanningStatus.INVALID_REQUEST, scene_id, 0, 0,
                    "candidate planning coordinator is closed"))
                return output
            if self._active is not None:
                output.set_result(CandidatePlanningResult(
                    CandidatePlanningStatus.BUSY, scene_id, 0, 0,
                    "another ranked-candidate operation is still active"))
                return output
            try:
                if isinstance(target_object_id, bool):
                    raise ValueError("target_object_id must be a positive integer")
                object_id = int(target_object_id)
                if object_id < 1 or object_id != target_object_id:
                    raise ValueError("target_object_id must be a positive integer")
                scene_index, header, candidates = _read_candidates(
                    suction_result, scene_id, object_id)
            except (TypeError, ValueError, AttributeError, OverflowError) as exc:
                output.set_result(CandidatePlanningResult(
                    CandidatePlanningStatus.INVALID_REQUEST, scene_id, scene_index,
                    object_id, str(exc)))
                return output
            if not candidates:
                output.set_result(CandidatePlanningResult(
                    CandidatePlanningStatus.NO_CANDIDATES, scene_id, scene_index,
                    object_id, "no suction candidates for the selected object"))
                return output

            session = _Session(
                scene_id=scene_id, scene_index=scene_index,
                object_id=object_id, header=header, candidates=candidates,
                output=output,
            )
            self._active = session
            self._advance_locked(session)
        return output

    def _finish_locked(self, session: _Session, status: CandidatePlanningStatus,
                       detail: str, *, selected_candidate: Optional[Any] = None,
                       motion_result: Optional[MotionResult] = None) -> None:
        if self._active is not session:
            return
        self._active = None  # invalidate callbacks before making result visible
        if not session.output.done():
            session.output.set_result(CandidatePlanningResult(
                status=status, scene_id=session.scene_id,
                scene_index=session.scene_index, target_object_id=session.object_id,
                detail=detail, attempts=tuple(session.attempts),
                selected_candidate=selected_candidate,
                motion_result=motion_result,
            ))

    def _advance_locked(self, session: _Session) -> None:
        """Iteratively consume immediate replies; never recurse through ranks."""
        while self._active is session:
            if session.next_index >= len(session.candidates):
                self._finish_locked(
                    session, CandidatePlanningStatus.NO_FEASIBLE_CANDIDATE,
                    "all available candidates for the selected object were rejected")
                return
            if session.next_index >= self.max_candidate_attempts:
                self._finish_locked(
                    session, CandidatePlanningStatus.ATTEMPT_LIMIT_REACHED,
                    f"stopped after {self.max_candidate_attempts} candidate attempts")
                return
            candidate = session.candidates[session.next_index]
            session.next_index += 1
            session.pending_candidate = candidate
            session.pending_started_at = time.monotonic()
            try:
                pending = self._motion.plan_candidate_async(
                    candidate, session.header, session.scene_id)
                if pending is None or not hasattr(pending, "add_done_callback"):
                    raise TypeError("MotionCoordinator did not return a Future")
                session.pending_future = pending
                if pending.done():
                    if not self._consume_locked(session, pending):
                        return
                    continue
                pending.add_done_callback(
                    lambda completed, active=session: self._on_motion_done(
                        active, completed))
                return
            except Exception as exc:
                self._append_attempt_locked(
                    session, MotionStatus.PLANNER_ERROR,
                    f"failed to submit motion request: {exc}")
                self._finish_locked(
                    session, CandidatePlanningStatus.SYSTEM_ERROR,
                    f"MotionCoordinator invocation failed: {exc}")
                return

    def _append_attempt_locked(self, session: _Session, status: MotionStatus,
                               detail: str, *, failure_code: int = 0,
                               failed_stage: str = "",
                               moveit_error_code: int = 0) -> None:
        candidate = session.pending_candidate
        session.attempts.append(CandidateAttempt(
            object_id=session.object_id, rank=int(candidate.rank),
            source_index=int(getattr(candidate, "source_index", -1)),
            ranking_score=float(candidate.ranking_score),
            motion_status=status, detail=detail,
            duration_ms=max(0.0, (time.monotonic() - session.pending_started_at) * 1000.0),
            failure_code=failure_code, failed_stage=failed_stage,
            moveit_error_code=moveit_error_code,
        ))

    def _consume_locked(self, session: _Session, pending: Any) -> bool:
        """Return True to start the next rank, False if operation finished."""
        if self._active is not session or session.pending_future is not pending:
            return False
        candidate = session.pending_candidate
        try:
            result = pending.result()
            if not isinstance(result, MotionResult):
                raise ValueError("MotionCoordinator returned an unexpected result type")
            if not isinstance(result.status, MotionStatus):
                raise ValueError("MotionCoordinator returned an unknown motion status")
            if (result.scene_id != session.scene_id
                    or result.object_id != session.object_id
                    or result.rank != int(candidate.rank)):
                raise ValueError("MotionCoordinator returned mismatched scene/object/rank")
        except Exception as exc:
            self._append_attempt_locked(session, MotionStatus.INVALID_RESPONSE,
                                        f"invalid motion result: {exc}")
            self._finish_locked(session, CandidatePlanningStatus.SYSTEM_ERROR,
                                f"MotionCoordinator result error: {exc}")
            return False

        self._append_attempt_locked(
            session, result.status, result.detail,
            failure_code=result.failure_code, failed_stage=result.failed_stage,
            moveit_error_code=result.moveit_error_code,
        )
        session.pending_future = None
        session.pending_candidate = None
        if result.status == MotionStatus.PLANNED:
            if result.failure_code != 0 or not _complete_plan_present(result):
                self._finish_locked(
                    session, CandidatePlanningStatus.SYSTEM_ERROR,
                    "MotionCoordinator reported success without complete pick plan")
                return False
            self._finish_locked(
                session, CandidatePlanningStatus.PICK_READY,
                f"selected suction candidate rank {int(candidate.rank)}",
                selected_candidate=candidate, motion_result=result)
            return False

        # The enum and the explicit flag must BOTH approve retry. All unknown
        # planner errors, TF outages, timeouts, scene problems etc. stop here.
        if (result.status in (MotionStatus.INVALID_CANDIDATE,
                              MotionStatus.CANDIDATE_REJECTED)
                and result.retry_next_candidate):
            return True
        self._finish_locked(
            session, CandidatePlanningStatus.SYSTEM_ERROR,
            f"candidate rank {int(candidate.rank)}: {result.status.value}: {result.detail}")
        return False

    def _on_motion_done(self, session: _Session, pending: Any) -> None:
        with self._lock:
            if self._active is not session or session.pending_future is not pending:
                return  # cancelled, replaced, or an old reply
            if self._consume_locked(session, pending):
                self._advance_locked(session)

    def cancel_scene(self, scene_id: str) -> bool:
        """Cancel local selection and invalidate late replies (not C++ work)."""
        with self._lock:
            session = self._active
            if session is None or session.scene_id != scene_id:
                return False
            # Invalidate the session *before* asking MotionCoordinator to
            # cancel: its Future may complete synchronously during cancellation.
            # Inform the caller only after downstream cancellation is requested,
            # so a result callback cannot start another request for this scene
            # that would then accidentally be cancelled by scene_id.
            self._active = None
            detail = "candidate planning was cancelled; late results ignored"
            try:
                self._motion.cancel_scene(scene_id)
            except Exception as exc:
                detail += f" (downstream cancellation raised: {exc})"
            if not session.output.done():
                session.output.set_result(CandidatePlanningResult(
                    CandidatePlanningStatus.CANCELLED, session.scene_id,
                    session.scene_index, session.object_id, detail,
                    attempts=tuple(session.attempts),
                ))
            return True

    def close(self) -> None:
        """Cancel any pending candidate evaluation; does not close MotionCoordinator."""
        with self._lock:
            self._closed = True
            if self._active is not None:
                self.cancel_scene(self._active.scene_id)
