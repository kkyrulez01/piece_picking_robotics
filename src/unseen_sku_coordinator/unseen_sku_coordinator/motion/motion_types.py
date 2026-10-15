"""Result types and the internal state of one asynchronous planning attempt."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from rclpy.task import Future


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
