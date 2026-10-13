"""Metadata for one perception request.

This module has no ROS dependencies and stores no large sensor arrays.
"""

from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from ..common.coordinator_states import (
    CoordinatorState,
    FailureCode,
    ResultStatus,
    TERMINAL_STATES,
)


@dataclass(frozen=True)
class ObservationStamp:
    """Source message timestamp and coordinate frame."""

    sec: int
    nanosec: int
    frame_id: str

    def __post_init__(self):
        if self.sec < 0 or not 0 <= self.nanosec < 1_000_000_000:
            raise ValueError("Invalid ROS timestamp")


@dataclass
class SceneContext:
    """State and metadata for one perception transaction."""

    # Unique coordinator-level identifiers
    request_id: str = field(default_factory=lambda: uuid4().hex)
    scene_id: str = field(default_factory=lambda: uuid4().hex)

    # Workflow
    state: CoordinatorState = CoordinatorState.REQUESTING_CAPTURE

    # SAM2's own scene index, assigned separately
    sam2_scene_index: int | None = None

    # Selected observations and timing information
    observations: dict[str, ObservationStamp] = field(default_factory=dict)
    stage_durations_sec: dict[str, float] = field(default_factory=dict)

    # Final result
    result_status: ResultStatus | None = None
    failure_code: FailureCode = FailureCode.NONE
    failure_detail: str = ""

    # Monotonic times are used for durations, not sensor synchronization
    started_monotonic: float = field(default_factory=monotonic)
    stage_started_monotonic: float = field(init=False)
    finished_monotonic: float | None = None

    def __post_init__(self):
        self.stage_started_monotonic = self.started_monotonic

    @property
    def is_finished(self) -> bool:
        return self.result_status is not None

    @property
    def elapsed_sec(self) -> float:
        end = self.finished_monotonic

        return max(
            0.0,
            (monotonic() if end is None else end)
            - self.started_monotonic,
        )

    def _record_stage_duration(self, now: float) -> None:
        key = self.state.value

        duration = max(
            0.0,
            now - self.stage_started_monotonic,
        )

        self.stage_durations_sec[key] = (
            self.stage_durations_sec.get(key, 0.0) + duration
        )

    def enter_state(self, next_state: CoordinatorState) -> None:
        """Move to a nonterminal processing stage."""

        if self.is_finished:
            raise RuntimeError("Cannot change a finished scene")

        if next_state in TERMINAL_STATES:
            raise ValueError("Use finish() for a terminal state")

        if next_state == self.state:
            return

        now = monotonic()

        self._record_stage_duration(now)

        self.state = next_state
        self.stage_started_monotonic = now

    def record_observation(
        self,
        source: str,
        stamp: ObservationStamp,
    ) -> None:
        """Record one selected source observation for this scene."""

        if self.is_finished:
            raise RuntimeError("Cannot change a finished scene")

        if source in self.observations:
            raise ValueError(
                f"Observation already selected: {source}"
            )

        self.observations[source] = stamp

    def finish(
        self,
        status: ResultStatus,
        failure_code: FailureCode = FailureCode.NONE,
        detail: str = "",
    ) -> None:
        """Finalize the scene transaction."""

        if self.is_finished:
            raise RuntimeError("Scene already finished")

        final_states = {
            ResultStatus.SUCCESS: CoordinatorState.COMPLETE,
            ResultStatus.NO_OBJECTS_DETECTED:
                CoordinatorState.NO_OBJECTS_DETECTED,
            ResultStatus.NO_VALID_CANDIDATES:
                CoordinatorState.NO_VALID_CANDIDATES,
            ResultStatus.FAILED: CoordinatorState.FAILED,
            ResultStatus.TIMEOUT: CoordinatorState.TIMEOUT,
            ResultStatus.CANCELLED: CoordinatorState.CANCELLED,
        }

        now = monotonic()

        self._record_stage_duration(now)

        self.state = final_states[status]
        self.result_status = status
        self.failure_code = failure_code
        self.failure_detail = detail
        self.finished_monotonic = now
