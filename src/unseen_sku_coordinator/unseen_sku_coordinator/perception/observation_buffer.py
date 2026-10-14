"""Track recent perception metadata without storing full sensor messages."""

from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from .scene_context import ObservationStamp


@dataclass(frozen=True)
class ReceivedObservation:
    """Metadata and optional reference to a received ROS 2 message."""

    stamp: ObservationStamp
    received_monotonic: float
    scene_index: int | None = None
    object_count: int | None = None

    # Only populated for Helios2 clouds and SAM2 masks.
    # Store a reference, not a duplicate of the message.
    payload: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )


def timestamp_difference_ms(
    first: ObservationStamp,
    second: ObservationStamp,
) -> float:
    """Difference between timestamps, assuming comparable clocks."""

    first_ns = first.sec * 1_000_000_000 + first.nanosec
    second_ns = second.sec * 1_000_000_000 + second.nanosec

    return abs(first_ns - second_ns) / 1_000_000.0


class ObservationBuffer:
    """Store recent RealSense observations and latest source metadata."""

    def __init__(self, rgbd_history_size: int):

        if type(rgbd_history_size) is not int or rgbd_history_size <= 0:
            raise ValueError(
                "rgbd_history_size must be a positive integer"
            )

        self._lock = Lock()

        # Latest metadata for each perception source.
        self._latest = {}

        # Recent RealSense frames, metadata only.
        self._recent_rgbd = deque(maxlen=rgbd_history_size)

    def record(
        self,
        source: str,
        observation: ReceivedObservation,
    ) -> None:
        """Record a received observation."""

        with self._lock:

            self._latest[source] = observation

            if source == "realsense":
                self._recent_rgbd.append(observation)

    def get_latest(self) -> dict:
        """Return the latest observations."""

        with self._lock:
            return dict(self._latest)

    def find_realsense_by_stamp(
        self,
        stamp: ObservationStamp,
        not_before_monotonic: float,
    ) -> ReceivedObservation | None:
        """Find the RealSense observation used by SAM2.

        Match both timestamp and frame ID.

        Exclude observations received before the current
        perception transaction started.
        """

        with self._lock:

            for observation in reversed(self._recent_rgbd):

                if observation.received_monotonic < not_before_monotonic:
                    break

                if observation.stamp == stamp:
                    return observation

        return None

    def recent_realsense_count(self) -> int:
        """Return the number of cached RealSense observations."""

        with self._lock:
            return len(self._recent_rgbd)