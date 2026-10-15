"""Validate a pre-transformed base-frame suction candidate; no TF conversion.

The upstream candidate_transform ROS node owns Helios2 -> base_link geometry.
This module only copies a verified base-frame point and normal to stamped ROS
messages for the C++ PlanCandidate service. Quaternion rotation is retained
for validating the returned TCP orientation, not for camera transformation.
"""

import math
from typing import Any

from geometry_msgs.msg import PointStamped, Vector3Stamped


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
    """Rotate a direction for *trajectory orientation checks*, not camera TF."""
    x, y, z, w = (
        float(quaternion.x), float(quaternion.y),
        float(quaternion.z), float(quaternion.w),
    )
    if not all(math.isfinite(v) for v in (x, y, z, w)):
        raise ValueError("nonfinite orientation quaternion")
    length = math.sqrt(x*x + y*y + z*z + w*w)
    if length < 1e-9:
        raise ValueError("zero-length orientation quaternion")
    x, y, z, w = (x/length, y/length, z/length, w/length)
    vx, vy, vz = vector
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


def prepare_base_candidate(candidate: Any, header: Any,
                           base_frame: str = "base_link") -> tuple:
    """Return stamped point/normal without changing reference coordinates.

    Fail closed if a camera-frame candidate bypasses upstream transformation.
    """
    if str(header.frame_id) != base_frame:
        raise ValueError(
            f"candidate frame {header.frame_id!r} is not {base_frame!r}; "
            "upstream candidate transformation is required"
        )
    position = _xyz(candidate.position)
    if not all(math.isfinite(v) for v in position):
        raise ValueError("nonfinite suction position")
    normal = _normalized(_xyz(candidate.normal))

    point = PointStamped()
    point.header.frame_id = base_frame
    point.header.stamp = header.stamp
    point.point.x, point.point.y, point.point.z = position

    vector = Vector3Stamped()
    vector.header.frame_id = base_frame
    vector.header.stamp = header.stamp
    vector.vector.x, vector.vector.y, vector.vector.z = normal
    return point, vector
