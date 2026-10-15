"""Transform a suction candidate position and surface normal between frames.

A point receives rotation and translation; a direction receives rotation only.
"""

import math
from typing import Any, Optional

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
