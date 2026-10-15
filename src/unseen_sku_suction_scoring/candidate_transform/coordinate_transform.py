#!/usr/bin/env python3

"""
Coordinate transformation utilities for suction candidates.

Transforms:
    - Suction point: rotation + translation
    - Surface normal: rotation only

The TransformStamped is obtained by the ROS 2 node.
"""

import copy
import math

from geometry_msgs.msg import Point, Vector3, TransformStamped

def rotate_vector(x, y, z, quaternion):
    """
    Rotate a 3D vector using a quaternion.

    Args:
        x, y, z:
            Vector components.

        quaternion:
            geometry_msgs/Quaternion.

    Returns:
        Rotated (x, y, z).
    """
    x = float(x)
    y = float(y)
    z = float(z)

    qx = float(quaternion.x)
    qy = float(quaternion.y)
    qz = float(quaternion.z)
    qw = float(quaternion.w)

    values = (x, y, z, qx, qy, qz, qw)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite vector or quaternion")

    # Normalize quaternion
    magnitude = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if magnitude < 1e-12:
        raise ValueError("Invalid zero-length quaternion")

    qx /= magnitude
    qy /= magnitude
    qz /= magnitude
    qw /= magnitude

    # Quaternion rotation:
    # t = 2 * cross(q_xyz, vector)
    # rotated = vector + qw * t + cross(q_xyz, t)

    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)

    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)

    return rx, ry, rz

def transform_point(point, transform) -> Point:
    """
    Transform a suction contact point by applying rotation
    and translation.

    P_target = R * P_source + t
    """
    rotation = transform.transform.rotation
    translation = transform.transform.translation

    x, y, z = rotate_vector(point.x, point.y, point.z, rotation)

    if not all(math.isfinite(v) for v in (translation.x, translation.y, translation.z)):
        raise ValueError("Non-finite TF translation")

    result = Point()
    result.x = x + translation.x
    result.y = y + translation.y
    result.z = z + translation.z

    return result

def transform_normal(normal: Vector3, transform) -> Vector3:
    """
    Transform a suction surface normal by only applying rotation.

    N_target = R * N_source
    """
    rotation = transform.transform.rotation
    nx, ny, nz = rotate_vector(normal.x, normal.y, normal.z,rotation)

    # Normalize transformed normal
    magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
    if magnitude < 1e-12:
        raise ValueError("Invalid zero-length surface normal")

    result = Vector3()
    result.x = nx / magnitude
    result.y = ny / magnitude
    result.z = nz / magnitude

    return result

def transform_candidate(candidate, transform):
    """
    Transform a suction candidate into the target frame.

    Only the position and normal is modified, all other fields are preserved.
    """
    result = copy.deepcopy(candidate)

    result.position = transform_point(candidate.position, transform)
    result.normal = transform_normal(candidate.normal, transform)

    return result
