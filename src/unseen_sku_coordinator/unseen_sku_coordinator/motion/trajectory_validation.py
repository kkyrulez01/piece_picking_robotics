"""Validate the shape and continuity of a successful planning response.

This does NOT prove collision clearance or physical execution safety.
"""

import math
from typing import Any

from .candidate_geometry import _rotate, _xyz


JOINT_NAMES = tuple(f"Revolute_{index}" for index in range(1, 7))


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
