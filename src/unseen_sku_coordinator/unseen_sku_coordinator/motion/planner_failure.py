"""Classify structured C++ PlanCandidate service failures, failing closed."""

from typing import Any

from unseen_sku_motion_planning.srv import PlanCandidate

from .motion_types import MotionStatus


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
