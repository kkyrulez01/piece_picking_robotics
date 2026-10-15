"""Wire conversions for planning-only ROS actions; no robot execution."""

from action_msgs.msg import GoalStatus
from unseen_sku_interfaces.action import PlanSuctionCandidate
from unseen_sku_interfaces.msg import PickPlan
from unseen_sku_motion_planning.srv import PlanCandidate

from .motion_types import MotionResult, MotionStatus
from .trajectory_validation import validate_plan_response


_PLAN_FIELDS = (
    "pre_pick_pose", "pick_pose", "retract_pose",
    "pre_pick_trajectory", "pick_trajectory", "retract_trajectory",
)


def pack_pick_plan(plan_response):
    """Copy the six planned poses/trajectories from the C++ service response."""
    msg = PickPlan()
    if plan_response is not None:
        for name in _PLAN_FIELDS:
            setattr(msg, name, getattr(plan_response, name))
    return msg


def unpack_pick_plan(msg):
    """Recreate the C++ service response shape expected by the Python core."""
    response = PlanCandidate.Response()
    response.success = True
    response.failure_code = PlanCandidate.Response.FAILURE_NONE
    for name in _PLAN_FIELDS:
        setattr(response, name, getattr(msg, name))
    return response


def to_motion_action_result(motion):
    """Encode one MotionResult; never imply execution occurred."""
    wire = PlanSuctionCandidate.Result()
    wire.success = bool(motion.success)
    wire.status = motion.status.value
    wire.detail = motion.detail
    wire.scene_id = motion.scene_id
    wire.object_id = motion.object_id
    wire.rank = motion.rank
    wire.retry_next_candidate = bool(motion.retry_next_candidate)
    wire.failure_code = motion.failure_code
    wire.failed_stage = motion.failed_stage
    wire.moveit_error_code = motion.moveit_error_code
    if motion.suction_point_base is not None:
        wire.suction_point_base = motion.suction_point_base
    if motion.surface_normal_base is not None:
        wire.surface_normal_base = motion.surface_normal_base
    if motion.success:
        wire.plan = pack_pick_plan(motion.planning_response)
    return wire


def from_motion_action_result(wire, terminal_status, scene_id, object_id, rank):
    """Fail closed on malformed/contradictory action results.

    An ABORTED action can legitimately represent a candidate-specific IK
    rejection, which is rank-retryable. A CANCELED result is never retryable.
    """
    def error(detail):
        return MotionResult(
            MotionStatus.PLANNER_ERROR, detail, scene_id, object_id, rank)

    if terminal_status == GoalStatus.STATUS_CANCELED:
        return MotionResult(
            MotionStatus.CANCELLED, "motion action was cancelled",
            scene_id, object_id, rank)
    if terminal_status not in (
            GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED):
        return error(f"unexpected ROS action terminal status {terminal_status}")
    if wire is None:
        return error("motion action returned no result")
    if (wire.scene_id != scene_id or wire.object_id != object_id
            or wire.rank != rank):
        return error("motion action returned mismatched scene/object/rank")
    try:
        status = MotionStatus(wire.status)
    except ValueError:
        return error(f"unrecognized motion action status {wire.status!r}")
    is_success = status == MotionStatus.PLANNED
    if (bool(wire.success) != is_success
            or bool(wire.retry_next_candidate) != (
                status in (MotionStatus.INVALID_CANDIDATE,
                           MotionStatus.CANDIDATE_REJECTED))
            or (terminal_status == GoalStatus.STATUS_SUCCEEDED) != is_success):
        return error("contradictory motion action status/success/retry fields")
    if (is_success and wire.failure_code != 0):
        return error("successful motion action returned a failure code")
    if status == MotionStatus.CANDIDATE_REJECTED and wire.failure_code != (
            PlanCandidate.Response.FAILURE_CANDIDATE_INFEASIBLE):
        return error("candidate rejection missing explicit infeasibility code")

    response = None
    if is_success:
        try:
            response = unpack_pick_plan(wire.plan)
            frame = wire.suction_point_base.header.frame_id
            if frame != "base_link" or wire.surface_normal_base.header.frame_id != "base_link":
                raise ValueError("planning action must return base_link point and normal")
            validate_plan_response(
                response, wire.suction_point_base, wire.surface_normal_base, frame)
        except (AttributeError, TypeError, ValueError) as exc:
            return error(f"invalid motion action pick plan: {exc}")

    return MotionResult(
        status=status, detail=wire.detail, scene_id=scene_id,
        object_id=object_id, rank=rank,
        suction_point_base=wire.suction_point_base if is_success else None,
        surface_normal_base=wire.surface_normal_base if is_success else None,
        planning_response=response,
        failure_code=wire.failure_code,
        failed_stage=wire.failed_stage,
        moveit_error_code=wire.moveit_error_code,
    )
