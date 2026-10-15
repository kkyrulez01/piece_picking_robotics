"""Convert ranked candidate coordinator results to ROS action messages."""

from unseen_sku_interfaces.action import PlanSuctionObject
from unseen_sku_interfaces.msg import CandidatePlanningAttempt

from unseen_sku_coordinator.motion.action_bridge import pack_pick_plan


def to_candidate_action_result(result, candidate_header=None):
    msg = PlanSuctionObject.Result()
    msg.success = result.success
    msg.status = result.status.value
    msg.detail = result.detail
    msg.scene_id = result.scene_id
    msg.scene_index = result.scene_index
    msg.target_object_id = result.target_object_id
    for attempt in result.attempts:
        wire = CandidatePlanningAttempt()
        wire.object_id = attempt.object_id
        wire.rank = attempt.rank
        wire.source_index = attempt.source_index
        wire.ranking_score = attempt.ranking_score
        wire.motion_status = attempt.motion_status.value
        wire.detail = attempt.detail
        wire.duration_ms = attempt.duration_ms
        wire.failure_code = attempt.failure_code
        wire.failed_stage = attempt.failed_stage
        wire.moveit_error_code = attempt.moveit_error_code
        msg.attempts.append(wire)
    if result.success:
        msg.selected_rank = result.selected_rank
        msg.selected_candidate = result.selected_candidate
        msg.candidate_header = candidate_header
        msg.plan = pack_pick_plan(result.planning_response)
    return msg
