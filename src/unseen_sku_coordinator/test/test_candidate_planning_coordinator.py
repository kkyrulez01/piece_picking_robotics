"""Unit tests for ranked, planning-only candidate selection (no ROS graph).

On ROS 2 Humble:
python3 -m unittest discover -s src/unseen_sku_coordinator/test \
    -p 'test_candidate_planning_coordinator.py' -v
"""

import math
import unittest
from collections import deque

from rclpy.task import Future
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint
from unseen_sku_interfaces.msg import (
    SuctionCandidate,
    SuctionCandidateArray,
    SuctionResult,
)
from unseen_sku_motion_planning.srv import PlanCandidate

from unseen_sku_coordinator.candidate.candidate_planning_coordinator import (
    CandidatePlanningCoordinator,
    CandidatePlanningStatus,
)
from unseen_sku_coordinator.motion.motion_coordinator import MotionResult, MotionStatus


def candidate(object_id=2, rank=1, score=0.8):
    item = SuctionCandidate()
    item.object_id = object_id
    item.rank = rank
    item.ranking_score = score
    item.position.z = 0.4
    item.normal.z = 1.0
    return item


def suction_scene(candidates=(), *, scene_id="scene-42", status="OK"):
    result = SuctionResult()
    result.scene_id = scene_id
    array = SuctionCandidateArray()
    array.scene_index = 42
    array.status = status
    array.header = Header()
    array.header.frame_id = "base_link"
    array.header.stamp.sec = 1791976782
    array.header.stamp.nanosec = 123456789
    array.candidates = list(candidates)
    result.candidates = array
    return result


def motion_result(rank, status=MotionStatus.PLANNED, *, scene_id="scene-42",
                  object_id=2, detail="", failure_code=0,
                  failed_stage="", moveit_error_code=0,
                  complete=True):
    response = None
    if status == MotionStatus.PLANNED:
        response = PlanCandidate.Response()
        if complete:
            for trajectory_name in (
                "pre_pick_trajectory", "pick_trajectory", "retract_trajectory"
            ):
                trajectory = getattr(response, trajectory_name)
                trajectory.points = [JointTrajectoryPoint()]
    return MotionResult(
        status=status, detail=detail or status.value,
        scene_id=scene_id, object_id=object_id, rank=rank,
        planning_response=response, failure_code=failure_code,
        failed_stage=failed_stage, moveit_error_code=moveit_error_code,
    )


class FakeMotionCoordinator:
    def __init__(self, outcomes=()):
        self.outcomes = deque(outcomes)
        self.calls = []
        self.cancelled = []

    def plan_candidate_async(self, candidate_item, header, scene_id):
        self.calls.append((candidate_item, header, scene_id))
        if not self.outcomes:
            raise AssertionError("unexpected extra MotionCoordinator call")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, Future):
            return outcome
        future = Future()
        future.set_result(outcome)
        return future

    def cancel_scene(self, scene_id):
        self.cancelled.append(scene_id)


class TestCandidatePlanningCoordinator(unittest.TestCase):
    def run_plan(self, scene, outcomes, *, selected_object=2, limit=5):
        motion = FakeMotionCoordinator(outcomes)
        planner = CandidatePlanningCoordinator(motion, max_candidate_attempts=limit)
        result = planner.plan_object_async(
            scene, selected_object, expected_scene_id="scene-42"
        )
        self.assertTrue(result.done())
        return result.result(), motion

    def test_best_rank_wins_even_when_input_unsorted_or_scores_conflict(self):
        scene = suction_scene([
            candidate(3, 1, 0.99), candidate(2, 3, 0.95),
            candidate(2, 1, 0.30), candidate(2, 2, 0.90),
        ])
        outcome, motion = self.run_plan(scene, [motion_result(1)])
        self.assertEqual(outcome.status, CandidatePlanningStatus.PICK_READY)
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.selected_rank, 1)
        self.assertEqual(outcome.selected_candidate.object_id, 2)
        self.assertEqual([call[0].rank for call in motion.calls], [1])
        self.assertEqual(outcome.scene_id, "scene-42")
        self.assertEqual(outcome.scene_index, 42)
        self.assertIsNotNone(outcome.planning_response)

    def test_retry_only_candidate_specific_and_stop_at_first_valid_plan(self):
        scene = suction_scene([
            candidate(2, 3), candidate(2, 1), candidate(2, 4), candidate(2, 2),
        ])
        responses = [
            motion_result(1, MotionStatus.INVALID_CANDIDATE,
                          detail="bad surface normal", failure_code=2,
                          failed_stage="INPUT"),
            motion_result(2, MotionStatus.CANDIDATE_REJECTED,
                          detail="NO_IK_SOLUTION", failure_code=3,
                          failed_stage="PICK_LIN", moveit_error_code=-31),
            motion_result(3),
        ]
        outcome, motion = self.run_plan(scene, responses)
        self.assertEqual(outcome.status, CandidatePlanningStatus.PICK_READY)
        self.assertEqual(outcome.selected_rank, 3)
        self.assertEqual([call[0].rank for call in motion.calls], [1, 2, 3])
        self.assertEqual([a.rank for a in outcome.attempts], [1, 2, 3])
        self.assertEqual(outcome.attempts[1].failure_code, 3)
        self.assertEqual(outcome.attempts[1].failed_stage, "PICK_LIN")
        self.assertEqual(outcome.attempts[1].moveit_error_code, -31)
        self.assertGreaterEqual(outcome.attempts[0].duration_ms, 0)

    def test_all_candidates_rejected(self):
        scene = suction_scene([candidate(2, 1), candidate(2, 2)])
        outcome, motion = self.run_plan(scene, [
            motion_result(1, MotionStatus.CANDIDATE_REJECTED),
            motion_result(2, MotionStatus.INVALID_CANDIDATE),
        ])
        self.assertEqual(outcome.status, CandidatePlanningStatus.NO_FEASIBLE_CANDIDATE)
        self.assertIsNone(outcome.selected_candidate)
        self.assertIsNone(outcome.planning_response)
        self.assertEqual(len(outcome.attempts), 2)
        self.assertEqual(len(motion.calls), 2)

    def test_attempt_cap_does_not_claim_all_candidates_were_checked(self):
        scene = suction_scene([candidate(2, i) for i in range(1, 5)])
        outcome, motion = self.run_plan(scene, [
            motion_result(1, MotionStatus.CANDIDATE_REJECTED),
            motion_result(2, MotionStatus.CANDIDATE_REJECTED),
        ], limit=2)
        self.assertEqual(outcome.status, CandidatePlanningStatus.ATTEMPT_LIMIT_REACHED)
        self.assertEqual([call[0].rank for call in motion.calls], [1, 2])

    def test_exhausted_exactly_at_cap_is_no_feasible(self):
        scene = suction_scene([candidate(2, 1), candidate(2, 2)])
        outcome, _ = self.run_plan(scene, [
            motion_result(1, MotionStatus.INVALID_CANDIDATE),
            motion_result(2, MotionStatus.INVALID_CANDIDATE),
        ], limit=2)
        self.assertEqual(outcome.status, CandidatePlanningStatus.NO_FEASIBLE_CANDIDATE)

    def test_generic_moveit_failure_does_not_retry(self):
        scene = suction_scene([candidate(2, 1), candidate(2, 2)])
        outcome, motion = self.run_plan(scene, [
            motion_result(1, MotionStatus.PLANNING_FAILED, detail="MoveIt generic failure")
        ])
        self.assertEqual(outcome.status, CandidatePlanningStatus.SYSTEM_ERROR)
        self.assertEqual(len(motion.calls), 1)
        self.assertEqual(outcome.attempts[0].motion_status, MotionStatus.PLANNING_FAILED)

    def test_system_errors_are_not_rank_fallback(self):
        fail_statuses = [
            MotionStatus.INVALID_REQUEST, MotionStatus.STALE_CANDIDATE,
            MotionStatus.TF_UNAVAILABLE, MotionStatus.SERVICE_UNAVAILABLE,
            MotionStatus.SERVICE_TIMEOUT, MotionStatus.ROBOT_STATE_UNAVAILABLE,
            MotionStatus.PLANNER_ERROR, MotionStatus.INVALID_RESPONSE,
            MotionStatus.BUSY, MotionStatus.CANCELLED,
        ]
        for status in fail_statuses:
            with self.subTest(status=status):
                scene = suction_scene([candidate(2, 1), candidate(2, 2)])
                outcome, motion = self.run_plan(scene, [motion_result(1, status)])
                self.assertEqual(outcome.status, CandidatePlanningStatus.SYSTEM_ERROR)
                self.assertEqual(len(motion.calls), 1)

    def test_wrong_scene_id_rejected_without_planning(self):
        scene = suction_scene([candidate(2, 1)], scene_id="old-scene")
        outcome, motion = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)
        self.assertEqual(motion.calls, [])

    def test_raw_helios_frame_rejected_without_planning(self):
        scene = suction_scene([candidate(2, 1)])
        scene.candidates.header.frame_id = "helios2_optical_frame"
        outcome, motion = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)
        self.assertFalse(motion.calls)

    def test_empty_expected_scene_id_rejected(self):
        planner = CandidatePlanningCoordinator(FakeMotionCoordinator())
        output = planner.plan_object_async(
            suction_scene([candidate(2, 1)]), 2, expected_scene_id="")
        self.assertEqual(output.result().status, CandidatePlanningStatus.INVALID_REQUEST)

    def test_no_candidate_for_selected_object_does_not_switch_objects(self):
        scene = suction_scene([candidate(3, 1), candidate(4, 1)])
        outcome, motion = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.NO_CANDIDATES)
        self.assertEqual(motion.calls, [])

    def test_no_candidates_status_no_motion(self):
        scene = suction_scene([], status="NO_CANDIDATES")
        outcome, motion = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.NO_CANDIDATES)
        self.assertFalse(motion.calls)

    def test_scoring_error_rejected_without_motion(self):
        scene = suction_scene([], status="ERROR")
        scene.candidates.error_message = "no depth"
        outcome, motion = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)
        self.assertIn("no depth", outcome.detail)
        self.assertFalse(motion.calls)

    def test_ok_but_zero_candidates_is_invalid(self):
        outcome, _ = self.run_plan(suction_scene([]), [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)

    def test_no_candidates_but_populated_is_invalid(self):
        outcome, _ = self.run_plan(
            suction_scene([candidate(2, 1)], status="NO_CANDIDATES"), [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)

    def test_duplicate_selected_ranks_are_invalid(self):
        scene = suction_scene([candidate(2, 1), candidate(2, 1)])
        outcome, motion = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)
        self.assertFalse(motion.calls)

    def test_invalid_score_is_invalid_scene(self):
        scene = suction_scene([candidate(2, 1, math.nan)])
        outcome, _ = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)

    def test_missing_frame_or_timestamp_rejected(self):
        scene = suction_scene([candidate(2, 1)])
        scene.candidates.header.frame_id = ""
        outcome, _ = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)
        scene.candidates.header.frame_id = "base_link"
        scene.candidates.header.stamp.sec = 0
        scene.candidates.header.stamp.nanosec = 0
        outcome, _ = self.run_plan(scene, [])
        self.assertEqual(outcome.status, CandidatePlanningStatus.INVALID_REQUEST)

    def test_invalid_target_object_id_rejected(self):
        for bad_id in (0, -1, True, 2.5, "2", None):
            with self.subTest(bad_id=bad_id):
                result, _ = self.run_plan(suction_scene([candidate(2, 1)]), [],
                                          selected_object=bad_id)
                self.assertEqual(result.status, CandidatePlanningStatus.INVALID_REQUEST)

    def test_incomplete_successful_motion_response_is_rejected(self):
        scene = suction_scene([candidate(2, 1), candidate(2, 2)])
        outcome, motion = self.run_plan(scene, [motion_result(1, complete=False)])
        self.assertEqual(outcome.status, CandidatePlanningStatus.SYSTEM_ERROR)
        self.assertEqual(len(motion.calls), 1)

    def test_success_with_nonzero_failure_code_is_rejected(self):
        scene = suction_scene([candidate(2, 1), candidate(2, 2)])
        outcome, motion = self.run_plan(scene, [
            motion_result(1, MotionStatus.PLANNED, failure_code=7)])
        self.assertEqual(outcome.status, CandidatePlanningStatus.SYSTEM_ERROR)
        self.assertEqual(len(motion.calls), 1)

    def test_unknown_motion_status_is_rejected(self):
        scene = suction_scene([candidate(2, 1), candidate(2, 2)])
        invalid = motion_result(1)
        # Frozen dataclass can be constructed with a bad enum by a buggy caller.
        invalid = MotionResult(status="planned", detail="incorrect type",
                               scene_id="scene-42", object_id=2, rank=1,
                               planning_response=invalid.planning_response)
        outcome, motion = self.run_plan(scene, [invalid])
        self.assertEqual(outcome.status, CandidatePlanningStatus.SYSTEM_ERROR)
        self.assertEqual(len(motion.calls), 1)

    def test_wrong_scene_or_candidate_in_motion_result_stops(self):
        for wrong_result in (
            motion_result(1, scene_id="scene-other"),
            motion_result(1, object_id=5),
            motion_result(10),
        ):
            with self.subTest(result=wrong_result):
                scene = suction_scene([candidate(2, 1), candidate(2, 2)])
                outcome, motion = self.run_plan(scene, [wrong_result])
                self.assertEqual(outcome.status, CandidatePlanningStatus.SYSTEM_ERROR)
                self.assertEqual(len(motion.calls), 1)

    def test_exceptional_and_malformed_motion_result_stop(self):
        exceptional = Future()
        exceptional.set_exception(RuntimeError("service callback crashed"))
        for bad_reply in (exceptional, "not-a-MotionResult", RuntimeError("cannot submit")):
            with self.subTest(type=type(bad_reply)):
                scene = suction_scene([candidate(2, 1), candidate(2, 2)])
                outcome, motion = self.run_plan(scene, [bad_reply])
                self.assertEqual(outcome.status, CandidatePlanningStatus.SYSTEM_ERROR)
                self.assertEqual(len(motion.calls), 1)

    def test_async_fallback_waits_for_each_result(self):
        pending_first = Future()
        pending_second = Future()
        motion = FakeMotionCoordinator([pending_first, pending_second])
        planner = CandidatePlanningCoordinator(motion)
        scene = suction_scene([candidate(2, 2), candidate(2, 1)])
        future = planner.plan_object_async(scene, 2, expected_scene_id="scene-42")
        self.assertFalse(future.done())
        self.assertEqual([call[0].rank for call in motion.calls], [1])
        pending_first.set_result(motion_result(1, MotionStatus.INVALID_CANDIDATE))
        self.assertFalse(future.done())
        self.assertEqual([call[0].rank for call in motion.calls], [1, 2])
        pending_second.set_result(motion_result(2))
        self.assertTrue(future.done())
        self.assertEqual(future.result().status, CandidatePlanningStatus.PICK_READY)
        self.assertEqual(future.result().selected_rank, 2)

    def test_busy_is_rejected_without_disturbing_active_session(self):
        pending = Future()
        motion = FakeMotionCoordinator([pending])
        planner = CandidatePlanningCoordinator(motion)
        scene = suction_scene([candidate(2, 1)])
        first = planner.plan_object_async(scene, 2, expected_scene_id="scene-42")
        second = planner.plan_object_async(scene, 2, expected_scene_id="scene-42")
        self.assertEqual(second.result().status, CandidatePlanningStatus.BUSY)
        self.assertFalse(first.done())
        pending.set_result(motion_result(1))
        self.assertEqual(first.result().status, CandidatePlanningStatus.PICK_READY)
        self.assertEqual(len(motion.calls), 1)

    def test_cancel_ignores_late_result_and_new_session_isolated(self):
        old_pending = Future()
        motion = FakeMotionCoordinator([old_pending, motion_result(1, scene_id="scene-99")])
        planner = CandidatePlanningCoordinator(motion)
        first = planner.plan_object_async(
            suction_scene([candidate(2, 1)]), 2, expected_scene_id="scene-42")
        self.assertFalse(planner.cancel_scene("not-this-scene"))
        self.assertTrue(planner.cancel_scene("scene-42"))
        self.assertEqual(first.result().status, CandidatePlanningStatus.CANCELLED)
        self.assertEqual(motion.cancelled, ["scene-42"])
        second = planner.plan_object_async(
            suction_scene([candidate(2, 1)], scene_id="scene-99"),
            2, expected_scene_id="scene-99")
        self.assertEqual(second.result().status, CandidatePlanningStatus.PICK_READY)
        old_pending.set_result(motion_result(1))
        self.assertEqual(first.result().status, CandidatePlanningStatus.CANCELLED)
        self.assertEqual(second.result().status, CandidatePlanningStatus.PICK_READY)

    def test_result_snapshot_is_not_affected_by_input_mutation(self):
        pending = Future()
        motion = FakeMotionCoordinator([pending])
        planner = CandidatePlanningCoordinator(motion)
        scene = suction_scene([candidate(2, 1)])
        future = planner.plan_object_async(scene, 2, expected_scene_id="scene-42")
        scene.candidates.candidates[0].rank = 99
        scene.candidates.header.frame_id = "wrong_frame"
        self.assertEqual(motion.calls[0][0].rank, 1)
        self.assertEqual(motion.calls[0][1].frame_id, "base_link")
        pending.set_result(motion_result(1))
        self.assertEqual(future.result().selected_rank, 1)

    def test_many_immediate_rejections_do_not_recurse(self):
        count = 250
        scene = suction_scene([candidate(2, rank) for rank in range(1, count + 1)])
        outcomes = [motion_result(rank, MotionStatus.INVALID_CANDIDATE)
                    for rank in range(1, count + 1)]
        result, motion = self.run_plan(scene, outcomes, limit=count)
        self.assertEqual(result.status, CandidatePlanningStatus.NO_FEASIBLE_CANDIDATE)
        self.assertEqual(len(motion.calls), count)
        self.assertEqual(len(result.attempts), count)

    def test_close_cancels_and_rejects_subsequent_work(self):
        pending = Future()
        motion = FakeMotionCoordinator([pending])
        planner = CandidatePlanningCoordinator(motion)
        scene = suction_scene([candidate(2, 1)])
        future = planner.plan_object_async(scene, 2, expected_scene_id="scene-42")
        planner.close()
        self.assertEqual(future.result().status, CandidatePlanningStatus.CANCELLED)
        new_future = planner.plan_object_async(scene, 2, expected_scene_id="scene-42")
        self.assertEqual(new_future.result().status, CandidatePlanningStatus.INVALID_REQUEST)

    def test_invalid_attempt_limit_rejected(self):
        for bad_limit in (0, -1, True, 2.5, "5"):
            with self.subTest(limit=bad_limit):
                with self.assertRaises(ValueError):
                    CandidatePlanningCoordinator(FakeMotionCoordinator(),
                                                 max_candidate_attempts=bad_limit)


if __name__ == "__main__":
    unittest.main()
