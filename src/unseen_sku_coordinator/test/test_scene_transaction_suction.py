"""Software-only regression tests for fused-scene to suction transaction handling.

Run from ~/unseen_sku_ws after sourcing ROS 2 and the colcon overlay:
  PYTHONPATH="$PWD/src/unseen_sku_coordinator:$PYTHONPATH" \
    python3 src/unseen_sku_coordinator/test/test_scene_transaction_suction.py

Uses plain Python message-shaped objects; does not start cameras, ROS 2 nodes,
or Sim-Suction inference. Assertions reflect the pre-refactor M3.4C behavior.
"""

import copy
from time import monotonic
from types import SimpleNamespace as NS
import unittest

from unseen_sku_coordinator.common.coordinator_states import (
    CoordinatorState,
    FailureCode,
    ResultStatus,
)
from unseen_sku_coordinator.perception.scene_context import (
    SceneContext,
    ObservationStamp,
)
from unseen_sku_coordinator.perception.scene_transaction import SceneTransaction


class FakeLogger:
    def info(self, message):
        pass


class FakeManager:
    def __init__(self):
        self.fusion_result = None
        self.suction_result = None
        self.suction_published = []
        self.fusion_published = []
        self.expected_suction = []
        self.cleared_fusion = []
        self.cleared_suction = []
        self.fail_suction_publish = False

    def get_fusion_result(self, scene_id):
        return self.fusion_result

    def clear_expected_fusion_result(self, scene_id):
        self.cleared_fusion.append(scene_id)

    def expect_suction_result(self, scene_id):
        self.expected_suction.append(scene_id)

    def publish_suction_input(self, scene_id, fused_objects):
        if self.fail_suction_publish:
            raise RuntimeError("Synthetic SuctionInput publish failure")
        self.suction_published.append((scene_id, fused_objects))

    def get_suction_result(self, scene_id):
        return self.suction_result

    def clear_expected_suction_result(self, scene_id):
        self.cleared_suction.append(scene_id)


def header(stamp):
    return NS(
        stamp=NS(sec=stamp.sec, nanosec=stamp.nanosec),
        frame_id=stamp.frame_id,
    )


def make_workflow(*, scoring=True):
    manager = FakeManager()
    settings = {
        "enable_mask_fusion": True,
        "enable_suction_scoring": scoring,
        "fusion_timeout_sec": 20.0,
        "scoring_timeout_sec": 60.0,
        "suction_result_frame": "base_link",
    }
    tx = SceneTransaction(manager, settings, FakeLogger())
    tx.scene = SceneContext()
    tx.scene.sam2_scene_index = 12
    tx.scene.enter_state(CoordinatorState.WAITING_FOR_MASK_FUSION)
    stamp = ObservationStamp(1234, 5, "helios2_optical_frame")
    tx._selected_cloud = NS(stamp=stamp, payload="selected_helios_cloud")
    tx._selected_masks = NS(payload=NS(objects=[
        NS(object_id=7), NS(object_id=8),
    ]))
    tx._fusion_sent_at = monotonic()

    fused = NS(
        scene_index=12,
        header=header(stamp),
        objects=[
            NS(object_id=7, cloud=NS(width=1, height=1, data=b"1")),
            NS(object_id=8, cloud=NS(width=1, height=1, data=b"2")),
        ],
    )
    manager.fusion_result = NS(
        scene_id=tx.scene.scene_id,
        scene_index=12,
        success=True,
        detail="",
        fused_objects=fused,
    )
    return tx, manager, fused


def valid_candidate(*, object_id=7, rank=1):
    return NS(
        object_id=object_id,
        rank=rank,
        position=NS(x=0.1, y=0.2, z=0.3),
        normal=NS(x=0.0, y=0.0, z=-1.0),
        ranking_score=0.85,
        sim_score=float("nan"),
        source_index=-1,
        clearance_valid=False,
    )


def make_suction_result(tx, *, status="OK"):
    result = NS(
        scene_id=tx.scene.scene_id,
        candidates=NS(
            header=header(tx._selected_cloud.stamp),
            scene_index=12,
            status=status,
            error_message="Synthetic scoring failure" if status == "ERROR" else "",
            geometry_is_mask_limited=True,
            processing_time_s=0.15,
            candidates=[valid_candidate()] if status == "OK" else [],
        ),
    )
    result.candidates.header.frame_id = "base_link"
    return result


def prepared_suction():
    tx, manager, fused = make_workflow()
    tx.tick()
    assert tx.scene.state == CoordinatorState.WAITING_FOR_SUCTION_CANDIDATES
    assert len(manager.suction_published) == 1
    assert manager.suction_published[0] == (tx.scene.scene_id, fused)
    return tx, manager


class TestFusionToSuctionWorkflow(unittest.TestCase):
    def test_fusion_only_compatibility(self):
        tx, manager, fused = make_workflow(scoring=False)
        tx.tick()
        self.assertEqual(tx.scene.result_status, ResultStatus.SUCCESS)
        self.assertEqual(tx.scene.state, CoordinatorState.COMPLETE)
        self.assertEqual(manager.suction_published, [])
        self.assertEqual(tx.get_validated_suction_result(), None)
        self.assertEqual(tx.scene.failure_detail,
                         "Mask Fusion completed; fused_objects=2; "
                         "geometric accuracy not yet verified")

    def test_fusion_to_suction_once(self):
        tx, manager = prepared_suction()
        self.assertEqual(tx.scene.result_status, None)
        self.assertEqual(manager.expected_suction, [tx.scene.scene_id])
        self.assertEqual(tx._approved_suction_object_ids, {7, 8})
        for _ in range(5):
            tx.tick()
        self.assertEqual(len(manager.suction_published), 1)

    def test_fusion_validation_cases(self):
        changes = {
            "wrong_result_index": (
                lambda r: setattr(r, "scene_index", 999),
                "FusionResult scene_index does not match the selected SAM2 scene",
            ),
            "wrong_fused_index": (
                lambda r: setattr(r.fused_objects, "scene_index", 999),
                "FusedObjectArray scene_index mismatch",
            ),
            "wrong_header": (
                lambda r: setattr(r.fused_objects.header, "frame_id", "wrong"),
                "FusedObjectArray header does not match the selected Helios2 cloud",
            ),
            "zero_fused_objects": (
                lambda r: setattr(r.fused_objects, "objects", []),
                "Fusion completed but produced zero fused objects",
            ),
            "unexpected_object": (
                lambda r: setattr(r.fused_objects.objects[0], "object_id", 999),
                "Fusion returned duplicate or unexpected object IDs",
            ),
            "duplicate_object_ids": (
                lambda r: setattr(r.fused_objects.objects[1], "object_id", 7),
                "Fusion returned duplicate or unexpected object IDs",
            ),
            "empty_fused_cloud": (
                lambda r: setattr(r.fused_objects.objects[0].cloud, "data", b""),
                "Fused object 7 has an empty cloud",
            ),
        }
        for name, (modify, reason) in changes.items():
            with self.subTest(name=name):
                tx, manager, _ = make_workflow()
                modify(manager.fusion_result)
                tx.tick()
                self.assertEqual(tx.scene.result_status, ResultStatus.FAILED)
                self.assertEqual(tx.scene.failure_code, FailureCode.INVALID_RESULT)
                self.assertEqual(tx.scene.failure_detail, reason)
                self.assertEqual(manager.suction_published, [])

    def test_fusion_reported_failure(self):
        tx, manager, _ = make_workflow()
        manager.fusion_result.success = False
        manager.fusion_result.detail = "Synthetic fusion failure"
        tx.tick()
        self.assertEqual(tx.scene.failure_code, FailureCode.MASK_FUSION_FAILED)
        self.assertEqual(tx.scene.failure_detail,
                         "Mask Fusion reported failure: Synthetic fusion failure")

    def test_fusion_wrong_scene_ignored(self):
        tx, manager, _ = make_workflow()
        manager.fusion_result.scene_id = "stale_scene"
        tx.tick()
        self.assertEqual(tx.scene.state, CoordinatorState.WAITING_FOR_MASK_FUSION)
        self.assertEqual(manager.suction_published, [])

    def test_fusion_timeout(self):
        tx, manager, _ = make_workflow()
        manager.fusion_result = None
        tx._fusion_sent_at = monotonic() - 21.0
        tx.tick()
        self.assertEqual(tx.scene.result_status, ResultStatus.TIMEOUT)
        self.assertEqual(tx.scene.failure_code, FailureCode.MASK_FUSION_TIMEOUT)

    def test_suction_publish_failure(self):
        tx, manager, _ = make_workflow()
        manager.fail_suction_publish = True
        tx.tick()
        self.assertEqual(tx.scene.failure_code, FailureCode.SCORING_FAILED)
        self.assertIn("Synthetic SuctionInput publish failure", tx.scene.failure_detail)
        self.assertIn(tx.scene.scene_id, manager.cleared_suction)

    def test_suction_success_retains_result(self):
        tx, manager = prepared_suction()
        manager.suction_result = make_suction_result(tx)
        tx.tick()
        self.assertEqual(tx.scene.result_status, ResultStatus.SUCCESS)
        self.assertEqual(tx.scene.failure_code, FailureCode.NONE)
        self.assertIs(tx.get_validated_suction_result(), manager.suction_result)
        self.assertEqual(tx.scene.failure_detail,
                         "Sim-Suction completed successfully; candidates=1; "
                         "approved_objects=2; geometry_is_mask_limited=True; "
                         "robot collision safety not verified")
        self.assertIn(tx.scene.scene_id, manager.cleared_fusion)
        self.assertIn(tx.scene.scene_id, manager.cleared_suction)
        self.assertIsNone(tx._approved_suction_object_ids)

    def test_suction_reported_statuses(self):
        for status, expected_status, expected_failure, expected_detail in [
            ("NO_CANDIDATES", ResultStatus.NO_VALID_CANDIDATES,
             FailureCode.NONE,
             "Sim-Suction completed but produced zero suction candidates; "
             "this does not prove the bin is empty"),
            ("ERROR", ResultStatus.FAILED, FailureCode.SCORING_FAILED,
             "Sim-Suction reported ERROR: Synthetic scoring failure"),
        ]:
            with self.subTest(status=status):
                tx, manager = prepared_suction()
                manager.suction_result = make_suction_result(tx, status=status)
                tx.tick()
                self.assertEqual(tx.scene.result_status, expected_status)
                self.assertEqual(tx.scene.failure_code, expected_failure)
                self.assertEqual(tx.scene.failure_detail, expected_detail)
                self.assertIsNone(tx.get_validated_suction_result())

    def test_suction_invalid_result_cases(self):
        changes = {
            "scene_index": (lambda r: setattr(r.candidates, "scene_index", 999),
                            "scene_index mismatch: expected=12; received=999"),
            "frame": (lambda r: setattr(r.candidates.header, "frame_id", "bad"),
                      "SuctionCandidateArray must be in base_link; received 'bad'"),
            "timestamp": (lambda r: setattr(r.candidates.header.stamp, "nanosec", 999),
                          "SuctionCandidateArray capture stamp does not match the approved Helios2 cloud"),
            "processing_time_nan": (lambda r: setattr(r.candidates, "processing_time_s", float("nan")),
                                    "Invalid processing_time_s"),
            "processing_time_negative": (lambda r: setattr(r.candidates, "processing_time_s", -1.0),
                                         "Invalid processing_time_s"),
            "unknown_status": (lambda r: setattr(r.candidates, "status", "UNKNOWN"),
                               "Unsupported scoring status: 'UNKNOWN'"),
            "ok_empty": (lambda r: setattr(r.candidates, "candidates", []),
                         "OK status contains zero candidates"),
            "object_id": (lambda r: setattr(r.candidates.candidates[0], "object_id", 999),
                          "Candidate references unexpected object_id=999"),
            "rank_zero": (lambda r: setattr(r.candidates.candidates[0], "rank", 0),
                          "Invalid candidate rank=0 for object_id=7"),
            "rank_duplicate": (lambda r: r.candidates.candidates.append(copy.deepcopy(r.candidates.candidates[0])),
                               "Duplicate candidate rank=1 for object_id=7"),
            "position_nan": (lambda r: setattr(r.candidates.candidates[0].position, "x", float("nan")),
                             "Non-finite position for object_id=7, rank=1"),
            "normal_inf": (lambda r: setattr(r.candidates.candidates[0].normal, "x", float("inf")),
                           "Non-finite normal for object_id=7, rank=1"),
            "normal_zero": (lambda r: setattr(r.candidates.candidates[0].normal, "z", 0.0),
                            "Zero or near-zero surface normal for object_id=7, rank=1"),
            "ranking_nan": (lambda r: setattr(r.candidates.candidates[0], "ranking_score", float("nan")),
                            "Non-finite ranking_score for object_id=7, rank=1"),
        }
        for name, (modify, reason) in changes.items():
            with self.subTest(name=name):
                tx, manager = prepared_suction()
                manager.suction_result = make_suction_result(tx)
                modify(manager.suction_result)
                tx.tick()
                self.assertEqual(tx.scene.result_status, ResultStatus.FAILED)
                self.assertEqual(tx.scene.failure_code, FailureCode.INVALID_RESULT)
                self.assertEqual(tx.scene.failure_detail, "Invalid SuctionResult: " + reason)
                self.assertIsNone(tx.get_validated_suction_result())

    def test_suction_status_candidate_conflict(self):
        for status in ("ERROR", "NO_CANDIDATES"):
            with self.subTest(status=status):
                tx, manager = prepared_suction()
                manager.suction_result = make_suction_result(tx, status=status)
                manager.suction_result.candidates.candidates.append(valid_candidate())
                tx.tick()
                self.assertEqual(tx.scene.failure_code, FailureCode.INVALID_RESULT)
                self.assertEqual(tx.scene.failure_detail,
                                 f"Invalid SuctionResult: {status} status contains candidates")

    def test_suction_no_approved_ids_is_internal(self):
        tx, manager = prepared_suction()
        tx._approved_suction_object_ids = None
        manager.suction_result = make_suction_result(tx)
        tx.tick()
        self.assertEqual(tx.scene.result_status, ResultStatus.FAILED)
        self.assertEqual(tx.scene.failure_code, FailureCode.INTERNAL_ERROR)
        self.assertEqual(tx.scene.failure_detail,
                         "No approved fused-object IDs registered for suction scoring")

    def test_suction_timeout(self):
        tx, manager = prepared_suction()
        tx._suction_sent_at = monotonic() - 61.0
        tx.tick()
        self.assertEqual(tx.scene.result_status, ResultStatus.TIMEOUT)
        self.assertEqual(tx.scene.failure_code, FailureCode.SCORING_TIMEOUT)

    def test_suction_unrelated_result_ignored(self):
        tx, manager = prepared_suction()
        manager.suction_result = make_suction_result(tx)
        manager.suction_result.scene_id = "old_scene"
        tx.tick()
        self.assertEqual(tx.scene.result_status, None)
        self.assertEqual(tx.scene.state,
                         CoordinatorState.WAITING_FOR_SUCTION_CANDIDATES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
