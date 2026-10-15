"""Unit tests for the typed perception-result handoff data gate."""

from types import SimpleNamespace
from unittest import TestCase

from unseen_sku_coordinator.common.coordinator_states import (
    FailureCode,
    ResultStatus,
)
from unseen_sku_coordinator.perception.scene_context import SceneContext
from unseen_sku_coordinator.perception.scene_result_access import (
    snapshot_scene_result,
)


class FakeTransaction:
    def __init__(self, scene=None, suction_result=None):
        self.scene = scene
        self.suction_result = suction_result

    def get_validated_suction_result(self):
        if self.scene is None or self.scene.result_status != ResultStatus.SUCCESS:
            return None
        return self.suction_result


class TestSceneResultAccess(TestCase):
    def setUp(self):
        self.scene = SceneContext()
        self.transaction = FakeTransaction(scene=self.scene)

    def test_no_scene(self):
        result = snapshot_scene_result(FakeTransaction(), "any-scene")
        self.assertFalse(result.found)
        self.assertIsNone(result.suction_result)

    def test_blank_id_is_rejected(self):
        result = snapshot_scene_result(self.transaction, "")
        self.assertFalse(result.found)

    def test_wrong_id_cannot_read_current_result(self):
        self.scene.finish(ResultStatus.SUCCESS)
        self.transaction.suction_result = SimpleNamespace(
            scene_id=self.scene.scene_id
        )
        result = snapshot_scene_result(self.transaction, "previous-scene")
        self.assertFalse(result.found)
        self.assertIsNone(result.suction_result)

    def test_running_scene_is_found_but_not_finished(self):
        result = snapshot_scene_result(
            self.transaction,
            self.scene.scene_id,
        )
        self.assertTrue(result.found)
        self.assertFalse(result.finished)
        self.assertEqual(result.result_status, "running")
        self.assertIsNone(result.suction_result)

    def test_successful_scoring_exposes_only_matching_result(self):
        self.scene.finish(ResultStatus.SUCCESS)
        approved = SimpleNamespace(scene_id=self.scene.scene_id)
        self.transaction.suction_result = approved
        result = snapshot_scene_result(
            self.transaction,
            self.scene.scene_id,
        )
        self.assertTrue(result.found)
        self.assertTrue(result.finished)
        self.assertEqual(result.result_status, "success")
        self.assertIs(result.suction_result, approved)

    def test_fusion_only_success_has_no_suction_result(self):
        self.scene.finish(ResultStatus.SUCCESS)
        result = snapshot_scene_result(
            self.transaction,
            self.scene.scene_id,
        )
        self.assertTrue(result.found)
        self.assertTrue(result.finished)
        self.assertIsNone(result.suction_result)

    def test_scoring_failure_never_exposes_candidates(self):
        self.scene.finish(
            ResultStatus.FAILED,
            FailureCode.SCORING_FAILED,
            "scoring failed",
        )
        self.transaction.suction_result = SimpleNamespace(
            scene_id=self.scene.scene_id
        )
        result = snapshot_scene_result(
            self.transaction,
            self.scene.scene_id,
        )
        self.assertTrue(result.finished)
        self.assertEqual(result.result_status, "failed")
        self.assertEqual(result.failure_code, "scoring_failed")
        self.assertEqual(result.detail, "scoring failed")
        self.assertIsNone(result.suction_result)

    def test_success_mismatched_scoring_id_is_not_exposed(self):
        self.scene.finish(ResultStatus.SUCCESS)
        self.transaction.suction_result = SimpleNamespace(
            scene_id="stale-scene"
        )
        result = snapshot_scene_result(
            self.transaction,
            self.scene.scene_id,
        )
        self.assertTrue(result.found)
        self.assertIsNone(result.suction_result)
        self.assertIn("mismatch", result.detail)

    def test_replacing_current_scene_invalidates_old_id(self):
        old_scene_id = self.scene.scene_id
        self.scene.finish(ResultStatus.SUCCESS)
        self.transaction.suction_result = SimpleNamespace(
            scene_id=old_scene_id
        )
        self.transaction.scene = SceneContext()
        self.transaction.suction_result = None
        result = snapshot_scene_result(self.transaction, old_scene_id)
        self.assertFalse(result.found)
        self.assertIsNone(result.suction_result)
