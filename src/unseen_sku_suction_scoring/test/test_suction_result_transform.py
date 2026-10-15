"""Regression tests for preserving SuctionResult scene identity across TF.

Run after sourcing ROS 2 Humble and the workspace overlay:
  PYTHONPATH="$PWD/src/unseen_sku_suction_scoring:$PYTHONPATH" \
    python3 -m unittest discover -s src/unseen_sku_suction_scoring/test \
    -p 'test_suction_result_transform.py' -v
"""

import unittest

from geometry_msgs.msg import TransformStamped
from unseen_sku_interfaces.msg import SuctionCandidate, SuctionResult

from candidate_transform.message_transform import (
    transform_candidate_array, transform_suction_result,
)


def source_result(*, status="OK", count=1):
    result = SuctionResult()
    result.scene_id = "transaction-42"
    arr = result.candidates
    arr.scene_index = 13
    arr.header.frame_id = "helios2_optical_frame"
    arr.header.stamp.sec = 1234
    arr.header.stamp.nanosec = 5678
    arr.status = status
    arr.processing_time_s = 0.15
    arr.geometry_is_mask_limited = True
    arr.error_message = "synthetic failure" if status == "ERROR" else ""
    items = []
    for rank in range(1, count + 1):
        candidate = SuctionCandidate()
        candidate.object_id = 3
        candidate.rank = rank
        candidate.ranking_score = 0.9 / rank
        candidate.source_index = rank + 10
        candidate.position.x = 1.0
        candidate.normal.x = 1.0
        items.append(candidate)
    arr.candidates = items
    return result


def base_transform():
    tf = TransformStamped()
    tf.header.frame_id = "base_link"
    tf.child_frame_id = "helios2_optical_frame"
    tf.transform.translation.x = 1.0
    tf.transform.translation.y = 2.0
    tf.transform.translation.z = 3.0
    tf.transform.rotation.z = 0.7071067811865476
    tf.transform.rotation.w = 0.7071067811865476
    return tf


class TestTransformedSuctionResult(unittest.TestCase):
    def test_scene_identity_stamp_rank_and_scores_are_preserved(self):
        original = source_result(count=2)
        transformed = transform_suction_result(original, base_transform(), "base_link")
        self.assertEqual(transformed.scene_id, "transaction-42")
        self.assertEqual(transformed.candidates.scene_index, 13)
        self.assertEqual(transformed.candidates.header.stamp.sec, 1234)
        self.assertEqual(transformed.candidates.header.stamp.nanosec, 5678)
        self.assertEqual(transformed.candidates.header.frame_id, "base_link")
        self.assertEqual(transformed.candidates.status, "OK")
        self.assertEqual(len(transformed.candidates.candidates), 2)
        for i, item in enumerate(transformed.candidates.candidates):
            self.assertEqual(item.rank, i + 1)
            self.assertEqual(item.source_index, i + 11)
            self.assertAlmostEqual(item.ranking_score, 0.9 / (i + 1))
            self.assertEqual(item.object_id, 3)
            self.assertAlmostEqual(item.position.x, 1.0)
            self.assertAlmostEqual(item.position.y, 3.0)
            self.assertAlmostEqual(item.position.z, 3.0)
            self.assertAlmostEqual(item.normal.x, 0.0, places=6)
            self.assertAlmostEqual(item.normal.y, 1.0, places=6)
            self.assertAlmostEqual(item.normal.z, 0.0, places=6)
        self.assertEqual(original.candidates.header.frame_id, "helios2_optical_frame")
        self.assertEqual(original.candidates.candidates[0].position.x, 1.0)
        self.assertAlmostEqual(original.candidates.candidates[0].position.y, 0.0)

    def test_no_candidates_does_not_require_tf(self):
        original = source_result(status="NO_CANDIDATES", count=0)
        transformed = transform_suction_result(original, None, "base_link")
        self.assertEqual(transformed.candidates.header.frame_id, "base_link")
        self.assertEqual(transformed.candidates.status, "NO_CANDIDATES")
        self.assertEqual(transformed.scene_id, original.scene_id)

    def test_scoring_error_does_not_require_tf(self):
        original = source_result(status="ERROR", count=0)
        transformed = transform_suction_result(original, None, "base_link")
        self.assertEqual(transformed.candidates.error_message, "synthetic failure")
        self.assertEqual(transformed.candidates.status, "ERROR")

    def test_missing_tf_for_nonempty_candidates_rejected(self):
        with self.assertRaisesRegex(ValueError, "without a TF"):
            transform_suction_result(source_result(), None, "base_link")

    def test_unrelated_scene_id_cannot_be_silently_forwarded(self):
        original = source_result()
        original.scene_id = ""
        with self.assertRaisesRegex(ValueError, "scene_id"):
            transform_suction_result(original, base_transform(), "base_link")

    def test_array_mode_remains_compatible(self):
        original = source_result()
        transformed = transform_candidate_array(
            original.candidates, base_transform(), "base_link"
        )
        self.assertEqual(transformed.header.frame_id, "base_link")
        self.assertEqual(transformed.scene_index, original.candidates.scene_index)


if __name__ == "__main__":
    unittest.main()
