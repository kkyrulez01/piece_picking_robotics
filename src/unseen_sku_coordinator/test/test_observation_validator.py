"""Software tests for M2.2 observation validation (no live cameras)."""

from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

from unseen_sku_coordinator.perception.observation_buffer import ReceivedObservation
from unseen_sku_coordinator.perception.scene_context import ObservationStamp
from unseen_sku_coordinator.perception.observation_validator import validate_observation_pair


def header(stamp):
    return NS(
        stamp=NS(sec=stamp.sec, nanosec=stamp.nanosec),
        frame_id=stamp.frame_id,
    )


def make_pair():
    cloud_stamp = ObservationStamp(100, 20_000_000, "helios2_optical_frame")
    rgb_stamp = ObservationStamp(100, 220_000_000, "camera_color_optical_frame")

    cloud_msg = NS(
        header=header(cloud_stamp),
        width=2, height=2, point_step=12, row_step=24,
        data=bytes(48), is_dense=False,
        fields=[
            NS(name="x", offset=0, datatype=7, count=1),
            NS(name="y", offset=4, datatype=7, count=1),
            NS(name="z", offset=8, datatype=7, count=1),
        ],
    )

    image = NS(
        height=2, width=2, encoding="mono8", step=2,
        data=bytes([0, 255, 0, 0]),
    )
    masks_msg = NS(
        header=header(rgb_stamp),
        scene_index=12,
        objects=[NS(object_id=1, mask=image)],
    )

    cloud = ReceivedObservation(cloud_stamp, 11.0, payload=cloud_msg)
    masks = ReceivedObservation(
        rgb_stamp, 12.0, scene_index=12, object_count=1,
        payload=masks_msg,
    )
    return cloud, masks, rgb_stamp


class ObservationValidatorTests(unittest.TestCase):
    def setUp(self):
        self.cloud, self.masks, self.rgb_stamp = make_pair()

    def check(self):
        return validate_observation_pair(
            cloud=self.cloud,
            masks=self.masks,
            realsense_stamp=self.rgb_stamp,
            sam2_scene_index=12,
            scene_started_monotonic=10.0,
            expected_cloud_frame="helios2_optical_frame",
        )

    def assert_invalid(self, text):
        result = self.check()
        self.assertFalse(result.valid, result)
        self.assertTrue(any(text in error for error in result.errors), result.errors)

    def test_valid_pair(self):
        self.assertTrue(self.check().valid, self.check().errors)

    def test_truncated_cloud(self):
        self.cloud.payload.data = bytes(12)
        self.assert_invalid("data length")

    def test_missing_xyz_field(self):
        self.cloud.payload.fields.pop()
        self.assert_invalid("missing 'z'")

    def test_mask_encoding(self):
        self.masks.payload.objects[0].mask.encoding = "rgb8"
        self.assert_invalid("expected mono8")

    def test_truncated_mask(self):
        self.masks.payload.objects[0].mask.data = bytes([255])
        self.assert_invalid("data length")

    def test_empty_object_mask(self):
        self.masks.payload.objects[0].mask.data = bytes(4)
        self.assert_invalid("no foreground")

    def test_duplicate_object_ids(self):
        self.masks.payload.objects.append(
            deepcopy(self.masks.payload.objects[0])
        )
        self.masks = ReceivedObservation(
            self.masks.stamp, 12.0, scene_index=12, object_count=2,
            payload=self.masks.payload,
        )
        self.assert_invalid("duplicate object_id")

    def test_inconsistent_dimensions(self):
        other = deepcopy(self.masks.payload.objects[0])
        other.object_id = 2
        other.mask.width = 3
        other.mask.step = 3
        other.mask.data = bytes([255, 0, 0, 0, 0, 0])
        self.masks.payload.objects.append(other)
        self.masks = ReceivedObservation(
            self.masks.stamp, 12.0, scene_index=12, object_count=2,
            payload=self.masks.payload,
        )
        self.assert_invalid("inconsistent mask dimensions")

    def test_wrong_sam2_scene_index(self):
        self.masks.payload.scene_index = 99
        self.assert_invalid("payload scene_index")

    def test_wrong_rgb_source_stamp(self):
        self.masks.payload.header.stamp.nanosec += 1
        self.assert_invalid("payload header differs")

    def test_wrong_rgb_frame(self):
        self.rgb_stamp = ObservationStamp(100, 220_000_000, "wrong_frame")
        self.assert_invalid("selected RealSense")

    def test_wrong_cloud_frame(self):
        self.cloud.payload.header.frame_id = "unexpected_frame"
        self.cloud = ReceivedObservation(
            ObservationStamp(100, 20_000_000, "unexpected_frame"),
            11.0,
            payload=self.cloud.payload,
        )
        self.assert_invalid("expected cloud frame")

    def test_old_cloud(self):
        self.cloud = ReceivedObservation(self.cloud.stamp, 9.0, payload=self.cloud.payload)
        self.assert_invalid("before the scene started")

    def test_zero_objects_is_structurally_valid(self):
        self.masks.payload.objects = []
        self.masks = ReceivedObservation(
            self.masks.stamp, 12.0, scene_index=12, object_count=0,
            payload=self.masks.payload,
        )
        self.assertTrue(self.check().valid, self.check().errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
