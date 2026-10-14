"""Software-only M2.3C.4 controlled Mask Fusion integration tests.

Run from workspace after sourcing ROS 2 and its installed overlay, e.g.:
  PYTHONPATH="$PWD/src/unseen_sku_coordinator:$PYTHONPATH" \
    python3 src/unseen_sku_coordinator/test/test_scene_transaction_fusion.py
"""

from concurrent.futures import Future
from time import monotonic
from types import SimpleNamespace as NS

from unseen_sku_coordinator.perception.observation_buffer import (
    ObservationBuffer,
    ReceivedObservation,
)
from unseen_sku_coordinator.perception.scene_context import ObservationStamp
from unseen_sku_coordinator.perception.scene_transaction import SceneTransaction


class FakeManager:
    def __init__(self):
        self.observations = ObservationBuffer(rgbd_history_size=10)
        self.capture = Future()
        self.sam2 = Future()
        self.requested = []

    def get_service_availability(self):
        return {"helios2": True, "sam2": True}

    def get_latest_observations(self):
        return self.observations.get_latest()

    def request_helios_capture(self):
        self.requested.append("helios2")
        return self.capture

    def request_sam2_processing(self):
        self.requested.append("sam2")
        return self.sam2

    def find_realsense_frame_for_sam2(self, mask_stamp, scene_started_monotonic):
        return self.observations.find_realsense_by_stamp(
            mask_stamp, scene_started_monotonic
        )


class FakeLogger:
    def info(self, message):
        print("INFO:", message)


def fake_header(stamp):
    return NS(
        stamp=NS(sec=stamp.sec, nanosec=stamp.nanosec),
        frame_id=stamp.frame_id,
    )


def fake_cloud(stamp):
    """Small, structurally valid PointCloud2-like object."""
    return NS(
        header=fake_header(stamp),
        width=2,
        height=2,
        point_step=12,
        row_step=24,
        data=bytes(48),
        is_dense=False,
        fields=[
            NS(name="x", offset=0, datatype=7, count=1),
            NS(name="y", offset=4, datatype=7, count=1),
            NS(name="z", offset=8, datatype=7, count=1),
        ],
    )


def fake_masks(stamp, scene_index=12, mask_encoding="mono8", object_count=2):
    """Small ObjectMaskArray-like object containing nonempty masks."""
    objects = []
    for i in range(object_count):
        mask = NS(
            height=2,
            width=2,
            encoding=mask_encoding,
            step=2,
            data=bytes([255, 0, 0, 0]),
        )
        objects.append(NS(object_id=i + 1, mask=mask))
    return NS(
        header=fake_header(stamp),
        scene_index=scene_index,
        objects=objects,
    )


def make_workflow():
    manager = FakeManager()
    workflow = SceneTransaction(
        manager,
        {
            "camera_inputs_timeout_sec": 15.0,
            "capture_timeout_sec": 15.0,
            "sam2_timeout_sec": 120.0,
        },
        FakeLogger(),
    )
    return manager, workflow


def advance_to_sam2_publication(manager, workflow, mask_encoding="mono8", object_count=2):
    """Publish cloud and masks before their corresponding service responses."""
    assert workflow.begin()[0]
    assert not workflow.begin()[0], "Overlapping request should return BUSY"

    workflow.tick()
    assert not manager.requested

    rgb_stamp = ObservationStamp(100, 0, "realsense_color_frame")
    manager.observations.record(
        "realsense", ReceivedObservation(rgb_stamp, monotonic())
    )
    workflow.tick()
    assert manager.requested == ["helios2"]

    cloud_stamp = ObservationStamp(100, 20_000_000, "helios2_optical_frame")
    cloud_payload = fake_cloud(cloud_stamp)
    manager.observations.record(
        "helios2",
        ReceivedObservation(cloud_stamp, monotonic(), payload=cloud_payload),
    )
    workflow.tick()
    assert manager.requested == ["helios2"], "Cloud came before response"
    manager.capture.set_result(NS(success=True, message="ok"))
    workflow.tick()
    assert manager.requested == ["helios2", "sam2"]

    masks_payload = fake_masks(
        rgb_stamp, scene_index=12, mask_encoding=mask_encoding,
        object_count=object_count,
    )
    manager.observations.record(
        "sam2",
        ReceivedObservation(
            rgb_stamp,
            monotonic(),
            scene_index=12,
            object_count=object_count,
            payload=masks_payload,
        ),
    )
    workflow.tick()
    assert not workflow.scene.is_finished, "Masks came before response"
    assert workflow.get_selected_pair() is None
    return cloud_payload, masks_payload



# The original M1/M2.1/M2.2 regression file remains unchanged.
# These tests explicitly turn on fusion and exercise the new state.

class FakeFusionManager(FakeManager):
    def __init__(self):
        super().__init__()
        self.expected_scene_id = None
        self.fusion_result = None
        self.sent_pairs = []
        self.fail_publish = False
        self.clear_count = 0

    def expect_fusion_result(self, scene_id):
        assert scene_id
        self.expected_scene_id = scene_id
        self.fusion_result = None

    def get_fusion_result(self, scene_id):
        if scene_id != self.expected_scene_id:
            return None
        if self.fusion_result is None or self.fusion_result.scene_id != scene_id:
            return None
        return self.fusion_result

    def clear_expected_fusion_result(self, scene_id):
        if scene_id == self.expected_scene_id:
            self.expected_scene_id = None
            self.fusion_result = None
            self.clear_count += 1

    def publish_fusion_input(self, scene_id, cloud, masks):
        if self.fail_publish:
            raise RuntimeError("Synthetic FusionInput publish failure")
        self.sent_pairs.append(NS(scene_id=scene_id, cloud=cloud, masks=masks))


def make_fusion_workflow():
    manager = FakeFusionManager()
    workflow = SceneTransaction(
        manager,
        {
            "camera_inputs_timeout_sec": 15.0,
            "capture_timeout_sec": 15.0,
            "sam2_timeout_sec": 120.0,
            "fusion_timeout_sec": 20.0,
            "enable_mask_fusion": True,
        },
        FakeLogger(),
    )
    return manager, workflow


def advance_to_fusion(manager, workflow, mask_encoding="mono8", object_count=2):
    cloud, masks = advance_to_sam2_publication(
        manager, workflow, mask_encoding=mask_encoding, object_count=object_count,
    )
    manager.sam2.set_result(NS(success=True, message="ok"))
    workflow.tick()
    return cloud, masks


def fake_fusion_result(workflow, *, success=True, scene_index=12,
                       fused_count=1, cloud_stamp=None, scene_id=None):
    source_cloud_stamp = workflow._selected_cloud.stamp
    output_cloud_stamp = cloud_stamp or source_cloud_stamp
    objects = [
        NS(object_id=i + 1,
           cloud=fake_cloud(output_cloud_stamp))
        for i in range(fused_count)
    ]
    return NS(
        scene_id=scene_id or workflow.scene.scene_id,
        scene_index=scene_index,
        success=success,
        detail="Synthetic fusion failure" if not success else "Fusion completed",
        fused_objects=NS(
            scene_index=scene_index,
            header=fake_header(output_cloud_stamp),
            objects=objects,
        ),
    )


# A. A valid pair is published exactly once; scene waits for result.
manager, workflow = make_fusion_workflow()
cloud, masks = advance_to_fusion(manager, workflow)
assert workflow.scene.state.value == "waiting_for_mask_fusion", workflow.describe()
assert not workflow.scene.is_finished
assert len(manager.sent_pairs) == 1
assert manager.expected_scene_id == workflow.scene.scene_id
assert manager.sent_pairs[0].scene_id == workflow.scene.scene_id
assert manager.sent_pairs[0].cloud is cloud
assert manager.sent_pairs[0].masks is masks
assert workflow.get_selected_pair() is None, "Active transaction must not expose final pair"
workflow.tick()
assert len(manager.sent_pairs) == 1, "FusionInput must not be re-published on tick"
print("PASS: Validated pair published exactly once; waiting for fusion")

# B. A stale result is ignored; a correlated success completes the scene.
manager.fusion_result = fake_fusion_result(workflow, scene_id="old_scene_id")
workflow.tick()
assert not workflow.scene.is_finished
manager.fusion_result = fake_fusion_result(workflow)
workflow.tick()
assert workflow.scene.result_status.value == "success", workflow.describe()
assert "mask_fusion" in workflow.scene.observations
assert manager.expected_scene_id is None
assert workflow.get_selected_pair() is not None
print("PASS: Unrelated result ignored; correct FusionResult completes scene")

# C. Explicit processing failure: do not wait for timeout.
manager_fail, workflow_fail = make_fusion_workflow()
advance_to_fusion(manager_fail, workflow_fail)
manager_fail.fusion_result = fake_fusion_result(workflow_fail, success=False)
workflow_fail.tick()
assert workflow_fail.scene.result_status.value == "failed", workflow_fail.describe()
assert workflow_fail.scene.failure_code.value == "mask_fusion_failed"
assert workflow_fail._selected_cloud is None
assert manager_fail.expected_scene_id is None
print("PASS: Explicit fusion failure classified and released")

# D. Incorrect SAM2 scene index is an invalid result.
manager_idx, workflow_idx = make_fusion_workflow()
advance_to_fusion(manager_idx, workflow_idx)
manager_idx.fusion_result = fake_fusion_result(workflow_idx, scene_index=999)
workflow_idx.tick()
assert workflow_idx.scene.result_status.value == "failed", workflow_idx.describe()
assert workflow_idx.scene.failure_code.value == "invalid_result"
print("PASS: Wrong fusion scene_index rejected")

# E. Incorrect output cloud header is an invalid result.
manager_stamp, workflow_stamp = make_fusion_workflow()
advance_to_fusion(manager_stamp, workflow_stamp)
manager_stamp.fusion_result = fake_fusion_result(
    workflow_stamp,
    cloud_stamp=ObservationStamp(101, 0, "helios2_optical_frame"),
)
workflow_stamp.tick()
assert workflow_stamp.scene.result_status.value == "failed", workflow_stamp.describe()
assert workflow_stamp.scene.failure_code.value == "invalid_result"
print("PASS: Wrong fusion cloud header rejected")

# F. Successful computation with zero projected objects is NOT bin empty.
manager_empty, workflow_empty = make_fusion_workflow()
advance_to_fusion(manager_empty, workflow_empty)
manager_empty.fusion_result = fake_fusion_result(workflow_empty, fused_count=0)
workflow_empty.tick()
assert workflow_empty.scene.result_status.value == "failed", workflow_empty.describe()
assert workflow_empty.scene.failure_code.value == "invalid_result"
print("PASS: Zero fused objects rejected rather than marked bin empty")

# G. Mask fusion timeout terminates the scene, releasing its data.
manager_timeout, workflow_timeout = make_fusion_workflow()
advance_to_fusion(manager_timeout, workflow_timeout)
workflow_timeout._fusion_sent_at -= 21.0
workflow_timeout.tick()
assert workflow_timeout.scene.result_status.value == "timeout", workflow_timeout.describe()
assert workflow_timeout.scene.failure_code.value == "mask_fusion_timeout"
assert manager_timeout.expected_scene_id is None
assert workflow_timeout._selected_cloud is None
print("PASS: Fusion timeout classified and released")

# H. Publication failure terminates the scene and clears expectation.
manager_pub, workflow_pub = make_fusion_workflow()
manager_pub.fail_publish = True
advance_to_fusion(manager_pub, workflow_pub)
assert workflow_pub.scene.result_status.value == "failed", workflow_pub.describe()
assert workflow_pub.scene.failure_code.value == "mask_fusion_failed"
assert manager_pub.expected_scene_id is None
assert not manager_pub.sent_pairs
print("PASS: FusionInput publish failure handled")

# I. A malformed mask never gets published to Mask Fusion.
manager_bad, workflow_bad = make_fusion_workflow()
advance_to_fusion(manager_bad, workflow_bad, mask_encoding="rgb8")
assert workflow_bad.scene.result_status.value == "failed", workflow_bad.describe()
assert workflow_bad.scene.failure_code.value == "invalid_result"
assert manager_bad.sent_pairs == []
print("PASS: Invalid SAM2 payload blocked before publication")

# J. An empty SAM2 mask array bypasses fusion.
manager_no, workflow_no = make_fusion_workflow()
advance_to_fusion(manager_no, workflow_no, object_count=0)
assert workflow_no.scene.result_status.value == "no_objects_detected", workflow_no.describe()
assert manager_no.sent_pairs == []
print("PASS: SAM2 zero-object result bypasses Mask Fusion")
