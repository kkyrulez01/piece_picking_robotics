"""Software-only M1 / M2.1B / M2.2B transaction regression tests.

Run from workspace after sourcing ROS 2 and its installed overlay, e.g.:
  PYTHONPATH="$PWD/src/unseen_sku_coordinator:$PYTHONPATH" \
    python3 src/unseen_sku_coordinator/test/test_scene_transaction.py
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


# Existing successful transaction and M2.1B message retention.
manager, workflow = make_workflow()
cloud_payload, masks_payload = advance_to_sam2_publication(manager, workflow)
manager.sam2.set_result(NS(success=True, message="ok"))
workflow.tick()
assert workflow.scene.result_status.value == "success", workflow.describe()
assert workflow.scene.sam2_scene_index == 12
assert set(workflow.scene.observations) == {"helios2", "realsense", "sam2"}
print("PASS: M1 transaction, early publications, frame matching, BUSY guard")

pair = workflow.get_selected_pair()
assert pair is not None
assert pair[0].payload is cloud_payload
assert pair[1].payload is masks_payload
print("PASS: M2.1B selected Helios2 and SAM2 payloads")

manager.observations.record(
    "helios2",
    ReceivedObservation(
        ObservationStamp(101, 0, "helios2_optical_frame"),
        monotonic(),
        payload=object(),
    ),
)
manager.observations.record(
    "sam2",
    ReceivedObservation(
        ObservationStamp(101, 0, "realsense_color_frame"),
        monotonic(), scene_index=13, object_count=1, payload=object(),
    ),
)
assert workflow.get_selected_pair()[0].payload is cloud_payload
assert workflow.get_selected_pair()[1].payload is masks_payload
print("PASS: New observations do not replace the selected pair")

workflow.release_selected_pair()
assert workflow.get_selected_pair() is None
print("PASS: Selected pair released successfully")

# New M2.2B integration failure path.
manager_invalid, workflow_invalid = make_workflow()
advance_to_sam2_publication(
    manager_invalid, workflow_invalid, mask_encoding="rgb8"
)
manager_invalid.sam2.set_result(NS(success=True, message="ok"))
workflow_invalid.tick()
assert workflow_invalid.scene.result_status.value == "failed", workflow_invalid.describe()
assert workflow_invalid.scene.failure_code.value == "invalid_result", workflow_invalid.describe()
assert "expected mono8" in workflow_invalid.scene.failure_detail
assert workflow_invalid.get_selected_pair() is None
assert workflow_invalid._selected_cloud is None
assert workflow_invalid._selected_masks is None
print("PASS: M2.2B invalid mask rejected and payloads released")

# Empty mask array is a valid result, but not a successful picking scene.
manager_empty, workflow_empty = make_workflow()
advance_to_sam2_publication(manager_empty, workflow_empty, object_count=0)
manager_empty.sam2.set_result(NS(success=True, message="ok"))
workflow_empty.tick()
assert workflow_empty.scene.result_status.value == "no_objects_detected", workflow_empty.describe()
assert workflow_empty.scene.failure_code.value == "none"
assert workflow_empty.get_selected_pair() is None
print("PASS: M2.2B zero objects classified separately")
