"""Structural validation of Mask Fusion output for one approved scene.

Pure validation: no ROS publishers, subscriptions, timers, or workflow decisions.
The transaction orchestrator handles Mask Fusion's explicit failure status,
transaction correlation, and timeout before calling this validator.
"""

from dataclasses import dataclass

from .scene_context import ObservationStamp


@dataclass(frozen=True)
class FusionValidation:
    """Validation outcome and original fused observation stamp, if accepted."""

    valid: bool
    reason: str = ""
    fused_stamp: ObservationStamp | None = None


def validate_fusion_output(
    fused_objects,
    *,
    expected_scene_index: int,
    expected_cloud_stamp: ObservationStamp,
    approved_mask_objects,
) -> FusionValidation:
    """Validate fused geometry against the selected SAM2 and Helios2 inputs.

    A fused-object subset is acceptable because some masks may project to
    no points. This is structural validation, not geometric calibration.
    """

    if int(fused_objects.scene_index) != int(expected_scene_index):
        return FusionValidation(False, "FusedObjectArray scene_index mismatch")

    fused_stamp = ObservationStamp(
        sec=int(fused_objects.header.stamp.sec),
        nanosec=int(fused_objects.header.stamp.nanosec),
        frame_id=str(fused_objects.header.frame_id),
    )
    if fused_stamp != expected_cloud_stamp:
        return FusionValidation(
            False,
            "FusedObjectArray header does not match the selected Helios2 cloud",
        )

    if not fused_objects.objects:
        return FusionValidation(
            False, "Fusion completed but produced zero fused objects"
        )

    mask_ids = {int(obj.object_id) for obj in approved_mask_objects}
    fused_ids = [int(obj.object_id) for obj in fused_objects.objects]
    if (
        len(fused_ids) != len(set(fused_ids))
        or not set(fused_ids).issubset(mask_ids)
    ):
        return FusionValidation(
            False, "Fusion returned duplicate or unexpected object IDs"
        )

    for obj in fused_objects.objects:
        cloud = obj.cloud
        if (
            int(cloud.width) * int(cloud.height) == 0
            or len(cloud.data) == 0
        ):
            return FusionValidation(
                False, f"Fused object {obj.object_id} has an empty cloud"
            )

    return FusionValidation(True, fused_stamp=fused_stamp)
