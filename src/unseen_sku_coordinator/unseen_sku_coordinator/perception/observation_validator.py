"""Validate selected Helios2 and SAM2 observations before mask fusion.

This module performs structural and transaction-metadata checks. It does not
prove camera synchronization, calibration quality, or segmentation accuracy.
"""

from dataclasses import dataclass

from .observation_buffer import ReceivedObservation
from .scene_context import ObservationStamp


@dataclass(frozen=True)
class ValidationResult:
    """All validation errors; empty errors means the checks passed."""

    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


def _header_stamp(message, source: str, errors: list[str]) -> ObservationStamp | None:
    """Read the header using the same representation as ObservationBuffer."""

    try:
        stamp = ObservationStamp(
            sec=int(message.header.stamp.sec),
            nanosec=int(message.header.stamp.nanosec),
            frame_id=str(message.header.frame_id),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"{source}: invalid or missing message header ({exc})")
        return None

    if not stamp.frame_id.strip():
        errors.append(f"{source}: empty header.frame_id")
    return stamp


def _validate_cloud(message, errors: list[str]) -> None:
    """Check PointCloud2 layout without decoding the entire point cloud."""

    width = int(message.width)
    height = int(message.height)
    point_step = int(message.point_step)
    row_step = int(message.row_step)

    if width <= 0 or height <= 0:
        errors.append("helios2: cloud width/height must be positive")
    if point_step <= 0:
        errors.append("helios2: point_step must be positive")
    if width > 0 and point_step > 0 and row_step < width * point_step:
        errors.append("helios2: row_step is too small for width * point_step")
    if row_step <= 0 or height <= 0 or len(message.data) != row_step * height:
        errors.append("helios2: data length does not equal row_step * height")

    # PointField type identifiers from sensor_msgs/msg/PointField.
    float_bytes = {7: 4, 8: 8}  # FLOAT32 and FLOAT64
    xyz = {}
    for field in message.fields:
        if field.name not in ("x", "y", "z"):
            continue
        if field.name in xyz:
            errors.append(f"helios2: duplicate '{field.name}' field")
        xyz[field.name] = field

    for name in ("x", "y", "z"):
        field = xyz.get(name)
        if field is None:
            errors.append(f"helios2: missing '{name}' field")
            continue

        byte_width = float_bytes.get(int(field.datatype))
        if byte_width is None or int(field.count) != 1:
            errors.append(f"helios2: '{name}' must be scalar FLOAT32/FLOAT64")
            continue
        if int(field.offset) < 0 or int(field.offset) + byte_width > point_step:
            errors.append(f"helios2: '{name}' offset exceeds point_step")

    # is_dense=False is normal for depth cameras with invalid measurements.


def _validate_masks(message, errors: list[str]) -> None:
    """Check the format and contents of each reported object mask."""

    ids = set()
    expected_dimensions = None

    for index, obj in enumerate(message.objects):
        label = f"sam2: objects[{index}]"

        object_id = int(obj.object_id)
        if object_id in ids:
            errors.append(f"{label}: duplicate object_id={object_id}")
        ids.add(object_id)

        image = obj.mask
        width = int(image.width)
        height = int(image.height)
        step = int(image.step)

        if width <= 0 or height <= 0:
            errors.append(f"{label}: mask width/height must be positive")
            continue

        dimensions = (width, height)
        if expected_dimensions is None:
            expected_dimensions = dimensions
        elif dimensions != expected_dimensions:
            errors.append(f"{label}: inconsistent mask dimensions {dimensions}")

        # SAM2 publishes sensor_msgs/Image; fusion requests mono8 via CvBridge.
        if image.encoding != "mono8":
            errors.append(f"{label}: expected mono8, got {image.encoding!r}")
            continue

        if step < width:
            errors.append(f"{label}: step smaller than width for mono8")
            continue

        if len(image.data) != step * height:
            errors.append(f"{label}: data length does not equal step * height")
            continue

        # Ignore row padding when checking for foreground pixels.
        has_foreground = any(
            any(image.data[row * step:row * step + width])
            for row in range(height)
        )
        if not has_foreground:
            errors.append(f"{label}: mask has no foreground pixels")

    # An empty objects[] is structurally valid; M1 classifies it separately.


def validate_observation_pair(
    cloud: ReceivedObservation,
    masks: ReceivedObservation,
    realsense_stamp: ObservationStamp,
    sam2_scene_index: int,
    scene_started_monotonic: float | None = None,
    expected_cloud_frame: str | None = None,
) -> ValidationResult:
    """Validate the selected pair without mutating its messages.

    The caller supplies the RealSense stamp and accepted SAM2 scene index
    recorded in the active SceneContext. No physical camera sync is assumed.
    """

    errors: list[str] = []

    if cloud.payload is None:
        errors.append("helios2: missing PointCloud2 payload")
    if masks.payload is None:
        errors.append("sam2: missing ObjectMaskArray payload")
    if errors:
        return ValidationResult(tuple(errors))

    cloud_stamp = _header_stamp(cloud.payload, "helios2", errors)
    masks_stamp = _header_stamp(masks.payload, "sam2", errors)

    if cloud_stamp is not None and cloud_stamp != cloud.stamp:
        errors.append("helios2: payload header differs from selected metadata")
    if masks_stamp is not None and masks_stamp != masks.stamp:
        errors.append("sam2: payload header differs from selected metadata")
    if masks.stamp != realsense_stamp:
        errors.append("sam2: source header does not match selected RealSense stamp/frame")

    if masks.scene_index != sam2_scene_index:
        errors.append("sam2: selected metadata scene_index does not match transaction")
    if int(masks.payload.scene_index) != sam2_scene_index:
        errors.append("sam2: payload scene_index does not match transaction")
    if masks.object_count is not None and masks.object_count != len(masks.payload.objects):
        errors.append("sam2: object_count metadata differs from payload")

    if scene_started_monotonic is not None:
        if cloud.received_monotonic < scene_started_monotonic:
            errors.append("helios2: cloud was received before the scene started")
        if masks.received_monotonic < scene_started_monotonic:
            errors.append("sam2: masks were received before the scene started")
    if masks.received_monotonic < cloud.received_monotonic:
        errors.append("sam2: masks were received before the selected cloud")

    if (
        expected_cloud_frame is not None
        and cloud.stamp.frame_id != expected_cloud_frame
    ):
        errors.append(
            "helios2: frame_id does not match expected cloud frame"
        )

    _validate_cloud(cloud.payload, errors)
    _validate_masks(masks.payload, errors)

    return ValidationResult(tuple(errors))
