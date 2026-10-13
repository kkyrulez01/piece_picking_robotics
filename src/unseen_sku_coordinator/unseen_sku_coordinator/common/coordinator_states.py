"""Coordinator workflow states and transaction result categories."""

from enum import Enum


class CoordinatorState(str, Enum):
    IDLE = "idle"

    # Camera acquisition
    REQUESTING_CAPTURE = "requesting_capture"
    WAITING_FOR_CAMERA_INPUTS = "waiting_for_camera_inputs"

    # Segmentation
    REQUESTING_SAM2 = "requesting_sam2"
    WAITING_FOR_SAM2_MASKS = "waiting_for_sam2_masks"

    # Point-cloud fusion
    WAITING_FOR_MASK_FUSION = "waiting_for_mask_fusion"

    # Suction scoring
    WAITING_FOR_SUCTION_CANDIDATES = "waiting_for_suction_candidates"

    # Result processing
    VALIDATING_RESULT = "validating_result"

    # Terminal states
    COMPLETE = "complete"
    NO_OBJECTS_DETECTED = "no_objects_detected"
    NO_VALID_CANDIDATES = "no_valid_candidates"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    NO_OBJECTS_DETECTED = "no_objects_detected"
    NO_VALID_CANDIDATES = "no_valid_candidates"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class FailureCode(str, Enum):
    NONE = "none"
    BUSY = "busy"

    # Camera acquisition
    CAMERA_UNAVAILABLE = "camera_unavailable"
    CAPTURE_TIMEOUT = "capture_timeout"
    RGBD_TIMEOUT = "rgbd_timeout"
    HELIOS_CLOUD_TIMEOUT = "helios_cloud_timeout"

    # SAM2
    SAM2_UNAVAILABLE = "sam2_unavailable"
    SAM2_FAILED = "sam2_failed"
    SAM2_TIMEOUT = "sam2_timeout"

    # Mask Fusion
    MASK_FUSION_FAILED = "mask_fusion_failed"
    MASK_FUSION_TIMEOUT = "mask_fusion_timeout"

    # Sim-Suction
    SCORING_UNAVAILABLE = "scoring_unavailable"
    SCORING_FAILED = "scoring_failed"
    SCORING_TIMEOUT = "scoring_timeout"

    # General
    STALE_RESULT = "stale_result"
    INVALID_RESULT = "invalid_result"
    INTERNAL_ERROR = "internal_error"


TERMINAL_STATES = frozenset({
    CoordinatorState.COMPLETE,
    CoordinatorState.NO_OBJECTS_DETECTED,
    CoordinatorState.NO_VALID_CANDIDATES,
    CoordinatorState.FAILED,
    CoordinatorState.TIMEOUT,
    CoordinatorState.CANCELLED,
})
