"""ROS message-envelope conversion using the existing suction candidate TF math.

The sibling coordinate_transform module owns the point/normal mathematics.
This module preserves scene ID, ranking metadata, and capture timestamps when
building a base-frame SuctionResult for the Perception Coordinator.
"""

from copy import deepcopy

from .coordinate_transform import transform_candidate


def transform_candidate_array(array, transform, target_frame):
    """Copy a candidate array, transforming geometry and its frame exactly once.

    ``transform`` is the geometry_msgs/TransformStamped returned by tf2. It
    may be None *only* if the source and target frames already match or there
    are no candidates. A missing TF never authorizes relabeling geometry.
    """
    source_frame = str(array.header.frame_id)
    if not source_frame:
        raise ValueError("SuctionCandidateArray has no source frame")
    if not target_frame:
        raise ValueError("target frame is empty")
    if array.candidates and source_frame != target_frame and transform is None:
        raise ValueError("Cannot transform candidates without a TF")

    output = deepcopy(array)
    if source_frame != target_frame and array.candidates:
        output.candidates = [
            transform_candidate(candidate, transform)
            for candidate in array.candidates
        ]
    output.header.frame_id = target_frame
    return output


def transform_suction_result(result, transform, target_frame):
    """Transform a transaction-correlated result without losing scene_id.

    All candidate fields except position and normal are preserved. The source
    capture timestamp and scene_index remain unchanged.
    """
    if not isinstance(result.scene_id, str) or not result.scene_id.strip():
        raise ValueError("SuctionResult has no scene_id")
    output = deepcopy(result)
    output.candidates = transform_candidate_array(
        result.candidates, transform, target_frame
    )
    return output
