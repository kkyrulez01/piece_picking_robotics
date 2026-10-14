"""Structural validation of transaction-correlated Sim-Suction results.

The coordinator checks the scene ID before calling this validator. A valid
scoring message is NOT approval to execute a grasp: mask-limited clearance
and scoring confidence are not full robot collision or feasibility checks.
"""

from dataclasses import dataclass
from math import isfinite

from .scene_context import ObservationStamp


@dataclass(frozen=True)
class SuctionValidation:
    """Validation outcome, including whether coordinator state was missing."""

    valid: bool
    reason: str = ""
    internal_error: bool = False


def validate_suction_result(
    result,
    *,
    expected_scene_index: int,
    expected_cloud_stamp: ObservationStamp,
    approved_object_ids,
) -> SuctionValidation:
    """Validate result metadata, status shape, and each ranked candidate.

    Preserves the M3.4C contract:
      * OK requires candidates with finite positions, nonzero normals,
        positive per-object-unique ranks, and finite final ranking scores.
      * ERROR and NO_CANDIDATES must contain no candidates.
      * Optional quality metrics may be NaN; source_index may be -1.
      * clearance_valid is informative, not a safety authorization.
    """

    output = result.candidates

    if int(output.scene_index) != int(expected_scene_index):
        return SuctionValidation(
            False,
            f"scene_index mismatch: expected={int(expected_scene_index)}; "
            f"received={output.scene_index}",
        )

    received_header = (
        int(output.header.stamp.sec),
        int(output.header.stamp.nanosec),
        str(output.header.frame_id),
    )
    expected_header = (
        int(expected_cloud_stamp.sec),
        int(expected_cloud_stamp.nanosec),
        str(expected_cloud_stamp.frame_id),
    )
    if received_header != expected_header:
        return SuctionValidation(
            False,
            "SuctionCandidateArray header does not match "
            "the approved Helios2 cloud",
        )

    processing_time = float(output.processing_time_s)
    if not isfinite(processing_time) or processing_time < 0.0:
        return SuctionValidation(False, "Invalid processing_time_s")

    status = str(output.status)
    candidates = output.candidates

    if status == "ERROR":
        if candidates:
            return SuctionValidation(False, "ERROR status contains candidates")
        return SuctionValidation(True)

    if status == "NO_CANDIDATES":
        if candidates:
            return SuctionValidation(
                False, "NO_CANDIDATES status contains candidates"
            )
        return SuctionValidation(True)

    if status != "OK":
        return SuctionValidation(
            False, f"Unsupported scoring status: {status!r}"
        )

    if not candidates:
        return SuctionValidation(False, "OK status contains zero candidates")

    if not approved_object_ids:
        return SuctionValidation(
            False,
            "No approved fused-object IDs registered for suction scoring",
            internal_error=True,
        )

    seen_ranks = set()
    for candidate in candidates:
        object_id = int(candidate.object_id)
        rank = int(candidate.rank)

        if object_id not in approved_object_ids:
            return SuctionValidation(
                False, f"Candidate references unexpected object_id={object_id}"
            )
        if rank < 1:
            return SuctionValidation(
                False,
                f"Invalid candidate rank={rank} for object_id={object_id}"
            )

        rank_key = (object_id, rank)
        if rank_key in seen_ranks:
            return SuctionValidation(
                False,
                f"Duplicate candidate rank={rank} for object_id={object_id}"
            )
        seen_ranks.add(rank_key)

        position = (
            float(candidate.position.x),
            float(candidate.position.y),
            float(candidate.position.z),
        )
        if not all(isfinite(value) for value in position):
            return SuctionValidation(
                False,
                f"Non-finite position for object_id={object_id}, rank={rank}"
            )

        normal = (
            float(candidate.normal.x),
            float(candidate.normal.y),
            float(candidate.normal.z),
        )
        if not all(isfinite(value) for value in normal):
            return SuctionValidation(
                False,
                f"Non-finite normal for object_id={object_id}, rank={rank}"
            )
        if sum(value * value for value in normal) <= 1e-12:
            return SuctionValidation(
                False,
                f"Zero or near-zero surface normal for "
                f"object_id={object_id}, rank={rank}",
            )

        if not isfinite(float(candidate.ranking_score)):
            return SuctionValidation(
                False,
                f"Non-finite ranking_score for "
                f"object_id={object_id}, rank={rank}",
            )

    return SuctionValidation(True)
