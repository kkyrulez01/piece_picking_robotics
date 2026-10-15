"""Fail-closed read-only snapshot of a perception scene transaction.

This module intentionally has no ROS dependencies. Only the current retained
scene can be retrieved; callers must request its exact scene_id.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneResultSnapshot:
    """Metadata and optional coordinator-validated suction result."""

    found: bool = False
    finished: bool = False
    result_status: str = ""
    failure_code: str = ""
    detail: str = ""
    suction_result: object = None


def snapshot_scene_result(transaction, requested_scene_id: str):
    """Return only the retained scene requested by ID (never a newer scene)."""

    scene = transaction.scene
    if (
        not requested_scene_id
        or not requested_scene_id.strip()
        or scene is None
        or requested_scene_id != scene.scene_id
    ):
        return SceneResultSnapshot(
            detail="No retained perception scene matches the requested scene_id"
        )

    result_status = (
        scene.result_status.value if scene.result_status is not None
        else "running"
    )

    suction_result = None
    detail = scene.failure_detail
    if scene.is_finished:
        validated = transaction.get_validated_suction_result()
        if validated is not None:
            if validated.scene_id == requested_scene_id:
                suction_result = validated
            else:
                # Defensive check: never expose a result from another scene.
                detail = "Validated suction result scene_id mismatch"

    return SceneResultSnapshot(
        found=True,
        finished=scene.is_finished,
        result_status=result_status,
        failure_code=scene.failure_code.value,
        detail=detail,
        suction_result=suction_result,
    )
