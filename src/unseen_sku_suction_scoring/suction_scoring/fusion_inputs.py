"""Pure-NumPy adapter for the working FusedObjectArray payload.

The ROS wrapper creates FusionObjectData records. No rclpy dependency here.
IMPORTANT: This is the union of mask-associated Helios points, not the
unmasked full scene cloud. A masked union alone is not sufficient to certify
robot/tool clearance against unseen bin walls or unsegmented obstacles.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np

@dataclass(frozen=True)
class FusionObjectData:
    object_id: int
    mask: np.ndarray # (H, W), bool
    xyz_m: np.ndarray # (N, 3), Helios optical frame, metres
    uv: np.ndarray # (N, 2), pixel (u, v) in RealSense RGB
    source_indices: np.ndarray # (N,), original Helios point indices

@dataclass(frozen=True)
class SceneInput:
    xyz_m: np.ndarray
    uv: np.ndarray
    original_source_indices: np.ndarray
    object_ids: tuple
    masks: np.ndarray

def assemble_fused_scene(objects: Sequence[FusionObjectData]) -> SceneInput:
    """Concatenate unique Helios points, preserving source index and mask IDs.

    Duplicated source indices can arise from overlapping SAM2 masks. Reuse a
    single 3-D point in the scene-level Sim-Suction inference in that case.
    """
    if not objects:
        raise ValueError("No nonempty fused objects.")
    object_ids = tuple(int(obj.object_id) for obj in objects)
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Fused objects have duplicate object_id values")

    masks = [np.asarray(obj.mask, dtype=bool) for obj in objects]
    shape = masks[0].shape
    if len(shape) != 2 or any(mask.shape != shape for mask in masks):
        raise ValueError("All masks must have the same (H, W) dimensions")

    xyz_chunks = []
    uv_chunks = []
    source_chunks = []
    for obj, mask in zip(objects, masks):
        xyz = np.asarray(obj.xyz_m, dtype=np.float32)
        uv = np.asarray(obj.uv, dtype=np.int32)
        src = np.asarray(obj.source_indices, dtype=np.int64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Object {obj.object_id}: xyz must be (N, 3)")
        n = len(xyz)
        if uv.shape != (n, 2) or src.shape != (n,):
            raise ValueError(f"Object {obj.object_id}: XYZ/UV/source lengths differ")
        if n == 0:
            continue
        if np.any(src < 0) or not np.isfinite(xyz).all():
            raise ValueError(f"Object {obj.object_id}: invalid XYZ or source index")
        if np.any(uv[:, 0] < 0) or np.any(uv[:, 0] >= shape[1]):
            raise ValueError(f"Object {obj.object_id}: UV out of horizontal bounds")
        if np.any(uv[:, 1] < 0) or np.any(uv[:, 1] >= shape[0]):
            raise ValueError(f"Object {obj.object_id}: UV out of vertical bounds")
        # Mask fusion should never associate a point with an unrelated pixel.
        if not np.all(mask[uv[:, 1], uv[:, 0]]):
            raise ValueError(f"Object {obj.object_id}: associated UV outside mask")
        xyz_chunks.append(xyz)
        uv_chunks.append(uv)
        source_chunks.append(src)

    if not xyz_chunks:
        raise ValueError("No fused points remain in any object")

    xyz = np.concatenate(xyz_chunks)
    uv = np.concatenate(uv_chunks)
    sources = np.concatenate(source_chunks)
    sort_order = np.argsort(sources, kind="stable")
    xyz = xyz[sort_order]
    uv = uv[sort_order]
    sources = sources[sort_order]

    duplicate_positions = np.flatnonzero(np.diff(sources) == 0) + 1
    for i in duplicate_positions:
        if not np.allclose(xyz[i], xyz[i - 1], rtol=0, atol=1e-5):
            raise ValueError(f"Conflicting XYZ for Helios source {sources[i]}")
        if not np.array_equal(uv[i], uv[i - 1]):
            raise ValueError(f"Conflicting UV for Helios source {sources[i]}")

    unique_mask = np.ones(len(sources), dtype=bool)
    unique_mask[1:] = np.diff(sources) != 0
    return SceneInput(
        xyz_m=xyz[unique_mask],
        uv=uv[unique_mask],
        original_source_indices=sources[unique_mask],
        object_ids=object_ids,
        masks=np.stack(masks, axis=0),
    )