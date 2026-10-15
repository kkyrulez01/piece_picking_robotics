from pathlib import Path
from dataclasses import dataclass, replace
import cv2
import numpy as np

from .suction_candidate import SuctionCandidate

def load_sam2_masks(mask_directory, scene_index):
    """
    Load all individual SAM2 masks belonging to one scene.

    Example filenames:
        sam2_mask_0006_01.png
        sam2_mask_0006_02.png
        sam2_mask_0006_03.png

    Returns:
        object_masks:
            Boolean array with shape (K, H, W).
        mask_paths:
            Paths corresponding to each object mask.
    """
    mask_directory = Path(mask_directory).expanduser()

    pattern = f"sam2_mask_{scene_index:04d}_*.png"
    mask_paths = sorted(mask_directory.glob(pattern))

    if not mask_paths:
        raise FileNotFoundError(
           f"No SAM2 masks found using: {mask_directory / pattern}"
        )

    masks = []
    for mask_path in mask_paths:
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask_image is None:
            raise RuntimeError(f"Failed to load SAM2 mask: {mask_path}")

        # Convert black/white image into a Boolean mask.
        mask = mask_image > 127

        if not np.any(mask):
            raise ValueError(
                f"SAM2 mask contains no foreground pixels: {mask_path}"
            )

        masks.append(mask)

    object_masks = np.stack(masks, axis=0)

    return object_masks, mask_paths

def rank_candidates_per_mask(
    valid_xyz_m,
    valid_normals,
    valid_scores,
    valid_source_indices,
    valid_uv,
    object_masks,
    minimum_score=0.0,
    maximum_candidates_per_object=50,
):
    """
    Associate valid Sim-Suction samples with SAM2 masks and
    rank the samples belonging to each object.

    Args:
        object_masks:
            Boolean array with shape (K, H, W), where 
            K: number of SAM2 object masks, 
            H: image height in pixels,
            W: image width in pixels

    Returns:
        Dictionary:
            object_id -> ranked list of SuctionCandidate
    """
    valid_xyz_m = np.asarray(valid_xyz_m, dtype=np.float32)
    valid_normals = np.asarray(valid_normals, dtype=np.float32)
    valid_scores = np.asarray(valid_scores, dtype=np.float32)
    valid_source_indices = np.asarray(valid_source_indices, dtype=np.int64)
    valid_uv = np.asarray(valid_uv, dtype=np.int64)
    object_masks = np.asarray(object_masks, dtype=bool)

    point_count = valid_scores.shape[0]

    expected_shapes = {
        "valid_xyz_m": (point_count, 3),
        "valid_normals": (point_count, 3),
        "valid_source_indices": (point_count,),
        "valid_uv": (point_count, 2),
    }

    arrays = {
        "valid_xyz_m": valid_xyz_m,
        "valid_normals": valid_normals,
        "valid_source_indices": valid_source_indices,
        "valid_uv": valid_uv,
    }

    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}. "
                f"Received {arrays[name].shape}."
            )

    if object_masks.ndim != 3:
        raise ValueError("object_masks must have shape (K, H, W).")

    image_height = object_masks.shape[1]
    image_width = object_masks.shape[2]
    u = valid_uv[:, 0]
    v = valid_uv[:, 1]

    inside_image = (
        (u >= 0) & (u < image_width)
        & (v >= 0) & (v < image_height)
    )

    candidates_by_object = {}
    for mask_index, object_mask in enumerate(object_masks):
        object_id = mask_index + 1

        inside_object = np.zeros(point_count, dtype=bool)
        usable_indices = np.flatnonzero(inside_image)
        inside_object[usable_indices] = object_mask[v[usable_indices], u[usable_indices]]

        # Points should be in the object and have > minimum score
        eligible_indices = np.flatnonzero(
            inside_object & (valid_scores >= float(minimum_score))
        )

        # Sort eligible points
        sorted_indices = eligible_indices[
            np.argsort(valid_scores[eligible_indices])[::-1]
        ]
        sorted_indices = sorted_indices[:maximum_candidates_per_object]

        # Candidate points for 1 object
        object_candidates = []

        for rank, valid_index in enumerate(sorted_indices, start=1):
            object_candidates.append(
                SuctionCandidate(
                    object_id=object_id,
                    rank=rank,
                    sim_score=float(valid_scores[valid_index]),
                    ranking_score=float(valid_scores[valid_index]),
                    xyz_m=valid_xyz_m[valid_index].copy(),
                    normal=valid_normals[valid_index].copy(),
                    uv=valid_uv[valid_index].copy(),
                    source_index=int(valid_source_indices[valid_index]),
                    valid_index=int(valid_index),
                )
            )

        candidates_by_object[object_id] = object_candidates

    return candidates_by_object

def spatial_nms(candidates, minimum_distance_m=0.02, maximum_candidates=3):
    """
    Select spatially separated candidates for ONE object.

    Uses real XYZ in metres, not normalized XYZ or image UV.
    Returns up to maximum_candidates, ordered by score.
    """
    if (not np.isfinite(minimum_distance_m) or minimum_distance_m <= 0):
        raise ValueError("minimum_distance_m must be finite and positive.")

    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive.")

    candidates = list(candidates)
    if not candidates:
        return []

    xyz = np.asarray([candidate.xyz_m for candidate in candidates], dtype=np.float64)
    scores = np.asarray([candidate.ranking_score for candidate in candidates], dtype=np.float64)

    if xyz.shape != (len(candidates), 3):
        raise ValueError("Each candidate must have XYZ with shape (3,).")

    if not np.isfinite(xyz).all() or not np.isfinite(scores).all():
        raise ValueError("Candidate coordinates and scores must be finite.")

    # 1. Sort by descending order
    order = np.argsort(-scores, kind="stable")
    selected_indices = []

    # 2. Keep highest scoring candidate and reject candidates that are too close
    for index in order:
        if selected_indices:
            distances_squared = np.sum((xyz[selected_indices] - xyz[index]) ** 2, axis=1)

            if np.any(distances_squared < minimum_distance_m ** 2):
                continue

        selected_indices.append(int(index))

        # 3. Continue until n amount of candidates retained
        if len(selected_indices) >= maximum_candidates:
            break

    return [
        replace(candidates[index], rank=rank)
        for rank, index in enumerate(selected_indices, start=1)
    ]