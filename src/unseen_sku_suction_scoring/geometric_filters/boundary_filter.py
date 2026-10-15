#!/usr/bin/env python3

import cv2
import numpy as np
from dataclasses import replace

def build_mask_boundary_distance_maps(object_masks):
    """
    Compute distance from every foreground pixel to the nearest
    boundary of its corresponding SAM2 mask.

    Args:
        object_masks:
            Boolean array with shape (K, H, W).

    Returns:
        distance_maps:
            Float32 array with shape (K, H, W).

            Inside a mask:
                value = distance to nearest mask boundary in pixels.

            Outside a mask:
                value = 0.
    """
    object_masks = np.asarray(object_masks, dtype=bool)

    if object_masks.ndim != 3:
        raise ValueError("object_masks must have shape (K, H, W).")

    object_count, image_height, image_width = object_masks.shape
    distance_maps = np.empty((object_count, image_height, image_width), dtype=np.float32)

    for mask_index, object_mask in enumerate(object_masks):
        # Add background padding so masks touching the image edge
        # still have a well-defined boundary.
        padded_mask = np.pad(
            object_mask.astype(np.uint8),
            pad_width=1,
            mode="constant",
            constant_values=0,
        )

        padded_distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)

        distance_maps[mask_index] = padded_distance[
            1:image_height + 1,
            1: image_width + 1,
        ]

    return distance_maps

def normalize_boundary_distance_map(distance_map, object_mask):
    """
    Normalize one object's boundary-distance map into the range [0, 1].

    0.0:
        Pixels closest to the object's boundary.

    1.0:
        Pixels with the greatest available interior clearance
        for that object.

    This makes boundary clearance relative to the size/shape
    of each individual object.
    """
    distance_map = np.asarray(distance_map, dtype=np.float32)
    object_mask = np.asarray(object_mask, dtype=bool)

    object_distances = distance_map[object_mask]

    minimum_distance = float(np.min(object_distances))
    maximum_distance = float(np.max(object_distances))
    distance_range = maximum_distance - minimum_distance

    normalized_map = np.zeros_like(distance_map, dtype=np.float32)
    if distance_range > 1e-8:
        normalized_map[object_mask] = (
            (distance_map[object_mask] - minimum_distance) / distance_range
        )

    else:
        # Entire object has effectively the same boundary distance.
        # Every point therefore has the maximum relative clearance
        # available for this particular object.
        normalized_map[object_mask] = 1.0

    return normalized_map

def build_normalized_boundary_distance_maps(object_masks, distance_maps=None):
    """
    Build normalized boundary-clearance maps for all SAM2 masks.

    Returns:
        normalized_maps:
            Float32 array with shape (K, H, W).

            0.0 = closest to boundary
            1.0 = maximum relative interior clearance
    """
    object_masks = np.asarray(object_masks, dtype=bool)

    if distance_maps is None:
        distance_maps = build_mask_boundary_distance_maps(object_masks)
    else:
        distance_maps = np.asarray(distance_maps, dtype=np.float32)

        if distance_maps.shape != object_masks.shape:
            raise ValueError("distance_maps must have the same shape as object_masks.")

    normalized_maps = np.zeros_like(distance_maps, dtype=np.float32)

    for mask_index in range(object_masks.shape[0]):
        normalized_maps[mask_index] = (
            normalize_boundary_distance_map(
                distance_map=distance_maps[mask_index],
                object_mask=object_masks[mask_index]
            )
        )

    return normalized_maps

def calculate_boundary_quality(
    normalized_boundary_distance,
    exponent=2.0
):
    """
    Convert normalized boundary distance into a quality score.

    Returns:
        boundary_quality in [0, 1]

        0.0 = candidate is near the object boundary
        1.0 = candidate has maximum relative interior clearance
    """
    # Clamp between 0.0 to 1.0
    normalized_boundary_distance = np.clip(normalized_boundary_distance, 0.0, 1.0)

    # Returns boundary quality
    boundary_quality = normalized_boundary_distance ** exponent

    return float(boundary_quality)

def calculate_candidate_boundary_quality(candidates_by_object, object_masks, exponent=2.0):
    """
    Calculate normalized boundary quality for each suction candidate.

    No candidates are rejected and ranking_score is not modified.

    Args:
        candidates_by_object:
            Dictionary:
                object_id -> list[SuctionCandidate]

        object_masks:
            Boolean array with shape (K, H, W).

        exponent:
            Controls how strongly candidates near the object boundary
            receive a lower boundary quality.

    Returns:
        candidates_with_boundary_quality:
            Dictionary:
                object_id -> list[SuctionCandidate]

            Each candidate contains:
                boundary_distance_px
                boundary_distance_normalized
                boundary_quality

        distance_maps:
            Raw boundary-distance maps in pixels.

        normalized_distance_maps:
            Normalized boundary-distance maps in [0, 1].
    """
    object_masks = np.asarray(object_masks, dtype=bool)

    object_count, image_height, image_width = object_masks.shape

    distance_maps = build_mask_boundary_distance_maps(object_masks)
    normalized_distance_maps = build_normalized_boundary_distance_maps(
        object_masks=object_masks,
        distance_maps=distance_maps
    )

    # Dictionary to store weighted candidates for each object
    candidates_with_boundary_quality = {}

    for object_id, candidates in candidates_by_object.items():
        mask_index = object_id - 1
        if mask_index < 0 or mask_index >= object_count:
            raise ValueError(f"object_id {object_id} has no corresponding SAM2 mask.")

        object_candidates = []
        for candidate in candidates:
            u, v = int(candidate.uv[0]), int(candidate.uv[1])

            boundary_distance_px = float(distance_maps[mask_index, v, u])
            normalized_boundary_distance = float(normalized_distance_maps[mask_index, v, u])

            boundary_quality = calculate_boundary_quality(
                normalized_boundary_distance=normalized_boundary_distance,
                exponent=exponent
            )

            object_candidates.append(
                replace(
                    candidate,
                    boundary_distance_px=boundary_distance_px,
                    boundary_distance_normalized=normalized_boundary_distance,
                    boundary_quality=boundary_quality,
                )
            )

        candidates_with_boundary_quality[object_id] = object_candidates

    return candidates_with_boundary_quality, distance_maps, normalized_distance_maps


