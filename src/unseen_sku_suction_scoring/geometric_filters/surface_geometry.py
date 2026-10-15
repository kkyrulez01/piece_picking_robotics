#!/usr/bin/env python3

import numpy as np
from dataclasses import replace

def build_object_surface_points(foreground_xyz_m, foreground_uv, object_masks):
    """
    Associate the full-resolution foreground point cloud with SAM2 masks.

    This is intentionally done using the full foreground cloud rather
    than the downsampled 5,120-point Sim-Suction input.

    Returns:
        object_points_by_id:
            dict:
                object_id -> (N, 3) XYZ points in metres
    """
    foreground_xyz_m = np.asarray(foreground_xyz_m, dtype=np.float32)
    foreground_uv = np.asarray(foreground_uv, dtype=np.int64)
    object_masks = np.asarray(object_masks, dtype=bool)

    image_height = object_masks.shape[1]
    image_width = object_masks.shape[2]
    u, v = foreground_uv[:, 0], foreground_uv[:, 1]

    inside_image = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    finite_xyz = np.isfinite(foreground_xyz_m).all(axis=1)

    usable = inside_image & finite_xyz
    usable_indices = np.flatnonzero(usable)

    object_points_by_id = {}

    for mask_index, object_mask in enumerate(object_masks):
        object_id = mask_index + 1

        belongs_to_object = np.zeros(foreground_xyz_m.shape[0], dtype=bool)
        belongs_to_object[usable_indices] = object_mask[
            v[usable_indices], 
            u[usable_indices],
        ]

        object_points = foreground_xyz_m[belongs_to_object]
        object_points_by_id[object_id] = object_points

    return object_points_by_id

def calculate_surface_variation(
    candidate_xyz_m,
    object_points_m,
    neighbourhood_radius_m=0.015,
    minimum_neighbours=20
):
    """
    Measure local surface variation using PCA.
    Surface variation falls between 0 to 1/3, where:
        0 refers to a flat plane,
        1/3 refers to a sharp edge or corner.

    Lower value:
        locally planar / smooth

    Higher value:
        curved, edge-like, corner-like, or noisy

    Returns:
        surface_variation
        neighbour_count

    If there are too few neighbours:
        surface_variation = None
    """
    candidate_xyz_m = np.asarray(candidate_xyz_m,dtype=np.float64)
    object_points_m = np.asarray(object_points_m,dtype=np.float64)

    # Find points within the physical neighbourhood
    delta = object_points_m - candidate_xyz_m 
    distance_squared = np.sum(delta * delta, axis=1)

    neighbour_mask = (distance_squared <= neighbourhood_radius_m ** 2)
    # Apply the neighbour mask to get all neighbours
    neighbours = object_points_m[neighbour_mask]
    neighbour_count = neighbours.shape[0]

    if neighbour_count < minimum_neighbours:
        return None, neighbour_count

    # Centre the local points
    centred = neighbours - np.mean(neighbours, axis=0, keepdims=True)

    covariance = centred.T @ centred / float(neighbour_count) # Covariance formula
    # Return eigenvalues in ascending order
    eigenvalues = np.linalg.eigvalsh(covariance)

    # Small negative values can occur due to numerical precision
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # (λ0 + λ1 + λ2) should not be 0
    eigenvalue_sum = float(np.sum(eigenvalues))
    if eigenvalue_sum <= 1e-12:
        return None, neighbour_count

    # Surface variation (σ) = λ0 / (λ0 + λ1 + λ2)
    surface_variation = float(eigenvalues[0] / eigenvalue_sum)

    return surface_variation, neighbour_count

def calculate_surface_geometry_quality(
    surface_variation,
    maximum_surface_variation=0.02,
    exponent=1.0,
    missing_quality=0.5
):
    """
    Convert PCA surface variation into a normalized quality score.

    Returns:
        surface_quality in [0, 1]

        1.0:
            Very planar / smooth local surface.

        0.0:
            Surface variation is at or above
            maximum_surface_variation.

        missing_quality:
            Used when surface variation cannot be calculated,
            for example because a small object has too few neighbours.
    """
    # Surface variation could not be calculated.
    if surface_variation is None:
        return float(missing_quality)
    
    # Clamp normalized surface variation between 0 and 1
    normalized_variation = np.clip(surface_variation / maximum_surface_variation, 0.0, 1.0)

    # 1.0 = Planar, 0.0 = high surface variation
    surface_quality = 1.0 - normalized_variation ** exponent

    return float(surface_quality)

def calculate_candidate_surface_geometry_quality(
    candidates_by_object,
    object_points_by_id,
    neighbourhood_radius_m=0.015,
    minimum_neighbours=20,
    maximum_surface_variation=0.02,
    exponent=1.0,
    missing_quality=0.5
):
    """
    Calculate local surface-geometry quality for suction candidates.

    This function does NOT:
        - modify ranking_score
        - reject candidates
        - sort candidates
        - change candidate rank

    It only measures and stores:

        surface_neighbour_count
        surface_variation
        surface_quality
    """
    candidates_with_surface_quality = {}

    for object_id, candidates in candidates_by_object.items():
        object_points_m = object_points_by_id.get(object_id)

        object_candidates = []
        for candidate in candidates:
            surface_variation, neighbour_count = calculate_surface_variation(
                candidate_xyz_m=candidate.xyz_m,
                object_points_m=object_points_m,
                neighbourhood_radius_m=neighbourhood_radius_m,
                minimum_neighbours=minimum_neighbours
            )

            surface_quality = calculate_surface_geometry_quality(
                surface_variation=surface_variation,
                maximum_surface_variation=maximum_surface_variation,
                exponent=exponent,
                missing_quality=missing_quality
            )

            object_candidates.append(
                replace(
                    candidate,
                    surface_neighbour_count=neighbour_count,
                    surface_variation=surface_variation,
                    surface_quality=surface_quality,
                )
            )

        candidates_with_surface_quality[object_id] = object_candidates

    return candidates_with_surface_quality