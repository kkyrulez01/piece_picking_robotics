#!/usr/bin/env python3

from dataclasses import replace
import numpy as np

def build_tangent_basis(normal):
    """
    Build two orthogonal tangent vectors perpendicular
    to the supplied surface normal.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal_magnitude = np.linalg.norm(normal)

    # Normalize the normal vector so that it is unit length
    normal = normal / normal_magnitude

    # Pick a reference axis that is not almost parallel
    # to the surface normal.
    if abs(normal[2]) < 0.9:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    tangent_x =  np.cross(normal, reference)
    tangent_x /= np.linalg.norm(tangent_x)

    tangent_y =  np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)

    return normal, tangent_x, tangent_y

def calculate_footprint_support(
    candidate_xyz_m,
    candidate_normal,
    object_points_m,
    cup_radius_m,
    grid_resolution_m,
    maximum_surface_deviation_m
):
    """
    Calculate how much of a circular suction-cup footprint
    is supported by object surface.

    Returns:
        support_ratio
        footprint_point_count
        height_p90_m
    """
    candidate_xyz_m = np.asarray(candidate_xyz_m, dtype=np.float64)
    object_points_m = np.asarray(object_points_m, dtype=np.float64)

    normal, tangent_x, tangent_y = build_tangent_basis(candidate_normal)

    # Candidate-relative point positions
    relative_points = object_points_m - candidate_xyz_m

    # Distance in normal and tangent directions
    height = relative_points @ normal
    local_x = relative_points @ tangent_x
    local_y = relative_points @ tangent_y

    radial_distance = np.sqrt(local_x ** 2 + local_y ** 2)
    inside_radius = radial_distance <= cup_radius_m

    # Get no. of points inside suction cup radius
    footprint_point_count = int(np.count_nonzero(inside_radius))

    if footprint_point_count == 0:
        return 0.0, 0, None

    footprint_heights = np.abs(height[inside_radius])
    # Compute 90th percentile of absolute height deviation
    # This represent the local bumpiness
    height_p90_m = float(np.percentile(footprint_heights, 90.0))

    # A supporting point must:
    #
    # 1. lie inside the cup radius
    # 2. lie sufficiently close to the candidate tangent plane
    support_mask = inside_radius & (np.abs(height) <= maximum_surface_deviation_m)

    support_x = local_x[support_mask] # In tangent_x direction
    support_y = local_y[support_mask] # In tangent_y direction

    # Construct occupancy grid over physical cup footprint
    diameter_m = 2.0 * cup_radius_m
    grid_size = int(np.ceil(diameter_m / grid_resolution_m))

    cell_centres = -cup_radius_m + (np.arange(grid_size) + 0.5) * grid_resolution_m
    grid_x,  grid_y = np.meshgrid(cell_centres, cell_centres, indexing="xy")

    expected_cells = grid_x ** 2 + grid_y ** 2 <= cup_radius_m ** 2
    expected_cell_count = int(np.count_nonzero(expected_cells))

    if expected_cell_count == 0:
        raise ValueError(
            "Footprint grid contains no cells. Check cup radius and grid resolution."
        )

    occupancy = np.zeros((grid_size, grid_size), dtype=bool)

    if support_x.size > 0:
        x_indices = np.floor((support_x + cup_radius_m) / grid_resolution_m).astype(np.intp)
        y_indices = np.floor((support_y + cup_radius_m) / grid_resolution_m).astype(np.intp)

        valid_indices = (x_indices >= 0) & (x_indices < grid_size) & (y_indices >= 0) & (y_indices < grid_size)
        x_indices = x_indices[valid_indices].astype(np.intp, copy=False)
        y_indices = y_indices[ valid_indices].astype(np.intp, copy=False)

        occupancy[y_indices, x_indices] = True

    # Only count cells physically inside circular cup.
    occupied_inside_cup = occupancy & expected_cells
    occupied_cell_count = int(np.count_nonzero(occupied_inside_cup))

    support_ratio = occupied_cell_count / expected_cell_count

    return float(support_ratio), footprint_point_count, height_p90_m

def calculate_candidate_footprint_quality(
    candidates_by_object,
    object_points_by_id,
    cup_radius_m,
    grid_resolution_m,
    maximum_surface_deviation_m,
):
    """
    Calculate suction footprint quality for every candidate.

    For this first implementation:

        footprint_quality = footprint_support_ratio

    No candidate is rejected yet.
    """
    output_candidates = {}

    for object_id, candidates in candidates_by_object.items():
        object_points_m = object_points_by_id.get(object_id)

        object_candidates = []
        for candidate in candidates:
            support_ratio, footprint_point_count, height_p90_m = calculate_footprint_support(
                candidate_xyz_m=candidate.xyz_m,
                candidate_normal=candidate.normal,
                object_points_m=object_points_m,
                cup_radius_m=cup_radius_m,
                grid_resolution_m=grid_resolution_m,
                maximum_surface_deviation_m=maximum_surface_deviation_m
            )

            object_candidates.append(
                replace(
                    candidate,
                    footprint_support_ratio=support_ratio,
                    footprint_point_count=footprint_point_count,
                    footprint_height_p90_m=height_p90_m,
                    footprint_quality=support_ratio
                )
            )

        output_candidates[object_id] = object_candidates

    return output_candidates


