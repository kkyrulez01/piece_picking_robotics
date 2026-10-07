#!/usr/bin/env python3

from dataclasses import replace
import numpy as np

def orient_normal_towards_camera(
    candidate_xyz_m,
    candidate_normal,
    camera_position_m
):
    """
    Ensure the candidate normal points from the object surface
    towards the camera / robot approach side.
    """
    candidate_xyz_m = np.asarray(candidate_xyz_m, dtype=np.float64)
    normal = np.asarray(candidate_normal, dtype=np.float64)
    camera_position_m = np.asarray(camera_position_m, dtype=np.float64)

    normal_length = np.linalg.norm(normal)
    normal = normal / normal_length # Ensure unit length

    direction_to_camera = camera_position_m - candidate_xyz_m
    # Flip normal if necessary
    if np.dot(normal, direction_to_camera) < 0.0:
        normal = -normal

    return normal

def calculate_candidate_clearance(
    candidate_xyz_m,
    candidate_normal,
    scene_points_m,
    camera_position_m,
    tool_radius_m,
    approach_height_m,
    minimum_clearance_height_m,
    safety_margin_m,
):
    """
    Check whether foreground geometry enters the cylindrical
    approach volume of the suction tool.

    The cylinder starts slightly above the contact surface so
    that the object's normal contact surface is not incorrectly
    considered a collision.

    Returns:
        clearance_valid
        collision_count
        minimum_margin_m
    """
    candidate_xyz_m = np.asarray(candidate_xyz_m, dtype=np.float64)
    scene_points_m = np.asarray(scene_points_m, dtype=np.float64)

    normal = orient_normal_towards_camera(
        candidate_xyz_m=candidate_xyz_m,
        candidate_normal=candidate_normal,
        camera_position_m=camera_position_m,
    )

    relative_points = scene_points_m - candidate_xyz_m

    # Distance along candidate approach direction
    axial_distance = relative_points @ normal
    # Squared Euclidean distance from candidate
    total_distance_squared = np.sum(relative_points ** 2, axis=1)
    # Distance from tool's central axis
    radial_distance_squared = total_distance_squared - axial_distance ** 2
    radial_distance_squared = np.maximum(radial_distance_squared, 0.0)

    radial_distance = np.sqrt(radial_distance_squared)

    # Only points lying in front of the suction surface and inside
    # the approach distance matter
    approach_region = (axial_distance >= minimum_clearance_height_m) & (axial_distance <= approach_height_m)
    effective_radius_m = tool_radius_m + safety_margin_m

    # Potential tool collision
    collision_mask = approach_region & (radial_distance <= effective_radius_m)

    collision_count = int(np.count_nonzero(collision_mask))
    clearance_valid = (collision_count == 0) # Only valid if there are no collision points

    # Determine distance from the closest observed obstacle
    # to the side of the tool cylinder.
    approach_radial_distances = radial_distance[approach_region]

    # Positive: Scene is completely clear
    # Zero: Obstacle sits directly on the boundary of the safety cylinder
    # Negative: There is a collision
    if approach_radial_distances.size > 0:
        minimum_margin_m = float(np.min(approach_radial_distances) - effective_radius_m)
    else:
        minimum_margin_m = None

    return clearance_valid, collision_count, minimum_margin_m

def calculate_candidate_clearance_quality(
    candidates_by_object,
    scene_points_m,
    camera_position_m,
    tool_radius_m,
    approach_height_m,
    minimum_clearance_height_m,
    safety_margin_m,
    reject_collisions=False,
):
    """
    Evaluate tool clearance for all suction candidates.

    If reject_collisions=True:
        candidates whose tool volume intersects observed
        foreground geometry are removed.

    Otherwise all candidates are retained for debugging.
    """
    output_candidates = {}

    for object_id, candidates in candidates_by_object.items():
        object_candidates = []

        for candidate in candidates:
            clearance_valid, collision_count, minimum_margin_m,= calculate_candidate_clearance(
                candidate_xyz_m=candidate.xyz_m,
                candidate_normal=candidate.normal,
                scene_points_m=scene_points_m,
                camera_position_m=camera_position_m,
                tool_radius_m=tool_radius_m,
                approach_height_m=approach_height_m,
                minimum_clearance_height_m=minimum_clearance_height_m,
                safety_margin_m=safety_margin_m
            )

            updated_candidate = replace(
                candidate,
                clearance_valid=clearance_valid,
                clearance_collision_count=collision_count,
                clearance_minimum_margin_m=minimum_margin_m,
            )

            if (reject_collisions and not clearance_valid):
                continue

            object_candidates.append(updated_candidate)

        output_candidates[object_id] = object_candidates

    return output_candidates