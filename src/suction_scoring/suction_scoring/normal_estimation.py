"""
The foreground point should have already been converted to centimetres.
"""

import numpy as np
import open3d as o3d

def _validate_xyz(points_xyz_cm):
    """
    Validate and return (N, 3) float64 XYZ array.
    """
    xyz = np.asarray(points_xyz_cm)

    if (xyz.ndim != 2 or xyz.shape[1] != 3):
        raise ValueError(f"points_xyz_cm must have shape (N,3). Received {xyz.shape}")

    if xyz.shape[0] < 3:
        raise ValueError("At least 3 points required to estimate normals.")

    if not np.isfinite(xyz).all():
        raise ValueError(
            "points_xyz_cm contains NaN or infinite values. "
            "Remove invalid points before estimating normals."
        )

    return np.ascontiguousarray(xyz, dtype=np.float64)

def estimate_normals(points_xyz_cm, radius_cm=1.5, maximum_neighbours=30):
    """
    Estimate one surface normal for every point.

    Arguments:
        points_xyz_cm:
            Clean point cloud with shape (N, 3) in cm where N is no. of points.
        radius_cm:
            Maximum neighbourhood search radius in cm.
        maximum_neighbours:
            Max no. of neighbouring points used for each normal estimate.
    
    Returns:
        normals:
            Array with shape (N, 3). Every normal is approximately unit length.
    """
    xyz = _validate_xyz(points_xyz_cm)
    radius_cm = float(radius_cm)
    maximum_neighbours = int(maximum_neighbours)

    if radius_cm <= 0.0:
        raise ValueError("radius_cm must be greater than 0.")
    if maximum_neighbours < 3:
        raise ValueError("Maximum neighbours must be at least 3.")

    # Create empty point cloud container
    point_cloud = o3d.geometry.PointCloud()
    # Assign points into the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_cm,
            max_nn=maximum_neighbours
        )
    )

    normals = np.asarray(point_cloud.normals, dtype=np.float64)
    if normals.shape != xyz.shape:
        raise RuntimeError(f"Open3D returned an unexpected normal array with shape {normals.shape}")

    # Calculate normal vector magnitudes
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = np.isfinite(normals).all(axis=1) & np.isfinite(normal_lengths) & (normal_lengths > 1e-8)

    if not np.all(valid_normals):
        invalid_count = int(np.count_nonzero(~valid_normals))

        raise RuntimeError(f"Normal estimation produced {invalid_count} invalid normals.")

    # Normalize normals to unit vectors
    normals = normals / normal_lengths[:, None]

    return normals.astype(np.float32)

def orient_normals_towards_camera(points_xyz_cm, normals, camera_position_cm=(0.0, 0.0, 0.0)):
    """
    Flip normals so they point towards the camera.
    """
    xyz = _validate_xyz(points_xyz_cm)
    normals = np.asarray(normals, dtype=np.float64)

    if normals.shape != xyz.shape:
        raise ValueError(
            "normals should have same shape as points_xyz_cm." 
            f"Received points {xyz.shape} and normals {normals.shape}."
        )

    if not np.isfinite(normals).all():
        raise ValueError("normals contains NaN or infinite values.")

    camera_position = np.asarray(camera_position_cm, dtype=np.float64)
    if camera_position.shape != (3,):
        raise ValueError("camera_position_cm must contain exactly 3 values.")
    if not np.isfinite(camera_position).all():
        raise ValueError("camera_position_cm contains invalid values.")

    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(normal_lengths <= 1e-8):
        raise ValueError("normals contains zero-length vectors.")

    oriented_normals = normals / normal_lengths[:, None]

    # Vector from each surface point toward the camera
    directions_to_camera = camera_position[None, :] - xyz

    camera_distances = np.linalg.norm(directions_to_camera, axis=1)
    if np.any(camera_distances <= 1e-8):
        raise ValueError(
            "A point is located at the camera position, so its " \
            "viewing position is undefined.")

    # Computes dot product between each normal vector and camera direction
    alignment = np.einsum("ij, ij->i", oriented_normals, directions_to_camera)

    # Negative dot product means normal points away from the camera
    flip_mask = alignment < 0.0
    oriented_normals[flip_mask] *= -1.0

    return oriented_normals.astype(np.float32)   