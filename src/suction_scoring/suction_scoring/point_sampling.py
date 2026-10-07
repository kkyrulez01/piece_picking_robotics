"""
This module gurantees 5120 points for Sim-Suction while preserving
the mapping back to the original foreground.
"""
import numpy as np

def sample_or_pad_points(
    points_xyz,
    normals,
    source_indices=None,
    target_count=5120,
    random_seed=42,
):
    xyz = np.asarray(points_xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"points_xyz must have shape (N, 3). Received {xyz.shape}."
        )

    if normals.shape != xyz.shape:
        raise ValueError(
            "Normals should have same shape as points_xyz." \
            f"Received points {xyz.shape} and normals {normals.shape}"
        )

    point_count = xyz.shape[0]
    target_count = int(target_count)

    if point_count == 0:
        raise ValueError("Cannot sample from empty point cloud.")

    if not np.isfinite(xyz).all():
        raise ValueError("points_xyz contains NaN or infinite values.")
    if not np.isfinite(normals).all():
        raise ValueError("normals contain NaN or infinite values.")

    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(normal_lengths <= 1e-8):
        raise ValueError("normals contains zero-length vectors.")

    # Ensure all real normals are unit length.
    normals = normals / normal_lengths[:, None]

    if source_indices is None:
        source_indices = np.arange(point_count, dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)

    # Pre-allocate memory buffers of length 5120
    sampled_xyz = np.zeros((target_count, 3), dtype=np.float32)
    sampled_normals = np.zeros((target_count, 3), dtype=np.float32)
    sampled_source_indices = np.full(target_count, fill_value=-1, dtype=np.int64)
    valid_sample_mask = np.zeros(target_count, dtype=bool)

    if point_count > target_count:
        rng = np.random.default_rng(random_seed)

        selected_indices = rng.choice(point_count, size=target_count, replace=False)

        sampled_xyz[:] = xyz[selected_indices]
        sampled_normals[:] = normals[selected_indices]
        sampled_source_indices[:] = source_indices[selected_indices]
        valid_sample_mask[:] = True

    elif point_count == target_count:
        sampled_xyz[:] = xyz
        sampled_normals[:] = normals
        sampled_source_indices[:] = source_indices
        valid_sample_mask[:] = True

    else:
        sampled_xyz[:point_count] = xyz
        sampled_normals[:point_count] = normals
        sampled_source_indices[:point_count] = source_indices
        valid_sample_mask[:point_count] = True

    return (sampled_xyz, sampled_normals, sampled_source_indices, valid_sample_mask)