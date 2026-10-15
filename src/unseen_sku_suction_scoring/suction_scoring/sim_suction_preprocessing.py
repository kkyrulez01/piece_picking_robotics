"""
Combines cleaning, m-to-cm conversion, voxel downsampling, normal estimation,
sampling and model normalization.
"""
from dataclasses import dataclass

import numpy as np

from .normal_estimation import estimate_normals, orient_normals_towards_camera
from .point_sampling import sample_or_pad_points
from .pointcloud_utils import convert_points_to_cm, remove_invalid_points, voxel_downsample

@dataclass(frozen=True)
class PreparedSimSuctionCloud:
    """
    Point cloud prepared for Sim-Suction inference.
    """
    # Model inputs: normalized XYZ, unit normals
    features: np.ndarray
    # Real positions in cm. Padded rows contain 0
    xyz_cm: np.ndarray
    # Camera-facing unit normals. Padded rows contain zero.
    normals: np.ndarray
    # Maps each sample back to the original input cloud. Padded rows contain -1.
    source_indices: np.ndarray
    # True for real samples and False for padded samples.
    valid_sample_mask: np.ndarray
    # Parameters needed to reverse XYZ normalization.
    normalization_center_cm: np.ndarray
    normalization_scale_cm: float

    @property
    def real_sample_count(self):
        return int(np.count_nonzero(self.valid_sample_mask))

    @property
    def padded_sample_count(self):
        return int(np.count_nonzero(~self.valid_sample_mask))
    
def normalize_xyz_for_model(sampled_xyz_cm, valid_sample_mask):
    xyz = np.asarray(sampled_xyz_cm, dtype=np.float32)
    valid_mask = np.asarray(valid_sample_mask, dtype=bool)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"sampled_xyz_cm must have shape (N, 3). Received {xyz.shape}."
        )

    if valid_mask.shape != (xyz.shape[0],):
        raise ValueError(
            f"valid_sample_mask must have shape (N,). Received {valid_mask.shape}."
        )

    if not np.any(valid_mask):
        raise ValueError("There are no valid points to normalize.")

    valid_xyz = xyz[valid_mask].astype(np.float64)
    if not np.isfinite(valid_xyz).all():
        raise ValueError("Valid xyz points contain NaN or infinity.")

    # Perform unit-sphere normalization on the 3D point cloud
    center_cm = np.mean(valid_xyz, axis=0) # Calculate arithmetic centroid
    centered_valid_xyz = (valid_xyz - center_cm)

    distances_from_center = np.linalg.norm(centered_valid_xyz, axis=1)
    scale_cm = float(np.max(distances_from_center))
    if (not np.isfinite(scale_cm)or scale_cm <= 1e-8):
        raise ValueError(
            "Point cloud normalization scale is zero or "
            "invalid. The cloud may contain identical points."
        )

    normalized_xyz = np.zeros_like(xyz, dtype=np.float32)
    normalized_xyz[valid_mask] = (centered_valid_xyz / scale_cm).astype(np.float32)

    return normalized_xyz, center_cm.astype(np.float32), scale_cm

def prepare_sim_suction_cloud(
    points_xyz_m,
    camera_position_m=(0.0, 0.0, 0.0),
    voxel_size_cm=0.2,
    normal_radius_cm=1.5,
    maximum_normal_neighbours=30,
    target_count=5120,
    minimum_z_m=0.0,
    random_seed=42,
):
    """
    Prepares a foreground point cloud for Sim-Suction.

    Returns a PreparedSimSuctionCloud object.
    """
    camera_position_m = np.asarray(camera_position_m, dtype=np.float32)

    # 1. Remove Invalid points
    clean_xyz_m, clean_source_indices = remove_invalid_points(points_xyz_m,
                                                              minimum_z=minimum_z_m,
                                                              return_indices=True)

    if clean_xyz_m.shape[0] < 3:
        raise ValueError(
            "Fewer than three valid foreground points remain after cleaning."
        )

    # 2. Convert geometry to Sim-Suction's cm convention
    clean_xyz_cm = convert_points_to_cm(clean_xyz_m)
    camera_position_cm = camera_position_m * 100.0

    # 3. Voxel downsample
    if voxel_size_cm is None:
        downsampled_xyz_cm = clean_xyz_cm
        voxel_source_indices = np.arange(clean_xyz_cm.shape[0], dtype=np.int64)

    else:
        voxel_size_cm = float(voxel_size_cm)

        downsampled_xyz_cm, voxel_source_indices = voxel_downsample(clean_xyz_cm,
                                                                    voxel_size_cm,
                                                                    return_indices=True)

        if downsampled_xyz_cm.shape[0] < 3:
            raise ValueError(
                "Voxel downsampling left fewer than three points. "
                "Reduce voxel_size_cm."
            )

        # Map downsampled points back to the original foreground cloud.
        downsampled_source_indices = clean_source_indices[voxel_source_indices]

    # 4. Estimate and orient normals
    normals = estimate_normals(
        points_xyz_cm=downsampled_xyz_cm,
        radius_cm=normal_radius_cm,
        maximum_neighbours=maximum_normal_neighbours
    )

    normals = orient_normals_towards_camera(
        points_xyz_cm=downsampled_xyz_cm,
        normals=normals,
        camera_position_cm=camera_position_cm
    )

    # 5. Sample or pad to exactly target_count (5120)
    (sampled_xyz_cm, 
     sampled_normals, 
     sampled_source_indices, 
     valid_sample_mask) = sample_or_pad_points(points_xyz=downsampled_xyz_cm,
                                               normals=normals,
                                               source_indices=downsampled_source_indices,
                                               target_count=target_count,
                                               random_seed=random_seed)

    # 6. Normalize only the XYZ used by the Sim-Suction model
    (normalized_xyz,
     normalization_center_cm,
     normalization_scale_cm) = normalize_xyz_for_model(sampled_xyz_cm=sampled_xyz_cm,
                                                       valid_sample_mask=valid_sample_mask)

    # 7. Form Sim-Suction model's (5120, 6) input
    features = np.zeros((target_count, 6), dtype=np.float32)

    features[:, :3] = normalized_xyz
    features[:, 3:6] = sampled_normals

    # Padded rows must remain completely zero
    features[~valid_sample_mask] = 0.0

    # 8. Final validation
    expected_feature_shape = (target_count, 6) # (5120, 6)
    if features.shape != expected_feature_shape:
        raise RuntimeError(
            "Unexpected feature shape: "
            f"{features.shape}; expected "
            f"{expected_feature_shape}."
        )

    if not np.isfinite(features).all():
        raise RuntimeError("Prepared features contain NaN or infinity.")

    valid_normal_lengths = np.linalg.norm(sampled_normals[valid_sample_mask],axis=1)
    if not np.allclose(valid_normal_lengths, 1.0, atol=1e-4,):
        raise RuntimeError("Prepared normals are not unit length.")

    if not np.all(sampled_source_indices[~valid_sample_mask] == -1):
        raise RuntimeError("Padded source indices must be -1.")

    return PreparedSimSuctionCloud(
        features=features,
        xyz_cm=sampled_xyz_cm,
        normals=sampled_normals,
        source_indices=sampled_source_indices,
        valid_sample_mask=valid_sample_mask,
        normalization_center_cm=normalization_center_cm,
        normalization_scale_cm=normalization_scale_cm
    )