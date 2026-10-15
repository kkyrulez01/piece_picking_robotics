import cv2
import numpy as np
from pathlib import Path

def extract_foreground_cloud(
    background_xyz,
    scene_xyz,
    scene_rgb,
    roi_mask,
    minimum_height = 0.010,
    maximum_height=0.600
):
    if background_xyz.shape != scene_xyz.shape:
        raise ValueError("Background and scene cloud shapes differ")

    # Only compare valid background and scene points.
    valid = (
        np.isfinite(background_xyz).all(axis=2)
        & np.isfinite(scene_xyz).all(axis=2)
        & (background_xyz[..., 2] > 0)
        & (scene_xyz[..., 2] > 0)
    )

    # Only extract valid pointcloud points
    valid = (
        np.isfinite(background_xyz).all(axis=2)
        & np.isfinite(scene_xyz).all(axis=2)
        & (background_xyz[..., 2] > 0)
        & (scene_xyz[..., 2] > 0)
    )

    # Positive when the scene surface is closer to the camera.
    height = background_xyz[..., 2] - scene_xyz[..., 2]

    foreground_mask = (
        roi_mask.astype(bool)
        & valid
        & (height > minimum_height)
        & (height < maximum_height)
    )

    # Get foreground mask
    mask_uint8 = foreground_mask.astype(np.uint8) * 255

    mask_uint8 = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2
    )
    mask_uint8 = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    # Validity check again after morphology
    foreground_mask = (
        (mask_uint8 > 0)
        & roi_mask.astype(bool)
        & valid
        & (height > minimum_height)
        & (height < maximum_height)
    )
    rows, columns = np.nonzero(foreground_mask)

    foreground_xyz = scene_xyz[rows,columns]

    foreground_rgb = None
    if scene_rgb is not None:
        foreground_rgb = scene_rgb[rows,columns]

    foreground_uv = np.column_stack((columns, rows)).astype(np.uint32)

    return {
        "xyz": foreground_xyz,
        "rgb": foreground_rgb,
        "uv": foreground_uv,
        "mask": foreground_mask,
        "height": height,
    }

def load_point_cloud(file_path, npz_key=None):
    """
    Load XYZ points from a .npy or .npz file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Pointcloud file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(file_path, allow_pickle=False)

    elif suffix == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            available_keys = list(archive.files)

            if npz_key is not None:
                if npz_key not in archive:
                    raise KeyError(f"Key {npz_key} not found in {file_path}.")

                data = archive[npz_key]

            else:
                selected_key = None

                for candidate in ("xyz", "points", "point_cloud", "cloud"):
                    if candidate in archive:
                        selected_key = candidate
                        break

                if selected_key is None:
                    if len(available_keys) == 1:
                        selected_key = available_keys[0]
                    else:
                      raise ValueError(
                            f"Unable to select XYZ data from {file_path}. "
                            f"Available keys: {available_keys}. "
                            "Specify npz_key explicitly."
                        )

                data = archive[selected_key]

    else:
        raise ValueError(
            f"Unsupported point-cloud format '{suffix}'.Use .npy or .npz."
        )

    xyz = _as_xyz_array(data)

    if xyz.shape[0] == 0:
        raise ValueError(f"Point cloud is empty.")

    return xyz

def _as_xyz_array(points):
    """
    Convert supported point cloud layouts to an (N, 3) XYZ array.
    """
    points = np.asarray(points)

    if points.ndim == 2 and points.shape[1] >= 3:
        xyz = points[:, :3]

    elif points.ndim == 3 and points.shape[2] >= 3:
        xyz = points[..., :3].reshape(-1, 3)

    else:
        raise ValueError(
            "Point cloud must have shape (N, 3), (N, 6), "
            "(H, W, 3), or (H, W, 6). "
            f"Received {points.shape}."
        )

    return np.asarray(xyz, dtype=np.float32)

def remove_invalid_points(points_xyz, minimum_z=None, return_indices=False):
    """
    Remove points containing NaN or infinity.

    Takes in:
        points_xyz: Point cloud in a support XYZ layout.
        minimum_z: Optional minimum allowed Z value.
        return_indices: If True, also return the indices of retained input points. 
    """
    xyz = _as_xyz_array(points_xyz)
    valid_mask = np.isfinite(xyz).all(axis=1)

    if minimum_z is not None:
        valid_mask &= xyz[:, 2] > float(minimum_z)

    valid_indices = np.flatnonzero(valid_mask)
    valid_xyz = xyz[valid_indices]

    if return_indices:
        return valid_xyz, valid_indices

    return valid_xyz

def convert_points_to_cm(points_xyz_metres):
    """
    Convert XYZ coordinates to centimetres to match Sim-Suction's
    physical-distance conventions and prevent radius/voxel/offset mistakes.
    """
    xyz_metres = _as_xyz_array(points_xyz_metres)

    return (xyz_metres.astype(np.float32, copy=True) * 100.0)

def voxel_downsample(points_xyz, voxel_size, return_indices=False):
    """
    Downsample a point cloud using a voxel grid.
    Allows for faster processing, more even spatial distribution and
    removes redundant measurements.
    """
    xyz = _as_xyz_array(points_xyz)
    voxel_size = float(voxel_size)

    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be greater than 0.")

    if xyz.shape[0] == 0:
        empty_indices = np.empty((0,), dtype=np.int64)

        if return_indices:
            return xyz.copy(), empty_indices

        return xyz.copy()

    if not np.isfinite(xyz).all():
        raise ValueError(
            "voxel_downsample received invalid points. "
            "Call remove_invalid_points first."
        )

    voxel_coordinates = np.floor(xyz / voxel_size).astype(np.int64)
    _, inverse_indices = np.unique(voxel_coordinates, axis=0, return_inverse=True)

    voxel_count = int(inverse_indices.max()) + 1 # Get total no. of points
    # Tallies number of points falling into each voxel
    counts = np.bincount(inverse_indices, minlength=voxel_count).astype(np.float64)

    # Get centroid of voxel
    coordinate_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    for axis in range(3):
        coordinate_sums[:, axis] = np.bincount(
            inverse_indices,
            weights=xyz[:, axis],
            minlength=voxel_count
        )

    voxel_centroids = coordinate_sums / counts[:, None]

    # Calculate Euclidean squared distance from each point to its voxel centroid
    squared_distances = np.sum(
        (xyz.astype(np.float64) - voxel_centroids[inverse_indices]) ** 2,
        axis=1
    )

    input_indices = np.arange(xyz.shape[0], dtype=np.int64)
    # Sort by voxel, then distance to centroid, then input index.
    sorted_indices = np.lexsort((
        input_indices,
        squared_distances,
        inverse_indices
    ))

    sorted_voxel_ids = inverse_indices[sorted_indices]

    # Extract closes point to centroid per voxel
    first_in_voxel = np.empty(sorted_indices.shape[0], dtype=bool)
    first_in_voxel[0] = True
    first_in_voxel[1:] = (sorted_voxel_ids[1:] != sorted_voxel_ids[:-1])

    selected_indices = sorted_indices[first_in_voxel]

    # Preserve approximately the original point ordering
    selected_indices.sort()

    downsampled_xyz = xyz[selected_indices]
    if return_indices:
        return downsampled_xyz, selected_indices

    return downsampled_xyz