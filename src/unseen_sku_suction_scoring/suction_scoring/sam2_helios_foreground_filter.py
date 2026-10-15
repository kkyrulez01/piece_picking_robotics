#!/usr/bin/env python3

import cv2
import numpy as np

def build_sam2_union_mask(object_masks, dilation_size=5):
    """
    Combine all accepted SAM2 object masks into one foreground mask.

    Args:
        object_masks:
            Boolean or uint8 array with shape (K, H, W), where K is
            the number of accepted SAM2 object masks.

        dilation_size:
            Kernel size used to slightly expand the union mask.
            This helps tolerate small projection/calibration errors
            between the 3D camera and RealSense RGB image.

            Set to 1 to disable dilation.

    Returns:
        combined_mask:
            Boolean array with shape (H, W).
    """
    object_masks = np.asarray(object_masks, dtype=bool)

    if object_masks.ndim != 3:
        raise ValueError(
            "object_masks must have shape (K, H, W), "
            f"but got {object_masks.shape}"
        )

    if object_masks.shape[0] == 0:
        raise ValueError("No SAM2 object masks were provided.")

    # Union of every accepted SAM2 object mask
    combined_mask = np.any(object_masks, axis=0)

    if dilation_size > 1:
        kernel = np.ones((dilation_size, dilation_size), dtype=np.uint8)

        combined_mask = cv2.dilate(combined_mask.astype(np.uint8),
                                    kernel,
                                    iterations=1).astype(bool)

    return combined_mask

def foreground_combined_sam2_intersect(
    xyz,
    uv,
    object_masks,
    rgb=None,
    xyz_realsense=None,
    scene_indices=None,
    dilation_size=5
):
    """
    Keep only foreground 3D points whose projected RealSense pixel
    lies inside the union of the accepted SAM2 masks.

    The input foreground cloud is assumed to already contain only
    geometrically detected foreground points.

    Therefore the resulting cloud is effectively:
        geometric_foreground INTERSECTION sam2_union

    Args:
        xyz:
            Nx3 point cloud.
        uv:
            Nx2 projected RealSense image coordinates.
            Expected order is [u, v].
        object_masks:
            SAM2 masks with shape (K, H, W).
        rgb:
            Optional Nx3 point colours.
        xyz_realsense:
            Optional Nx3 points transformed into RealSense coordinates.
        scene_indices:
            Optional per-point source indices.
        dilation_size:
            SAM2 union-mask dilation kernel size.

    Returns:
        Dictionary containing the filtered per-point arrays.

        Additional fields:
            keep_mask:
                Boolean mask referring to the original input cloud.
            source_indices:
                Original point indices retained by this filter.
            combined_sam_mask:
                The union mask used for filtering.
    """
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)

    point_count = len(xyz)
    if len(uv) != point_count:
        raise ValueError(
            "xyz and uv must have the same number of points: "
            f"{len(xyz)} != {len(uv)}"
        )

    # Verify all optional per-point arrays remain aligned
    optional_arrays = {
        "rgb": rgb,
        "xyz_realsense": xyz_realsense,
        "scene_indices": scene_indices,
    }

    for name, array in optional_arrays.items():
        if array is not None and len(array) != point_count:
            raise ValueError(
                f"{name} has {len(array)} entries, "
                f"but xyz has {point_count}"
            )

    # Build the combined SAM2 mask
    combined_sam_mask = build_sam2_union_mask(object_masks=object_masks, 
                                              dilation_size=dilation_size)

    image_height, image_width = combined_sam_mask.shape
    # Projected image coordinates.
    #
    # uv[:, 0] -> horizontal coordinate u
    # uv[:, 1] -> vertical coordinate v
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)

    # Only access pixels that actually fall inside the RGB image.
    inside_image = (
        (u >= 0)
        & (u < image_width)
        & (v >= 0)
        & (v < image_height)
    )

    keep_mask = np.zeros(point_count, dtype=bool)

    valid_indices = np.flatnonzero(inside_image)
    # Foreground point must project into the SAM2 union
    keep_mask[valid_indices] = combined_sam_mask[v[valid_indices], u[valid_indices]]

    source_indices = np.flatnonzero(keep_mask)
    retained_count = len(source_indices)

    percentage = 100.0 * (retained_count / point_count) if point_count > 0 else 0.0

    print("\nSAM2 foreground filtering")
    print("-------------------------")
    print(f"Input foreground points: {point_count}")
    print(f"Points inside image:      {np.count_nonzero(inside_image)}")
    print(f"Points inside SAM2 union: {retained_count}")
    print(f"Retained:                 {percentage:.2f}%")

    if retained_count == 0:
        raise RuntimeError(
            "SAM2 union filtering removed every foreground point. "
            "Check UV projection, mask resolution, or calibration."
        )

    result = {
        "xyz": xyz[keep_mask],
        "uv": uv[keep_mask],
        "keep_mask": keep_mask,
        "source_indices": source_indices,
        "combined_sam_mask": combined_sam_mask,
    }

    if rgb is not None:
        result["rgb"] = np.asarray(rgb)[keep_mask]

    if xyz_realsense is not None:
        result["xyz_realsense"] = np.asarray(xyz_realsense)[keep_mask]

    if scene_indices is not None:
        result["scene_indices"] = np.asarray(scene_indices[keep_mask])

    return result