from pathlib import Path

import cv2
import numpy as np
import argparse

def load_helios_xyz(pointcloud_path):
    """
    Load Helioz XYZ from .npy or .npz and return an (N, 3) array.
    """
    pointcloud_path = Path(pointcloud_path)

    if pointcloud_path.suffix.lower() == ".npy":
        xyz = np.load(pointcloud_path)

    elif pointcloud_path.suffix.lower() == ".npz":
        with np.load(pointcloud_path) as data:
            if "xyz" not in data.files:
                raise KeyError(
                    f"{pointcloud_path} does not contain an 'xyz' array. "
                    f"Available keys: {data.files}"
                )
            xyz = data["xyz"]

    else:
        raise ValueError("Point cloud must be a .npy or .npz file.")

    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim < 2 or xyz.shape[-1] != 3:
        raise ValueError(
            f"XYZ must have shape (..., 3), got {xyz.shape}"
        )

    return xyz.reshape(-1, 3)

def load_extrinsic(extrinsic_path):
    """
    Load T_helios2realsense from derived extrinsic .npz.
    """
    with np.load(extrinsic_path) as data:
        transform = np.asarray(data["T_helios2realsense"], dtype=np.float64)

    if transform.shape != (4, 4):
        raise ValueError(
            f"T_helios2realsense must be 4x4, got {transform.shape}"
        )

    return transform

def load_realsense_intrinsics(calibration_path):
    """
    Load RealSense RGB camera matrix and distortion coefficients.
    """
    with np.load(calibration_path) as data:
        if "camera_matrix" not in data.files:
            raise KeyError("RealSense calibration must contain 'camera_matrix'.")

        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)

        if "dist_coeffs" in data.files:
            dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
        else:
            dist_coeffs = np.zeros(5, dtype=np.float64)

    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {camera_matrix.shape}")

    return camera_matrix, dist_coeffs

def transform_helios_to_realsense(helios_xyz_m, transform):
    """
    Transform Helios XYZ into the RealSense camera frame.

    Returns:
        rs_xyz:
            Valid transformed points with shape (M, 3).
        source_indices:
            Indices of those points in the flattened original Helios cloud.
    """
    helios_xyz_m = np.asarray(helios_xyz_m, dtype=np.float64)

    finite = np.isfinite(helios_xyz_m).all(axis=1)
    source_indices = np.flatnonzero(finite)
    points = helios_xyz_m[finite]

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    rs_xyz = (rotation @ points.T).T + translation

    # Keep only points in front of the RealSense camera.
    in_front = np.isfinite(rs_xyz).all(axis=1) & (rs_xyz[:, 2] > 0.0)

    return rs_xyz[in_front], source_indices[in_front]

def project_realsense_xyz(
    rs_xyz,
    camera_matrix,
    image_width,
    image_height,
):
    """
    Project RealSense-frame XYZ into the RealSense RGB image.

    Returns:
        u, v:
            Integer pixel coordinates.
        rs_xyz_visible:
            3D points whose projected pixels are inside the image.
        kept_indices:
            Indices into the input rs_xyz array.
    """
    rs_xyz = np.asarray(rs_xyz, dtype=np.float64)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = rs_xyz[:, 0]
    y = rs_xyz[:, 1]
    z = rs_xyz[:, 2]

    # Using pinhole camera model
    uv_float = np.column_stack((fx * x / z + cx,
                                fy * y / z + cy))

    # Round all (u,v) pairs to discrete pixels
    u = np.rint(uv_float[:, 0]).astype(np.int32)
    v = np.rint(uv_float[:, 1]).astype(np.int32)

    inside = ((u >= 0) & (u < image_width)  & (v >=0) & (v < image_height))

    kept_indices = np.flatnonzero(inside)

    return (u[inside], v[inside], rs_xyz[inside], kept_indices)

def apply_z_buffer(
    u: np.ndarray,
    v: np.ndarray,
    rs_xyz: np.ndarray
) -> np.ndarray:
    """
    Return indices of the nearest projected Helios point at each RGB pixel.

    This removes duplicate/occluded Helios points that project to the same
    RealSense pixel.
    """
    if len(rs_xyz) == 0:
        return np.empty(0, dtype=np.int64)

    # Sort by depth so the first occurrence of each pixel is the nearest.
    depth_order = np.argsort(rs_xyz[:, 2])
    pixel_pairs = np.column_stack((v[depth_order], u[depth_order]))

    _, first_positions = np.unique(
        pixel_pairs,
        axis=0,
        return_index=True,
    )

    keep = depth_order[first_positions]
    return np.sort(keep)

def project_helios_to_image(
    helios_xyz_m: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    image_shape: tuple[int, ...],
    z_buffer: bool = True,
):
    """
    Complete Helios XYZ -> RealSense RGB pixel mapping.

    Returns:
        rs_xyz:
            Helios points expressed in the RealSense frame.
        u, v:
            Corresponding RGB pixel coordinates.
        source_indices:
            Indices into the flattened original Helios XYZ array.
    """
    image_height, image_width = image_shape[:2]

    # XYZ in RealSense frame
    rs_xyz, source_indices = transform_helios_to_realsense(helios_xyz_m, transform)

    u, v, rs_xyz, projected_indices = project_realsense_xyz(
        rs_xyz=rs_xyz,
        camera_matrix=camera_matrix,
        image_width=image_width,
        image_height=image_height
    )

    source_indices = source_indices[projected_indices]

    # Usually don't use Z buffer for merging pointclouds with RGB
    # Z buffer results in points removed
    if z_buffer:
        keep = apply_z_buffer(u, v, rs_xyz)
        u = u[keep]
        v = v[keep]
        rs_xyz = rs_xyz[keep]
        source_indices = source_indices[keep]

    return rs_xyz, u, v, source_indices

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project a Helios2 point cloud onto a RealSense RGB image."
    )
    parser.add_argument(
        "--pointcloud",
        required=True,
        help="Helios point cloud .npy or .npz. NPZ must contain key 'xyz'.",
    )
    parser.add_argument(
        "--rgb",
        required=True,
        help="RealSense RGB image path.",
    )
    parser.add_argument(
        "--extrinsic",
        default="/home/support/unseen_sku_ws/tools/camera_calibration/calibration_data/helios_to_realsense_260922/helios_to_realsense_260922.npz",
        help="Derived Helios2 -> RealSense extrinsic .npz.",
    )
    parser.add_argument(
        "--realsense-calibration",
        default="/home/support/unseen_sku_ws/tools/camera_calibration/calibration_data/realsense_D455f_260922/realsense_D455f_260922.npz",
        help="RealSense calibration .npz containing camera_matrix/dist_coeffs.",
    )
    parser.add_argument(
        "--z-buffer",
        action="store_true",
        help="Keep only the nearest Helios point at each RealSense RGB pixel.",
    )
    # parser.add_argument(
    #     "--output",
    #     default="helios_projected_to_realsense_rgb.npz",
    #     help="Output correspondence .npz.",
    # )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional output path for an RGB projection overlay image.",
    )
    args = parser.parse_args()

    helios_xyz = load_helios_xyz(args.pointcloud) # Already in metres
    
    transform = load_extrinsic(args.extrinsic)
    camera_matrix, _ = load_realsense_intrinsics(args.realsense_calibration)

    # OpenCV loads BGR. Keep it for overlay writing, but save sampled colors
    # in conventional RGB channel order.
    image_bgr = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {args.rgb}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    rs_xyz, u, v, source_indices = project_helios_to_image(
        helios_xyz_m=helios_xyz,
        transform=transform,
        camera_matrix=camera_matrix,
        image_shape=image_rgb.shape,
        z_buffer=args.z_buffer,
    )

    colors_rgb = image_rgb[v, u]

    rgb_path = Path(args.rgb)
    # /home/support/unseen_sku_ws/data/realsense/scene/images/image_0035.png
    index_stem = rgb_path.stem
    index = int(index_stem.rsplit("_", 1)[1])
    output_dir =  Path("/home/support/unseen_sku_ws/outputs") / f"scene_{index:04d}" / "mask_fusion"
    output_path = output_dir / f"helios_projected_to_realsense_rgb_{index:04d}.npz"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        xyz_realsense=rs_xyz.astype(np.float32),
        u=u,
        v=v,
        rgb=colors_rgb,
        source_indices=source_indices,
        xyz_unit=np.array("m"),
    )

    if args.overlay:
        overlay = image_bgr.copy()

        # Depth of each projected Helios point in the RealSense frame
        depth = rs_xyz[:, 2]

        # Ignore extreme outliers when defining colour range
        near_depth = np.percentile(depth, 2)
        far_depth = np.percentile(depth, 98)

        if far_depth <= near_depth:
            far_depth = near_depth + 1e-6

        # Normalize depth to 0 -> 255
        depth_normalized = (depth - near_depth) / (far_depth - near_depth)
        depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

        # Invert:
        # near -> high value -> red/yellow in TURBO
        # far  -> low value  -> blue/purple
        depth_u8 = ((1.0 - depth_normalized) * 255).astype(np.uint8)

        # Convert scalar depth values to BGR colours
        depth_colours = cv2.applyColorMap(
            depth_u8.reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        ).reshape(-1, 3)

        # Draw sparsely so the RGB image remains visible
        max_overlay_points = 30000
        step = max(1, len(u) // max_overlay_points,)

        for i in range(0, len(u), step):
            colour = tuple(int(value) for value in depth_colours[i])
            cv2.circle(
                overlay,
                (int(u[i]), int(v[i])),
                1,
                colour,
                -1,
            )

        overlay_path = Path(args.overlay)
        overlay_path.parent.mkdir(parents=True,  exist_ok=True,)

        cv2.imwrite(str(overlay_path),  overlay)

        print(
            f"Depth colour range: "
            f"{near_depth:.3f} m -> {far_depth:.3f} m"
        )

        # overlay = image_bgr.copy()

        # # Draw sparsely for readability if the cloud is very dense.
        # max_overlay_points = 20000
        # step = max(1, len(u) // max_overlay_points)

        # for px, py in zip(u[::step], v[::step]):
        #     cv2.circle(
        #         overlay,
        #         (int(px), int(py)),
        #         1,
        #         (0, 255, 0),
        #         -1,
        #     )

        # overlay_path = Path(args.overlay)
        # overlay_path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(str(overlay_path), overlay)

    print(f"Input Helios points: {len(helios_xyz)}")
    print(f"Projected points inside RGB image: {len(rs_xyz)}")
    print(f"Saved correspondence: {output_path.resolve()}")

    if args.overlay:
        print(f"Saved overlay: {Path(args.overlay).resolve()}")

if __name__ == "__main__":
    main()