#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


# ============================================================
# Paths
# ============================================================

BACKGROUND_PATH = Path(
    "/home/support/unseen_sku_ws/data/helios2/background/background_tote.npz"
)

SCENE_PATH = Path(
    "/home/support/unseen_sku_ws/data/helios2/scene/scene_0037.npz"
)

RGB_PATH = Path(
    "/home/support/unseen_sku_ws/data/realsense/scene/images/image_0037.png"
)

TRANSFORM_PATH = Path(
    "/home/support/unseen_sku_ws/tools/camera_calibration/"
    "calibration_data/helios_to_realsense_260922/"
    "helios_to_realsense_260922.npz"
)

REALSENSE_CALIBRATION_PATH = Path(
    "/home/support/unseen_sku_ws/tools/camera_calibration/"
    "calibration_data/realsense_D455f_260922/"
    "realsense_D455f_260922.npz"
)


# ============================================================
# Outputs
# ============================================================

OUTPUT_PATH = Path(
    "/home/support/unseen_sku_ws/outputs/test_0037.npz"
)

PLY_OUTPUT_PATH = Path(
    "/home/support/unseen_sku_ws/outputs/test_0037.ply"
)

PROJECTED_MASK_OUTPUT_PATH = Path(
    "/home/support/unseen_sku_ws/outputs/test_0037_projected_mask.png"
)

OVERLAY_OUTPUT_PATH = Path(
    "/home/support/unseen_sku_ws/outputs/test_0037_overlay.png"
)

ROI_OVERLAY_OUTPUT_PATH = Path(
    "/home/support/unseen_sku_ws/outputs/test_0037_overlay_roi.png"
)


# ============================================================
# Background subtraction settings
# ============================================================

# Metres
FOREGROUND_THRESHOLD = 0.015       # 5 mm
MAX_FOREGROUND_HEIGHT = 0.500      # 500 mm


# ============================================================
# RealSense ROI
#
# NumPy convention:
# x_min inclusive
# x_max exclusive
# y_min inclusive
# y_max exclusive
# ============================================================

ROI_X_MIN = 520
ROI_X_MAX = 740

ROI_Y_MIN = 145
ROI_Y_MAX = 465


# ============================================================
# Visualization settings
# ============================================================

OVERLAY_ALPHA = 0.45

# Only for visualization.
# Does NOT change the actual point cloud.
VISUALIZATION_DILATION_SIZE = 3


# ============================================================
# Loading
# ============================================================

def load_cloud(path):
    with np.load(path) as data:

        xyz = np.asarray(
            data["xyz"],
            dtype=np.float32,
        )

        rgb = None

        if "rgb" in data:
            rgb = np.asarray(data["rgb"])

    return xyz, rgb


def load_transform(path):
    with np.load(path) as data:

        if "T_helios2realsense" not in data:
            raise KeyError(
                f"{path} does not contain "
                "'T_helios2realsense'.\n"
                f"Available keys: {list(data.keys())}"
            )

        T = np.asarray(
            data["T_helios2realsense"],
            dtype=np.float64,
        )

    if T.shape != (4, 4):
        raise ValueError(
            f"T_helios2realsense must be 4x4, "
            f"got {T.shape}"
        )

    return T


def load_realsense_intrinsics(path):
    with np.load(path) as data:

        if "camera_matrix" not in data:
            raise KeyError(
                f"{path} does not contain "
                "'camera_matrix'."
            )

        camera_matrix = np.asarray(
            data["camera_matrix"],
            dtype=np.float64,
        )

    return camera_matrix


# ============================================================
# Background subtraction
# ============================================================

def subtract_background(
    background_xyz,
    scene_xyz,
    foreground_threshold=0.005,
    max_foreground_height=0.500,
):
    if background_xyz.shape != scene_xyz.shape:
        raise ValueError(
            f"Cloud shapes do not match:\n"
            f"background: {background_xyz.shape}\n"
            f"scene:      {scene_xyz.shape}"
        )

    original_shape = scene_xyz.shape

    background_flat = background_xyz.reshape(-1, 3)
    scene_flat = scene_xyz.reshape(-1, 3)

    valid_background = (
        np.isfinite(background_flat).all(axis=1)
        & (background_flat[:, 2] > 0)
    )

    valid_scene = (
        np.isfinite(scene_flat).all(axis=1)
        & (scene_flat[:, 2] > 0)
    )

    valid = valid_background & valid_scene

    background_z = background_flat[:, 2]
    scene_z = scene_flat[:, 2]

    # Positive means the scene point is closer
    # to the camera than the stored background.
    height = background_z - scene_z

    foreground_mask_flat = (
        valid
        & (height > foreground_threshold)
        & (height < max_foreground_height)
    )

    # Indices into the ORIGINAL scene cloud
    foreground_scene_indices = np.flatnonzero(
        foreground_mask_flat
    )

    foreground_xyz = scene_flat[
        foreground_scene_indices
    ]

    if len(original_shape) == 3:

        foreground_mask = foreground_mask_flat.reshape(
            original_shape[0],
            original_shape[1],
        )

        height_map = height.reshape(
            original_shape[0],
            original_shape[1],
        )

    else:

        foreground_mask = foreground_mask_flat
        height_map = height

    return (
        foreground_xyz,
        foreground_mask,
        height_map,
        foreground_scene_indices,
    )


# ============================================================
# Helios -> RealSense
# ============================================================

def transform_helios_to_realsense(
    helios_xyz,
    T_helios2realsense,
):
    """
    helios_xyz is expected to be in metres.

    Returns:
        xyz_rs:
            Valid points in the RealSense frame.

        source_indices:
            Index into helios_xyz corresponding to xyz_rs.
    """

    xyz = np.asarray(
        helios_xyz,
        dtype=np.float64,
    )

    valid = np.isfinite(xyz).all(axis=1)

    source_indices = np.flatnonzero(valid)

    xyz = xyz[valid]

    ones = np.ones(
        (xyz.shape[0], 1),
        dtype=np.float64,
    )

    xyz_h = np.concatenate(
        [xyz, ones],
        axis=1,
    )

    xyz_rs_h = (
        T_helios2realsense @ xyz_h.T
    ).T

    xyz_rs = xyz_rs_h[:, :3]

    # Remove anything behind RealSense
    valid_z = (
        np.isfinite(xyz_rs).all(axis=1)
        & (xyz_rs[:, 2] > 0)
    )

    xyz_rs = xyz_rs[valid_z]
    source_indices = source_indices[valid_z]

    return xyz_rs, source_indices


# ============================================================
# RealSense projection
# ============================================================

def project_realsense_xyz(
    xyz_rs,
    camera_matrix,
):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = xyz_rs[:, 0]
    y = xyz_rs[:, 1]
    z = xyz_rs[:, 2]

    u = fx * x / z + cx
    v = fy * y / z + cy

    uv = np.stack(
        [u, v],
        axis=1,
    )

    return uv


# ============================================================
# ROI filtering
# ============================================================

def filter_projected_points_to_roi(
    uv,
    xyz_rs,
    source_indices,
    image_shape,
):
    image_height, image_width = image_shape[:2]

    u = np.round(
        uv[:, 0]
    ).astype(np.int32)

    v = np.round(
        uv[:, 1]
    ).astype(np.int32)

    # First make sure point falls inside RGB image
    inside_image = (
        (u >= 0)
        & (u < image_width)
        & (v >= 0)
        & (v < image_height)
    )

    # Then apply your RealSense ROI
    inside_roi = (
        (u >= ROI_X_MIN)
        & (u < ROI_X_MAX)
        & (v >= ROI_Y_MIN)
        & (v < ROI_Y_MAX)
    )

    valid = inside_image & inside_roi

    return (
        u[valid],
        v[valid],
        xyz_rs[valid],
        source_indices[valid],
    )


# ============================================================
# Build projected mask
# ============================================================

def build_projected_mask(
    u,
    v,
    image_shape,
):
    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    mask[v, u] = 255

    return mask


# ============================================================
# RGB overlay
# ============================================================

def create_overlay(
    rgb_image,
    projected_mask,
):
    overlay = rgb_image.copy()

    # Dilation is ONLY for visualization.
    # It makes the projected point cloud easier to see.
    if VISUALIZATION_DILATION_SIZE > 1:

        kernel = np.ones(
            (
                VISUALIZATION_DILATION_SIZE,
                VISUALIZATION_DILATION_SIZE,
            ),
            dtype=np.uint8,
        )

        display_mask = cv2.dilate(
            projected_mask,
            kernel,
            iterations=1,
        )

    else:
        display_mask = projected_mask

    mask_bool = display_mask > 0

    # BGR green
    colour = np.array(
        [0, 255, 0],
        dtype=np.float32,
    )

    original_pixels = overlay[
        mask_bool
    ].astype(np.float32)

    overlay[
        mask_bool
    ] = (
        (1.0 - OVERLAY_ALPHA)
        * original_pixels
        +
        OVERLAY_ALPHA
        * colour
    ).astype(np.uint8)

    # Draw ROI rectangle
    cv2.rectangle(
        overlay,
        (ROI_X_MIN, ROI_Y_MIN),
        (ROI_X_MAX - 1, ROI_Y_MAX - 1),
        (0, 0, 255),
        2,
    )

    return overlay


# ============================================================
# PLY
# ============================================================

def save_ply(
    xyz,
    rgb,
    path,
):
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        xyz.astype(np.float64)
    )

    if (
        rgb is not None
        and len(rgb) == len(xyz)
        and len(rgb) > 0
    ):
        colours = np.asarray(
            rgb,
            dtype=np.float64,
        )

        if colours.max() > 1.0:
            colours /= 255.0

        cloud.colors = (
            o3d.utility.Vector3dVector(
                colours
            )
        )

    o3d.io.write_point_cloud(
        str(path),
        cloud,
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    background_xyz, _ = load_cloud(
        BACKGROUND_PATH
    )

    scene_xyz, scene_rgb = load_cloud(
        SCENE_PATH
    )

    rgb_image = cv2.imread(
        str(RGB_PATH),
        cv2.IMREAD_COLOR,
    )

    if rgb_image is None:
        raise FileNotFoundError(
            f"Could not read RGB image:\n{RGB_PATH}"
        )

    T_helios2realsense = load_transform(
        TRANSFORM_PATH
    )

    camera_matrix = load_realsense_intrinsics(
        REALSENSE_CALIBRATION_PATH
    )

    print()
    print("========================================")
    print("Input")
    print("========================================")

    print(
        "Background shape:",
        background_xyz.shape,
    )

    print(
        "Scene shape:",
        scene_xyz.shape,
    )

    print(
        "RealSense RGB shape:",
        rgb_image.shape,
    )

    print(
        "ROI:",
        f"x={ROI_X_MIN}:{ROI_X_MAX}, "
        f"y={ROI_Y_MIN}:{ROI_Y_MAX}",
    )


    # --------------------------------------------------------
    # Background subtraction
    # --------------------------------------------------------

    (
        foreground_xyz,
        foreground_mask,
        height_map,
        foreground_scene_indices,
    ) = subtract_background(
        background_xyz,
        scene_xyz,
        foreground_threshold=FOREGROUND_THRESHOLD,
        max_foreground_height=MAX_FOREGROUND_HEIGHT,
    )

    print()
    print("========================================")
    print("Background subtraction")
    print("========================================")

    print(
        "Total scene points:",
        scene_xyz.reshape(-1, 3).shape[0],
    )

    print(
        "Foreground points:",
        foreground_xyz.shape[0],
    )


    # --------------------------------------------------------
    # RGB belonging to foreground points
    # --------------------------------------------------------

    foreground_rgb = None

    if scene_rgb is not None:

        rgb_flat = scene_rgb.reshape(-1, 3)

        foreground_rgb = rgb_flat[
            foreground_scene_indices
        ]


    # --------------------------------------------------------
    # Transform Helios foreground -> RealSense
    # --------------------------------------------------------

    (
        xyz_rs,
        foreground_local_indices,
    ) = transform_helios_to_realsense(
        foreground_xyz,
        T_helios2realsense,
    )

    print()
    print("========================================")
    print("Projection")
    print("========================================")

    print(
        "Valid points after Helios -> RealSense:",
        xyz_rs.shape[0],
    )


    # --------------------------------------------------------
    # Project into RealSense RGB
    # --------------------------------------------------------

    uv = project_realsense_xyz(
        xyz_rs,
        camera_matrix,
    )


    # --------------------------------------------------------
    # Apply RealSense ROI
    # --------------------------------------------------------

    (
        u_roi,
        v_roi,
        xyz_rs_roi,
        foreground_local_indices_roi,
    ) = filter_projected_points_to_roi(
        uv,
        xyz_rs,
        foreground_local_indices,
        rgb_image.shape,
    )

    print(
        "Points inside RealSense ROI:",
        len(u_roi),
    )


    # --------------------------------------------------------
    # Recover ORIGINAL Helios foreground XYZ
    #
    # This means the resulting PLY contains Helios points
    # whose projection falls inside your RealSense ROI.
    # --------------------------------------------------------

    foreground_xyz_roi = foreground_xyz[
        foreground_local_indices_roi
    ]

    foreground_rgb_roi = None

    if foreground_rgb is not None:

        foreground_rgb_roi = foreground_rgb[
            foreground_local_indices_roi
        ]


    # --------------------------------------------------------
    # Map ROI points back to original full scene indices
    # --------------------------------------------------------

    scene_indices_roi = foreground_scene_indices[
        foreground_local_indices_roi
    ]


    # --------------------------------------------------------
    # Projected mask
    # --------------------------------------------------------

    projected_mask = build_projected_mask(
        u_roi,
        v_roi,
        rgb_image.shape,
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        rgb_image,
        projected_mask,
    )

    roi_overlay = overlay[
        ROI_Y_MIN:ROI_Y_MAX,
        ROI_X_MIN:ROI_X_MAX,
    ]

    uv_roi = np.stack([u_roi, v_roi], axis=1).astype(np.int32)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,

        # Original Helios coordinates
        xyz=foreground_xyz_roi,

        rgb=foreground_rgb_roi,

        # Same points transformed into RealSense frame
        xyz_realsense=xyz_rs_roi,

        # Full RealSense image coordinates
        uv=uv_roi,

        # Original Helios scene indices
        scene_indices=scene_indices_roi,

        # Full Helios foreground result
        helios_foreground_mask=foreground_mask,
        height=height_map,

        # ROI for reference
        roi=np.array(
            [
                ROI_X_MIN,
                ROI_X_MAX,
                ROI_Y_MIN,
                ROI_Y_MAX,
            ],
            dtype=np.int32,
        ),
    )

    print()
    print("Saved NPZ:")
    print(OUTPUT_PATH)


    # --------------------------------------------------------
    # Save ROI-filtered PLY
    # --------------------------------------------------------

    save_ply(
        foreground_xyz_roi,
        foreground_rgb_roi,
        PLY_OUTPUT_PATH,
    )

    print()
    print("Saved PLY:")
    print(PLY_OUTPUT_PATH)


    # --------------------------------------------------------
    # Save projected mask
    # --------------------------------------------------------

    cv2.imwrite(
        str(PROJECTED_MASK_OUTPUT_PATH),
        projected_mask,
    )

    print()
    print("Saved projected mask:")
    print(PROJECTED_MASK_OUTPUT_PATH)


    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    cv2.imwrite(
        str(OVERLAY_OUTPUT_PATH),
        overlay,
    )

    print()
    print("Saved overlay:")
    print(OVERLAY_OUTPUT_PATH)


    # --------------------------------------------------------
    # Save cropped ROI overlay
    # --------------------------------------------------------

    cv2.imwrite(
        str(ROI_OVERLAY_OUTPUT_PATH),
        roi_overlay,
    )

    print()
    print("Saved ROI overlay:")
    print(ROI_OVERLAY_OUTPUT_PATH)


    # --------------------------------------------------------
    # Show RGB overlay
    # --------------------------------------------------------

    cv2.imshow(
        "Helios Foreground -> RealSense RGB",
        overlay,
    )

    cv2.imshow(
        "ROI",
        roi_overlay,
    )


    # --------------------------------------------------------
    # Show ROI-filtered foreground point cloud
    # --------------------------------------------------------

    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        foreground_xyz_roi.astype(
            np.float64
        )
    )

    if (
        foreground_rgb_roi is not None
        and len(foreground_rgb_roi) > 0
    ):

        colours = foreground_rgb_roi.astype(
            np.float64
        )

        if colours.max() > 1:
            colours /= 255.0

        cloud.colors = (
            o3d.utility.Vector3dVector(
                colours
            )
        )

    print()
    print("Close the Open3D window to continue.")

    o3d.visualization.draw_geometries(
        [cloud]
    )


    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()