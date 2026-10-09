import numpy as np
import cv2

from pathlib import Path
import argparse

from .project_to_image import load_helios_xyz

def associate_projected_points_with_masks(
    object_masks,
    helios_xyz_flat,
    rs_xyz,
    u,
    v,
    source_indices,
    erosion_size=5,
):
    """
    Associate projected Helios2 points with SAM2 masks.

    Args:
        object_masks:
            Boolean array (K, H, W).

        helios_xyz_flat:
            Original flattened Helios cloud (N, 3).

        rs_xyz:
            Projected Helios points expressed in RealSense frame (M, 3).

        u, v:
            RealSense RGB pixel coordinates for each projected point.

        source_indices:
            Indices mapping each projected point back to helios_xyz_flat.

        erosion_size:
            Optional SAM-mask erosion to avoid uncertain boundaries.

    Returns:
        points_by_object:
            {
                object_id: {
                    "helios_xyz": ...,
                    "realsense_xyz": ...,
                    "uv": ...
                }
            }
    """

    object_masks = np.asarray(object_masks, dtype=bool,)
    helios_xyz_flat = np.asarray(helios_xyz_flat, dtype=np.float64,).reshape(-1, 3)

    points_by_object = {}
    if erosion_size > 1:
        kernel = np.ones(
            (erosion_size, erosion_size),
            dtype=np.uint8,
        )
    else:
        kernel = None

    for mask_index, object_mask in enumerate(object_masks):
        object_id = mask_index + 1

        if kernel is not None:
            safe_mask = cv2.erode(
                object_mask.astype(np.uint8),
                kernel,
                iterations=1,
            ).astype(bool)
        else:
            safe_mask = object_mask

        # Which projected Helios points land inside this SAM mask?
        inside = safe_mask[v, u]

        selected_source_indices = source_indices[inside]
        selected_helios_xyz = helios_xyz_flat[selected_source_indices]

        selected_rs_xyz = rs_xyz[inside]
        selected_uv = np.column_stack((u[inside], v[inside]))

        points_by_object[object_id] = {
            "helios_xyz": selected_helios_xyz,
            "realsense_xyz": selected_rs_xyz,
            "uv": selected_uv,
            "source_indices": selected_source_indices,
        }

    return points_by_object

def load_projected_points(path):
    """
    Load output created by project_to_image.py.
    """

    path = Path(path).expanduser()

    with np.load(path) as data:
        required_keys = (
            "xyz_realsense",
            "u",
            "v",
            "source_indices",
        )

        for key in required_keys:
            if key not in data.files:
                raise KeyError(
                    f"{path} does not contain '{key}'. "
                    f"Keys: {data.files}"
                )

        rs_xyz = np.asarray(data["xyz_realsense"], dtype=np.float64)
        u = np.asarray(data["u"], dtype=np.int32)
        v = np.asarray(data["v"], dtype=np.int32)
        source_indices = np.asarray(data["source_indices"], dtype=np.int64)

    return (rs_xyz, u, v, source_indices)


def load_masks(mask_directory, scene_index):
    """
    Load final SAM2 object masks for one scene.

    Expected filenames:
        sam2_mask_0035_01.png
        sam2_mask_0035_02.png
        ...
    """

    mask_directory = Path(mask_directory).expanduser()
    mask_pattern = f"sam2_mask_{scene_index:04d}_*.png"
    mask_paths = sorted(mask_directory.glob(mask_pattern))

    if not mask_paths:
        raise FileNotFoundError(
            f"No masks found in "
            f"{mask_directory} "
            f"using pattern '{mask_pattern}'"
        )

    masks = []
    for mask_path in mask_paths:
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask_image is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        mask = mask_image > 127

        masks.append(mask)

    first_shape = masks[0].shape
    for mask in masks:
        if mask.shape != first_shape:
            raise ValueError(
                "All SAM2 masks must have the same dimensions."
            )

    object_masks = np.stack(masks, axis=0,)

    return object_masks, mask_paths

def convert_masks_to_full_image(
    object_masks,
    image_shape,
    roi_x_min,
    roi_x_max,
    roi_y_min,
    roi_y_max
):
    """
    Convert ROI-cropped SAM2 masks back to the
    full RealSense RGB coordinate system.

    If masks are already full-resolution,
    they are returned unchanged.
    """

    image_height, image_width = (image_shape[:2])

    mask_height = object_masks.shape[1]
    mask_width = object_masks.shape[2]

    # Already full-resolution.
    if (mask_height == image_height and mask_width == image_width):
        print("SAM2 masks are already full-resolution.")

        return object_masks

    roi_width = roi_x_max - roi_x_min
    roi_height = roi_y_max - roi_y_min

    mask_height = object_masks.shape[1]
    mask_width = object_masks.shape[2]

    if (
        roi_x_min < 0
        or roi_y_min < 0
        or roi_x_max > image_width
        or roi_y_max > image_height
    ):
        raise ValueError(
            "SAM2 ROI does not fit inside "
            "the RealSense image.\n"
            f"Image: {image_width}x{image_height}\n"
            f"Mask:  {mask_width}x{mask_height}\n"
            f"ROI: x={roi_x_min}:{roi_x_max}, "
            f"y={roi_y_min}:{roi_y_max}"
        )

    full_masks = np.zeros(
        (len(object_masks), image_height, image_width,),
        dtype=bool,
    )

    full_masks[:, roi_y_min:roi_y_max, roi_x_min:roi_x_max,] = object_masks

    print("Expanded cropped SAM2 masks to full RGB coordinates.")

    print(
        f"ROI: "
        f"x={roi_x_min}:{roi_x_max}, "
        f"y={roi_y_min}:{roi_y_max}"
    )

    return full_masks

def save_fused_objects(points_by_object, output_directory):
    """
    Save one NPZ per detected object.
    """

    output_directory = Path(output_directory).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True,)

    print()
    print("Fused object clouds")
    print("-------------------")

    for (object_id, object_data) in points_by_object.items():

        helios_xyz = object_data["helios_xyz"]
        rs_xyz = object_data["realsense_xyz"]
        uv = object_data["uv"]
        source_indices = object_data["source_indices"]

        output_path = output_directory / f"object_{object_id:02d}.npz"

        np.savez(
            output_path,
            xyz=helios_xyz.astype(np.float32),
            xyz_realsense=rs_xyz.astype(np.float32),
            uv=uv.astype(np.int32),
            source_indices=source_indices,
            xyz_unit=np.array("m"),
            object_id=np.array(object_id),
        )

        print(
            f"Object {object_id}: "
            f"{len(helios_xyz)} points "
            f"-> {output_path}"
        )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Associate projected Helios2 points with SAM2 object masks."
        )
    )

    parser.add_argument(
        "--scene-index",
        type=int,
        required=True,
        help=(
            "Scene number used in SAM2 mask names. "
            "For sam2_mask_0035_01.png, use 35."
        ),
    )

    parser.add_argument(
        "--erosion-size",
        type=int,
        default=1,
        help=(
            "Mask erosion kernel size. 1 disables erosion."
        ),
    )

    args = parser.parse_args()
    # --------------------------------------------------
    # Resolve all scene paths from one index
    # --------------------------------------------------

    scene_index = args.scene_index
    index_string = f"{scene_index:04d}"

    # Original Helios point cloud
    helios_dir = Path("/home/support/unseen_sku_ws/data/helios2/pointclouds")
    helios_path = helios_dir / f"scene_{index_string}.npz"

    # RealSense RGB
    rgb_dir = Path("/home/support/unseen_sku_ws/data/realsense/scene/images")
    rgb_path = rgb_dir / f"image_{index_string}.png"

    # Fusion output directory
    scene_dir = Path("/home/support/unseen_sku_ws/outputs") / f"scene_{index_string}"
    output_dir = scene_dir / "mask_fusion"

    # Projected Helios -> RealSense .npz file
    projected_path = Path(output_dir / f"helios_projected_to_realsense_rgb_{index_string}.npz")

    # Find SAM2 mask directory automatically
    mask_dir = scene_dir / "sam2"

    required_files = {
        "Helios": helios_path,
        "Projected": projected_path,
        "RGB": rgb_path
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found:\n"
                f"{path}"
            )

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"SAM2 mask directory not found:\n"
            f"{mask_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load original RGB
    # --------------------------------------------------
    rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR,)
    if rgb_image is None:
        raise FileNotFoundError(f"Could not read RGB image: {str(rgb_path)}")

    print(
        f"RGB dimensions: "
        f"{rgb_image.shape[1]} x "
        f"{rgb_image.shape[0]}"
    )

    # --------------------------------------------------
    # Load original Helios XYZ
    # --------------------------------------------------
    helios_xyz_flat = load_helios_xyz(helios_path)
    print(f"Original Helios points: {len(helios_xyz_flat)}")

    # --------------------------------------------------
    # Load projected correspondence
    # --------------------------------------------------
    rs_xyz, u, v, source_indices = load_projected_points(projected_path)

    print(f"Projected Helios points: {len(rs_xyz)}")

    # --------------------------------------------------
    # Load final SAM2 masks
    # --------------------------------------------------
    object_masks, mask_paths = load_masks(
        mask_directory=mask_dir,
        scene_index=scene_index,
    )

    print(f"SAM2 masks loaded: {len(object_masks)}")

    print(
        f"SAM2 mask dimensions: "
        f"{object_masks.shape[2]} x "
        f"{object_masks.shape[1]}"
    )

    for index, mask_path in enumerate(mask_paths, start=1,):
        print(
            f"  Object {index}: "
            f"{mask_path.name}"
        )

    # --------------------------------------------------
    # Convert cropped masks to full RGB coordinates
    # --------------------------------------------------
    object_masks = (
        convert_masks_to_full_image(
            object_masks=object_masks,
            image_shape=rgb_image.shape,
            roi_x_min=520,
            roi_x_max=710,
            roi_y_min=185,
            roi_y_max=465
        )
    )

    # Safety check before indexing mask[v, u]
    image_height, image_width = (rgb_image.shape[:2])

    if (
        np.any(u < 0)
        or np.any(u >= image_width)
        or np.any(v < 0)
        or np.any(v >= image_height)
    ):
        raise ValueError(
            "Projected UV coordinates contain pixels outside the RGB image."
        )

    # --------------------------------------------------
    # Fuse masks with Helios points
    # --------------------------------------------------
    points_by_object = (
        associate_projected_points_with_masks(
            object_masks=object_masks,
            helios_xyz_flat=helios_xyz_flat,
            rs_xyz=rs_xyz,
            u=u,
            v=v,
            source_indices=source_indices,
            erosion_size=args.erosion_size,
        )
    )

    # --------------------------------------------------
    # Save object clouds
    # --------------------------------------------------
    save_fused_objects(
        points_by_object=points_by_object,
        output_directory=output_dir,
    )

    print()
    print(f"Fusion complete. Saved to {output_dir}")

if __name__ == "__main__":
    main()