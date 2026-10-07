from pathlib import Path
import numpy as np
import argparse

from .pointcloud_utils import extract_foreground_cloud

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract an XYZ foreground cloud using an empty-background "
            "and populated-scene point cloud.")
    )

    parser.add_argument(
        "--background",
        required=True,
        help="Path to background .npz point cloud."
    )

    parser.add_argument(
        "--scene",
        required=True,
        help="Path to target scene .npz point cloud."
    )

    parser.add_argument(
        "--output-directory",
        default="/home/support/unseen_sku_ws/data/foreground_pointclouds",
        help="Directory in which to save the foreground .npy file.",
    )

    parser.add_argument(
        "--minimum_height",
        type=float,
        default=0.010,
        help="Minimum foreground height in metres.",
    )

    parser.add_argument(
        "--maximum-height",
        type=float,
        default=0.600,
        help="Maximum foreground height in metres.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing foreground file to be replaced.",
    )

    return parser.parse_args()

def create_output_path(scene_path, output_directory):
    """
    Derive the foreground filename from the scene filename.

    Examples:
        scene_0001.npz -> foreground_0001.npy
        test_box.npz   -> test_box_foreground.npy
    """
    scene_stem = scene_path.stem

    if scene_stem.startswith("scene_"):
        scene_suffix = scene_stem[len("scene_"):]
        output_name = f"foreground_{scene_suffix}.npy"
    else:
        output_name = f"{scene_stem}_foreground.npy"

    return output_directory / output_name

def load_capture(capture_path, require_rgb=False):
    capture_path = Path(capture_path).expanduser().resolve()

    if not capture_path.exists():
        raise FileNotFoundError(
            f"Point-cloud capture not found: {capture_path}"
        )

    if capture_path.suffix.lower() != ".npz":
        raise ValueError(
            f"Expected an .npz capture, received: {capture_path}"
        )

    with np.load(capture_path) as capture_data:
        if "xyz" not in capture_data.files:
            raise KeyError(
                f"Capture does not contain an 'xyz' array: {capture_path}"
            )

        xyz = np.asarray(
            capture_data["xyz"],
            dtype=np.float32,
        )

        rgb = None

        if "rgb" in capture_data.files:
            rgb = np.asarray(
                capture_data["rgb"],
                dtype=np.uint8,
            )
        elif require_rgb:
            raise KeyError(
                f"Scene capture does not contain an 'rgb' array: "
                f"{capture_path}"
            )

    if xyz.ndim != 3 or xyz.shape[2] < 3:
        raise ValueError(
            f"Expected organized XYZ with shape (H, W, 3). "
            f"Received {xyz.shape} from {capture_path}."
        )

    return xyz[..., :3], rgb

def main():
    args = parse_arguments()

    background_path = Path(args.background).expanduser().resolve()
    scene_path = Path(args.scene).expanduser().resolve()

    output_directory = Path(args.output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True,)
    output_path = create_output_path(
        scene_path=scene_path,
        output_directory=output_directory,
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    background_xyz, _ = load_capture(background_path, require_rgb=False)
    scene_xyz, scene_rgb = load_capture(scene_path, require_rgb=False)

    if background_xyz.shape != scene_xyz.shape:
        raise ValueError(
            "Background and scene cloud shapes differ: "
            f"{background_xyz.shape} and {scene_xyz.shape}."
        )

    # Set ROI
    image_height, image_width = background_xyz.shape[:2]
    roi_mask = np.zeros((image_height, image_width), dtype=bool)

    # Replace these with the measured tote boundaries.
    roi_x_min = 530
    roi_x_max = 1350
    roi_y_min = 0
    roi_y_max = 1000

    roi_mask[roi_y_min:roi_y_max,roi_x_min:roi_x_max] = True

    result = extract_foreground_cloud(
        background_xyz=background_xyz,
        scene_xyz=scene_xyz,
        scene_rgb=scene_rgb,
        roi_mask=roi_mask,
        minimum_height=args.minimum_height,
        maximum_height=args.maximum_height,
    )

    foreground_xyz = result["xyz"]
    foreground_uv = result["uv"]

    if foreground_xyz.size == 0:
        raise RuntimeError(
            "Foreground extraction produced no points. "
            "Check the background capture, ROI and height thresholds."
        )

    scene_stem = scene_path.stem
    if scene_stem.startswith("scene_"):
        scene_suffix = scene_stem[len("scene_"):]
        output_name = f"foreground_{scene_suffix}.npz"
    else:
        output_name = f"{scene_stem}_foreground.npz"

    output_path = output_directory / output_name
    save_data = {
        "xyz": foreground_xyz.astype(np.float32),
        "uv": foreground_uv.astype(np.uint32),
        "mask": result["mask"].astype(bool),
        "image_height": np.int32(scene_xyz.shape[0]),
        "image_width": np.int32(scene_xyz.shape[1]),
        "coordinate_unit": np.asarray("m"),
    }

    # Save foreground RGB only when it exists
    if result["rgb"] is not None:
        save_data["rgb"] = result["rgb"].astype(np.uint8)

    np.savez_compressed(output_path, **save_data)

    print(f"Saved: {output_path}")
    print(f"Background:       {background_path}")
    print(f"Scene:            {scene_path}")
    print(f"Foreground shape: {foreground_xyz.shape}")
    print("Coordinate unit:  metres")


if __name__ == "__main__":
    main()