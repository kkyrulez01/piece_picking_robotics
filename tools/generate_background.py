#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np

# Folder containing your 30 empty-background depth images
BACKGROUND_DIR = Path("/home/support/unseen_sku_ws/data/realsense/background/depth_images")

# Output median background
OUTPUT_PATH = Path("/home/support/unseen_sku_ws/data/realsense/background/depth_images/background.tiff")

def main():
    depth_paths = sorted(BACKGROUND_DIR.glob("*.tiff"))

    if not depth_paths:
        raise RuntimeError(f"No TIFF files found in {BACKGROUND_DIR}")

    print(f"Found {len(depth_paths)} background depth images.")

    depth_images = []
    expected_shape = None

    for path in depth_paths:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED,)

        if depth is None:
            print(f"WARNING: Could not read {path}")
            continue

        depth = depth.astype(np.float32)

        if expected_shape is None:
            expected_shape = depth.shape

        if depth.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: {path} has {depth.shape}, "
                f"expected {expected_shape}"
            )

        # Treat invalid RealSense depth as missing data
        depth[(depth <= 0) | (depth > 3.0)] = np.nan

        depth_images.append(depth)

    if not depth_images:
        raise RuntimeError("No valid depth images loaded.")

    print(f"Computing median from {len(depth_images)} images...")

    # Shape becomes: (N, H, W)
    depth_stack = np.stack(depth_images, axis=0)

    # Pixel-wise median, ignoring invalid measurements
    median_background = np.nanmedian(depth_stack, axis=0,)

    # If a pixel was invalid in every single image,
    # nanmedian will still produce NaN.
    invalid_pixels = ~np.isfinite(median_background)

    print(f"Pixels invalid in all frames: {np.sum(invalid_pixels)}")

    # Keep them as 0 so your normal invalid-depth handling can detect them
    median_background[invalid_pixels] = 0.0
    median_background = median_background.astype(np.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True,)

    success = cv2.imwrite(str(OUTPUT_PATH), median_background,)

    if not success:
        raise RuntimeError(f"Failed to save {OUTPUT_PATH}")

    valid = median_background > 0

    print()
    print(f"Saved median background:")
    print(f"  {OUTPUT_PATH}")
    print(f"Shape: {median_background.shape}")
    print(f"dtype: {median_background.dtype}")

    if np.any(valid):
        print(
            f"Depth range: "
            f"{median_background[valid].min():.4f} -> "
            f"{median_background[valid].max():.4f}"
        )
        print(
            f"Median depth: "
            f"{np.median(median_background[valid]):.4f}"
        )

if __name__ == "__main__":
    main()