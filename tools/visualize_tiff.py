#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_depth(path):
    path = Path(path).expanduser()
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED,)

    if depth is None:
        raise RuntimeError(f"Could not read depth image: {path}")

    return depth.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tiff",
        help="Path to depth TIFF",
    )

    args = parser.parse_args()

    depth = load_depth(args.tiff)
    print("Shape:", depth.shape)
    print("dtype:", depth.dtype)

    valid_mask = (np.isfinite(depth) & (depth > 0))
    valid_depth = depth[valid_mask]

    if valid_depth.size == 0:
        raise RuntimeError("No valid depth pixels found.")

    print("Valid pixels:", valid_depth.size)
    print("Minimum depth:", np.min(valid_depth))
    print("Median depth:", np.median(valid_depth))
    print("Maximum depth:", np.max(valid_depth))

    # Percentiles prevent a few extreme depth values
    # from destroying visualization contrast.
    minimum_display_depth = np.percentile(valid_depth, 2,)
    maximum_display_depth = np.percentile(valid_depth, 98,)

    print(
        "Display range:", minimum_display_depth,
        "to", maximum_display_depth,
    )

    display_depth = depth.copy()

    # Invalid pixels become NaN so matplotlib shows
    # them separately from actual depths.
    display_depth[~valid_mask] = np.nan

    plt.figure(figsize=(12, 7))

    image = plt.imshow(
        display_depth,
        cmap="turbo",
        vmin=minimum_display_depth,
        vmax=maximum_display_depth,
    )

    plt.colorbar(image, label="Depth")

    plt.title(Path(args.tiff).name)

    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()