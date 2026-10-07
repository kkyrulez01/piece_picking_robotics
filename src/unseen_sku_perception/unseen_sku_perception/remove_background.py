#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

@dataclass(frozen=True)
class ForegroundResult:
    """Intermediate results from empty-tote depth subtraction."""

    height: np.ndarray
    valid_depth: np.ndarray
    raw_foreground: np.ndarray
    core_foreground: np.ndarray
    kept_components: int

def load_depth(path):
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Depth file not found: {path}")

    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            raise RuntimeError(f"Could not read depth: {path}")

    return depth.astype(np.float32)


def extract_core_foreground(
    background,
    scene,
    threshold,
    minimum_area=150,
    close_kernel_size=3,
    close_iterations=1,
    open_kernel_size=3,
    open_iterations=1,
    roi_mask=None
):
    if background.shape != scene.shape:
        raise ValueError(
            "Depth dimensions do not match: "
            f"background={background.shape}, scene={scene.shape}"
        )

    valid_background = np.isfinite(background) & (background > 0)
    valid_scene = np.isfinite(scene) & (scene > 0)
    valid = valid_background & valid_scene

    # For an overhead camera, objects are closer than the empty tote.
    height = background - scene
    raw_foreground = valid & (height > threshold)

    if roi_mask is not None:
        raw_foreground &= roi_mask
        valid &= roi_mask

    # Convert Boolean mask to an OpenCV image.
    raw_foreground_mask = raw_foreground.astype(np.uint8) * 255

    # Remove isolated depth noise.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_kernel_size, open_kernel_size)
    )

    raw_foreground_mask = cv2.morphologyEx(
        raw_foreground_mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=open_iterations
    )
    
    # Fill small internal holes.
    # close_kernel = cv2.getStructuringElement(
    #     cv2.MORPH_ELLIPSE,
    #     (close_kernel_size, close_kernel_size)
    # )
    
    # raw_foreground_mask = cv2.morphologyEx(
    #     raw_foreground_mask,
    #     cv2.MORPH_CLOSE,
    #     close_kernel,
    #     iterations=close_iterations
    # )

    # Remove very small components.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        raw_foreground_mask,
        connectivity=8,
    )
    
    core_foreground = np.zeros(background.shape, dtype=bool)
    kept_components = 0

    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area < minimum_area:
            continue

        core_foreground[labels == component_id] = True
        kept_components += 1

    return ForegroundResult(
        height=height,
        valid_depth=valid,
        raw_foreground=raw_foreground_mask,
        core_foreground=core_foreground,
        kept_components=kept_components
    )

def create_support_foreground(
    core_foreground,
    height,
    valid_depth,
    support_height_threshold=0.003,
    dilation_size=15,
):
    # Include valid pixels slightly above the background.
    low_threshold_foreground = (
        valid_depth
        & np.isfinite(height)
        & (height > support_height_threshold)
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_size, dilation_size),
    )

    padded_core = cv2.dilate(
        core_foreground.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)

    support_foreground = (
        low_threshold_foreground
        | padded_core
    )

    return support_foreground