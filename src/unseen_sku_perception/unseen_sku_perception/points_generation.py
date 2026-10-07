import numpy as np
import cv2
from dataclasses import dataclass

@dataclass(frozen=True)
class PositivePoint:
    x: int
    y: int
    score: float
    component_id: int

def resolve_workspace_roi(image_shape, roi_config):
    """
    Returns x_min, x_max, y_min, y_max for SAM2 ROI.
    """
    image_height, image_width = image_shape[:2]

    if not roi_config.get("enabled", True):
        return (0, image_width, 0, image_height)

    x_min = int(roi_config["x_min"])
    x_max = int(roi_config["x_max"])
    y_min = int(roi_config["y_min"])
    y_max = int(roi_config["y_max"])

    if not (0 <= x_min < x_max <= image_width and 0 <= y_min < y_max <= image_height):
        raise ValueError(
            "Invalid ROI coordinates. "
            f"ROI=({x_min}, {y_min}) to ({x_max}, {y_max}), "
            f"image size={image_width}x{image_height}."
        )

    return (x_min, x_max, y_min, y_max)

def generate_positive_points(
    core_foreground,
    height,
    valid_depth,
    distance_weight=0.80,
    height_weight=0.20,
    maximum_points_per_component=3,
    minimum_point_distance_px=30
):  
    component_count, component_labels = cv2.connectedComponents(
        core_foreground.astype(np.uint8) * 255,
        connectivity=8
    )

    # Height cleaning
    height_clean = np.where(
        np.isfinite(height),
        height,
        0
    ).astype(np.float32)

    # Apply 5x5 median filter to remove noise and isolated height spikes
    height_smoothed = cv2.medianBlur(height_clean, 5)

    # Store all positive points in a list
    positive_points = []

    for component_id in range(1, component_count):
        component_mask = component_labels == component_id

        # Must be in foreground, have valid depth reading, have real finite height measurement
        candidates = (component_mask & valid_depth & np.isfinite(height))

        # No candidate positive points for prompting
        if not np.any(candidates):
            continue

        # Calculate euclidean distance from foreground boundaries
        distance = cv2.distanceTransform(
            component_mask.astype(np.uint8),
            cv2.DIST_L2,
            5
        )

        max_distance = float(distance.max())
        if max_distance <= 0:
            continue

        # Normalize distance 
        distance_normalized = distance / max_distance
        
        candidate_heights = height_smoothed[candidates]

        low_height = float(np.percentile(candidate_heights, 10))
        high_height = float(np.percentile(candidate_heights, 90))
        height_range = high_height - low_height

        if height_range > 1e-6:
            height_normalized = np.clip(
                (height_smoothed - low_height) / height_range,
                0.0,
                1.0,
            )
        else:
            height_normalized = np.zeros_like(height_smoothed, dtype=np.float32,)

        # Prioritize point safely inside an exposed region
        seed_score = distance_weight * distance_normalized + height_weight * height_normalized

        # Non candidate pixels suppressed to -1.0
        seed_score[~candidates] = -1.0

        # Select multiple spatially separated maxima
        working_score = seed_score.copy()
        for _ in range(maximum_points_per_component):
            prompt_y, prompt_x = np.unravel_index(
                np.argmax(working_score),
                working_score.shape
            )

            score = float(working_score[prompt_y, prompt_x])
            if score < 0.0:
                break

            positive_points.append(
                PositivePoint(
                    x=int(prompt_x),
                    y=int(prompt_y),
                    score=score,
                    component_id=component_id
                )
            )

            # Suppress nearby points
            cv2.circle(working_score,
                       center=(int(prompt_x), int(prompt_y)),
                       radius=int(minimum_point_distance_px),
                       color=-1.0,
                       thickness=-1
                       )
        
    # Process the strongest/exposed points first
    positive_points.sort(key=lambda point: point.score, reverse=True)

    return positive_points

def filter_prompt_components(
    core_foreground,
    valid_depth,
    minimum_component_area=300,
    minimum_component_radius=5.0,
    minimum_valid_depth_ratio=0.50,
):
    """
    Remove foreground components that are not reliable enough
    to generate SAM2 prompts.

    A component must:
      - be large enough
      - contain a reasonably thick interior region
      - contain enough valid depth pixels
    """

    foreground = core_foreground.astype(bool)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=8,
    )

    reliable_foreground = np.zeros_like(foreground, dtype=bool,)

    for component_id in range(1, count):
        component = labels == component_id

        area = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]

        # 1. Reject small components
        if area < minimum_component_area:
            continue

        # 2. Check how much valid depth exists
        valid_pixels = np.count_nonzero(component & valid_depth)
        valid_ratio = valid_pixels / max(area, 1)

        if valid_ratio < minimum_valid_depth_ratio:
            continue

        # 3. Reject very thin / fragmented components
        component_distance = cv2.distanceTransform(
            component.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        maximum_radius = float(component_distance.max())
        if maximum_radius < minimum_component_radius:
            continue

        reliable_foreground[component] = True

    return reliable_foreground

def build_foreground_point_grid(
    core_foreground,
    height,
    valid_depth,
    spacing=24,
    minimum_boundary_distance=8.0,
    anchor_points_per_component=3,
    minimum_component_area=500,
    minimum_component_radius=5.0,
    minimum_valid_depth_ratio=0.50,
):
    """
    Build SAM2 prompts using:
        1. strong depth-derived anchor points
        2. dense foreground grid points
    """
    foreground = filter_prompt_components(
        core_foreground=core_foreground,
        valid_depth=valid_depth,
        minimum_component_area=minimum_component_area,
        minimum_component_radius=minimum_component_radius,
        minimum_valid_depth_ratio=minimum_valid_depth_ratio,
    )
    image_height, image_width = foreground.shape

    # Distance of each foreground pixel from nearest background pixel
    distance = cv2.distanceTransform(
        foreground.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    prompt_points = []
    # 1. Add strong depth-based anchor points FIRST
    anchor_points = generate_positive_points(
        core_foreground=foreground,
        height=height,
        valid_depth=valid_depth,
        distance_weight=0.70,
        height_weight=0.30,
        maximum_points_per_component=anchor_points_per_component,
        minimum_point_distance_px=30,
    )

    for point in anchor_points:
        prompt_points.append((point.x, point.y))

    # 2. Add regular foreground grid
    # Divide the image into spacing x spacing cells
    for y0 in range(0, image_height, spacing):
        for x0 in range(0, image_width, spacing):
            y1 = min(y0 + spacing, image_height)
            x1 = min(x0 + spacing, image_width)

            # Extract this cell.
            foreground_cell = foreground[y0:y1, x0:x1]

            # Apply distance transform on the cell
            distance_cell = distance[y0:y1, x0:x1]

            # No foreground in this cell.
            if not np.any(foreground_cell):
                continue

            # Pixels must be far enough from boundary and have valid depth
            valid_depth_cell = valid_depth[y0:y1, x0:x1]
            height_cell = height[ y0:y1, x0:x1]

            candidates = (
                foreground_cell
                & valid_depth_cell
                & np.isfinite(height_cell)
                & (distance_cell >= minimum_boundary_distance)
            )

            if not np.any(candidates):
                continue

            # Remove invalid locations
            candidate_distance = np.where(
                candidates,
                distance_cell,
                -1.0,
            )

            # Find highest boundary-distance location inside this cell.
            local_y, local_x = np.unravel_index(
                np.argmax(candidate_distance),
                candidate_distance.shape,
            )

            # Convert from cell coordinates to full-image coordinates.
            x = x0 + local_x
            y = y0 + local_y

            prompt_points.append((int(x), int(y)))

    # Remove identical coordinates while retaining order.
    prompt_points = list(dict.fromkeys(prompt_points))

    if not prompt_points:
        raise RuntimeError("No foreground prompt points generated")

    # SAM2 requires normalized (x, y) coordinates.
    normalized_points = np.asarray(
        [
            [
                (x + 0.5) / image_width,
                (y + 0.5) / image_height,
            ]
            for x, y in prompt_points
        ],
        dtype=np.float32,
    )

    # One point grid because crop_n_layers will be zero.
    point_grids = [normalized_points]

    return prompt_points, point_grids

def build_uniform_rgb_point_grid(
    image_shape,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    spacing=24,
    margin=10,
):
    image_height, image_width = image_shape[:2]

    # Use the whole image if ROI is not provided.
    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = image_width

    if y_min is None:
        y_min = 0

    if y_max is None:
        y_max = image_height

    prompt_points = []
    for y in range(y_min + margin, y_max - margin, spacing):
        for x in range(x_min + margin, x_max - margin, spacing,):
            prompt_points.append((x, y))

    if not prompt_points:
        raise RuntimeError("No RGB SAM2 prompt points generated.")
    
    normalized_points = np.asarray(
        [
            [(x + 0.5) / image_width, 
             (y + 0.5) / image_height] for x, y in prompt_points
        ],
        dtype=np.float32,
    )

    point_grids = [normalized_points]

    return prompt_points, point_grids