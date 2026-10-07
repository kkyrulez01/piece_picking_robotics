import numpy as np
import cv2

def calculate_height_map(background_depth, scene_depth,):
    """
    Calculate physical height above the empty workbench.

    height = background_depth - scene_depth

    Positive value:
        scene is closer to camera than empty workbench
        -> potentially a physical object

    Approximately zero:
        scene lies at workbench depth
        -> likely table / printed texture / RGB shadow
    """

    background_depth = np.asarray(background_depth, dtype=np.float32,)
    scene_depth = np.asarray(scene_depth, dtype=np.float32,)

    if background_depth.shape != scene_depth.shape:
        raise ValueError(
            "Background and scene depth dimensions differ: "
            f"{background_depth.shape} vs "
            f"{scene_depth.shape}"
        )

    valid_depth = (
        np.isfinite(background_depth)
        & np.isfinite(scene_depth)
        & (background_depth > 0.0)
        & (scene_depth > 0.0)
    )

    height = np.full(
        scene_depth.shape,
        np.nan,
        dtype=np.float32,
    )

    height[valid_depth] = background_depth[valid_depth]- scene_depth[valid_depth]

    return height, valid_depth

def depth_support_statistics(
    mask,
    height,
    valid_depth,
    object_height_threshold=0.005,
    erosion_size=5,
):
    """
    Measure whether a SAM2 mask corresponds to something physically
    above the empty workbench.

    We erode the RGB mask slightly before evaluating depth. This avoids
    RGB-depth alignment errors around object boundaries.
    """

    mask = np.asarray(mask, dtype=bool,)

    # Use only interior of SAM2 mask for depth validation
    if erosion_size > 1:
        kernel = np.ones(
            (erosion_size, erosion_size),
            dtype=np.uint8,
        )

        interior_mask = cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)

        # Very small mask could disappear after erosion.
        if not np.any(interior_mask):
            interior_mask = mask

    else:
        interior_mask = mask

    mask_area = np.count_nonzero(interior_mask)
    if mask_area == 0:
        return {
            "valid_ratio": 0.0,
            "elevated_ratio": 0.0,
            "median_height": np.nan,
            "p75_height": np.nan,
            "p90_height": np.nan,
            "valid_pixels": 0,
        }

    # Valid depth inside mask
    valid_mask = (
        interior_mask
        & valid_depth
        & np.isfinite(height)
    )

    valid_count = np.count_nonzero(valid_mask)
    valid_ratio = valid_count / mask_area

    if valid_count == 0:
        return {
            "valid_ratio": valid_ratio,
            "elevated_ratio": 0.0,
            "median_height": np.nan,
            "p75_height": np.nan,
            "p90_height": np.nan,
            "valid_pixels": 0,
        }

    mask_heights = height[valid_mask]

    # How much of this mask is actually above the workbench?
    elevated_pixels = (mask_heights >= object_height_threshold)
    elevated_ratio = (np.count_nonzero(elevated_pixels) / valid_count)

    median_height = float(np.median(mask_heights))

    p75_height = float(np.percentile(mask_heights, 75,))
    p90_height = float(np.percentile(mask_heights, 90,))

    return {
        "valid_ratio": valid_ratio,
        "elevated_ratio": elevated_ratio,
        "median_height": median_height,
        "p75_height": p75_height,
        "p90_height": p90_height,
        "valid_pixels": valid_count,
    }

def filter_masks_by_depth(
    results,
    height,
    valid_depth,
    object_height_threshold=0.005,
    minimum_elevated_ratio=0.30,
    minimum_valid_depth_ratio=0.50,
    erosion_size=5,
    minimum_prompt_elevated_ratio=0.50,
    prompt_radius=3,
    minimum_prompt_valid_pixels=5
):
    """
    Keep SAM2 masks only when enough of the mask is physically above
    the workbench.

    A visual shadow normally has approximately the same depth as the
    empty workbench, so its elevated_ratio should be low.
    """

    accepted = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        stats = depth_support_statistics(
            mask=result.mask,
            height=height,
            valid_depth=valid_depth,
            object_height_threshold=object_height_threshold,
            erosion_size=erosion_size,
        )

        prompt_stats = prompt_depth_statistics(
            result=result,
            height=height,
            valid_depth=valid_depth,
            object_height_threshold=object_height_threshold,
            radius=prompt_radius,
        )

        prompt_valid_pixels = prompt_stats["valid_pixels"]
        prompt_median_height = prompt_stats["median_height"]
        prompt_elevated_ratio = prompt_stats["elevated_ratio"]

        valid_ratio = stats["valid_ratio"]
        elevated_ratio = stats["elevated_ratio"]
        median_height = stats["median_height"]
        p75_height = stats["p75_height"]
        p90_height = stats["p90_height"]

        print(
            f"Depth check mask {index}: "
            f"valid={valid_ratio:.3f}, "
            f"elevated={elevated_ratio:.3f}, "
            f"median={median_height:.4f} m, "
            f"p75={p75_height:.4f} m, "
            f"p90={p90_height:.4f} m"
        )

        # Not enough usable depth
        if (valid_ratio < minimum_valid_depth_ratio):
            print("  -> rejected: insufficient valid depth")
            continue

        # Prompt itself must have physical depth support.
        if (
            prompt_valid_pixels >= minimum_prompt_valid_pixels
            and
            prompt_elevated_ratio < minimum_prompt_elevated_ratio
        ):
            print("  -> rejected: SAM prompt is on workbench/shadow")
            continue

        # Mostly still at workbench height
        if (elevated_ratio < minimum_elevated_ratio):
            print("  -> rejected: likely shadow/background")
            continue

        print("  -> accepted: physical depth support")
        accepted.append(result)

    return accepted

def prompt_depth_statistics(
    result,
    height,
    valid_depth,
    object_height_threshold=0.005,
    radius=3,
):
    """
    Check physical depth support around the SAM2 generating prompt.

    radius=3 gives a 7x7 neighbourhood.
    """

    x = int(result.point.x)
    y = int(result.point.y)

    image_height, image_width = height.shape

    x0 = max(0, x - radius)
    x1 = min(image_width, x + radius + 1)

    y0 = max(0, y - radius)
    y1 = min(image_height, y + radius + 1)

    local_height = height[
        y0:y1,
        x0:x1
    ]

    local_valid = valid_depth[
        y0:y1,
        x0:x1
    ]

    valid = (
        local_valid
        & np.isfinite(local_height)
    )

    valid_count = np.count_nonzero(valid)

    if valid_count == 0:
        return {
            "valid_pixels": 0,
            "median_height": np.nan,
            "elevated_ratio": 0.0,
        }

    heights = local_height[valid]

    median_height = float(
        np.median(heights)
    )

    elevated_ratio = (
        np.count_nonzero(
            heights >= object_height_threshold
        )
        / valid_count
    )

    return {
        "valid_pixels": valid_count,
        "median_height": median_height,
        "elevated_ratio": elevated_ratio,
    }
