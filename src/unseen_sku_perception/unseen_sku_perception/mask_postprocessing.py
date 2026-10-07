import numpy as np
import cv2

from .sam2_segmenter import SamMaskResult
from .points_generation import PositivePoint

def suppress_contained_masks(
    results,
    containment_threshold=0.90,
    maximum_area_ratio=0.50
):
    """
    Remove small masks that are substantially contained inside a
    larger mask. Larger mask is retained, masks are not merged.
    """

    if not results:
        return []

    # Process largest masks first so potential parent masks
    # are added before their smaller contained masks.
    ordered_results = sorted(results, 
                            key=lambda result: np.count_nonzero(result.mask),
                            reverse=True)

    kept_results = []

    for candidate in ordered_results:
        candidate_mask = np.asarray(candidate.mask, dtype=bool)
        candidate_area = np.count_nonzero(candidate_mask)

        if candidate_area == 0:
            continue

        contained = False

        for parent in kept_results:
            parent_mask = np.asarray(parent.mask, dtype=bool)
            parent_area = np.count_nonzero(parent_mask)

            if parent_area == 0:
                continue

            intersection = np.count_nonzero(candidate_mask & parent_mask)
            containment = intersection / candidate_area
            area_ratio = candidate_area / parent_area

            if containment >= containment_threshold and area_ratio <= maximum_area_ratio:
                contained = True

                break

        if not contained:
            kept_results.append(candidate)

    return kept_results

def suppress_union_masks_by_depth(
    results,
    height,
    valid_depth,
    containment_threshold=0.90,
    minimum_child_area_ratio=0.15,
    maximum_child_iou=0.30,
    minimum_height_difference=0.010,
    minimum_valid_pixels=20,
):
    """
    Remove a large SAM2 mask if it appears to combine multiple
    separate physical objects at different heights.

    Smaller instance masks are preserved.
    """

    if not results:
        return []

    masks = [np.asarray(result.mask, dtype=bool) for result in results]
    areas = [np.count_nonzero(mask) for mask in masks]

    remove = [False] * len(results)

    for parent_index, parent_mask in enumerate(masks):
        parent_area = areas[parent_index]
        if parent_area == 0:
            continue

        children = []
        for child_index, child_mask in enumerate(masks):
            if child_index == parent_index:
                continue

            child_area = areas[child_index]
            if child_area == 0:
                continue

            # Must genuinely be smaller than parent.
            if child_area >= parent_area:
                continue

            # Ignore tiny SAM sub-parts such as labels/text.
            child_area_ratio = child_area / parent_area

            if child_area_ratio < minimum_child_area_ratio:
                continue

            intersection = np.count_nonzero(
                child_mask & parent_mask
            )

            containment = (
                intersection / child_area
            )

            if containment < containment_threshold:
                continue

            valid = (
                child_mask
                & valid_depth
                & np.isfinite(height)
            )

            values = height[valid]

            if values.size < minimum_valid_pixels:
                continue

            median_height = float(
                np.median(values)
            )

            children.append(
                (
                    child_index,
                    median_height,
                )
            )

        # Need at least two plausible child objects.
        if len(children) < 2:
            continue

        # Compare every pair of children.
        union_detected = False

        for i in range(len(children)):
            for j in range(i + 1, len(children)):

                child_a_index, height_a = children[i]
                child_b_index, height_b = children[j]

                mask_a = masks[child_a_index]
                mask_b = masks[child_b_index]

                intersection = np.count_nonzero(
                    mask_a & mask_b
                )

                union = np.count_nonzero(
                    mask_a | mask_b
                )

                child_iou = (
                    intersection / max(union, 1)
                )

                # Separate physical object masks should not
                # substantially duplicate one another.
                if child_iou > maximum_child_iou:
                    continue

                height_difference = abs(
                    height_a - height_b
                )

                if (
                    height_difference
                    >= minimum_height_difference
                ):
                    union_detected = True

                    print(
                        f"Rejecting parent mask "
                        f"{parent_index}: "
                        f"contains child masks "
                        f"{child_a_index} and "
                        f"{child_b_index} with "
                        f"height difference "
                        f"{height_difference:.4f} m"
                    )

                    break

            if union_detected:
                break

        if union_detected:
            remove[parent_index] = True

    return [
        result
        for index, result in enumerate(results)
        if not remove[index]
    ]

def get_mask_median_height(
    mask,
    height,
    valid_depth,
    minimum_valid_pixels=5,
):
    mask = np.asarray(mask, dtype=bool)
    valid = (mask & valid_depth & np.isfinite(height))
    values = height[valid]

    if values.size < minimum_valid_pixels:
        return None

    return float(np.median(values))

def get_inner_boundary(mask, kernel_size=5,):
    mask = np.asarray(mask, dtype=bool)
    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    eroded = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)

    return mask & ~eroded

def should_merge_masks(
    mask_a,
    mask_b,
    height,
    valid_depth,
    dilation_size=9,
    maximum_edge_height_difference=0.015,
    minimum_edge_depth_pixels=5,
):
    """
    Return True if two SAM2 mask fragments are spatially
    close and have continuous depth at their facing edges.
    """

    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    kernel = np.ones(
        (dilation_size, dilation_size),
        dtype=np.uint8,
    )

    # --------------------------------------------------
    # 1. Check spatial proximity
    # --------------------------------------------------
    dilated_a = cv2.dilate(
        mask_a.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)

    dilated_b = cv2.dilate(
        mask_b.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)

    if not np.any(dilated_a & mask_b):
        return False

    # --------------------------------------------------
    # 2. Find mask boundaries
    # --------------------------------------------------
    boundary_a = get_inner_boundary(mask_a, kernel_size=5,)
    boundary_b = get_inner_boundary(mask_b, kernel_size=5,)

    # --------------------------------------------------
    # 3. Only examine boundary regions facing each other
    # --------------------------------------------------
    near_boundary_a = (boundary_a & dilated_b)
    near_boundary_b = (boundary_b & dilated_a)

    # --------------------------------------------------
    # 4. Measure local edge height
    # --------------------------------------------------
    edge_height_a = get_mask_median_height(
        near_boundary_a,
        height,
        valid_depth,
        minimum_valid_pixels=minimum_edge_depth_pixels,
    )
    edge_height_b = get_mask_median_height(
        near_boundary_b,
        height,
        valid_depth,
        minimum_valid_pixels=minimum_edge_depth_pixels,
    )

    # Need actual depth evidence at the interface.
    if (edge_height_a is None or edge_height_b is None):
        return False

    edge_difference = abs(edge_height_a - edge_height_b)

    # --------------------------------------------------
    # 5. Check local surface continuity
    # --------------------------------------------------
    if (edge_difference > maximum_edge_height_difference):
        return False

    return True

def merge_same_object_masks(
    results,
    height,
    valid_depth,
    dilation_size=9,
    maximum_edge_height_difference=0.015,
    minimum_edge_depth_pixels=5
):
    """
    Merge SAM2 masks likely belonging to the same physical object.

    Conditions:
      1. Same foreground component ID.
      2. Masks overlap or become connected after slight dilation.
      3. Spatially close.
      4. Continuous local depth where the masks meet.
    """

    if not results:
        return []

    used = [False] * len(results)
    merged_results = []

    kernel = np.ones(
        (dilation_size, dilation_size),
        dtype=np.uint8,
    )

    for i, result_a in enumerate(results):
        if used[i]:
            continue

        merged_mask = np.asarray(result_a.mask, dtype=bool).copy()

        best_result = result_a
        component_id = result_a.point.component_id

        used[i] = True
        changed = True
        while changed:
            changed = False

            for j, result_b in enumerate(results):
                if used[j]:
                    continue

                # Must originate from the same depth foreground component.
                if (result_b.point.component_id != best_result.point.component_id):
                    continue

                mask_b = np.asarray(result_b.mask, dtype=bool,)

                # --------------------------------------
                # Spatial + depth continuity check
                # --------------------------------------
                if not should_merge_masks(
                    mask_a=merged_mask,
                    mask_b=mask_b,
                    height=height,
                    valid_depth=valid_depth,
                    dilation_size=dilation_size,
                    maximum_edge_height_difference=maximum_edge_height_difference,
                    minimum_edge_depth_pixels=minimum_edge_depth_pixels,
                ):
                    continue

                # Merge.
                merged_mask |= mask_b

                used[j] = True
                changed = True

                # Keep metadata from highest-scoring member.
                if (result_b.sam_score > best_result.sam_score):
                    best_result = result_b

        merged_results.append(
            SamMaskResult(
                mask=merged_mask,
                sam_score=best_result.sam_score,
                point=PositivePoint(
                    x=best_result.point.x,
                    y=best_result.point.y,
                    score=best_result.point.score,
                    component_id=component_id,
                ),
            )
        )

    return merged_results

def should_merge_same_object(
    mask_a,
    mask_b,
    height,
    valid_depth,
    dilation_size=11,
    maximum_edge_height_difference=0.010,
    maximum_gap_height_drop=0.006,
    minimum_depth_pixels=5,
):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    kernel = np.ones(
        (dilation_size, dilation_size),
        dtype=np.uint8,
    )

    dilated_a = cv2.dilate(
        mask_a.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)

    dilated_b = cv2.dilate(
        mask_b.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)

    # --------------------------------------------------
    # 1. Must be spatially close
    # --------------------------------------------------

    if not np.any(dilated_a & dilated_b):
        return False

    # --------------------------------------------------
    # 2. Find facing boundaries
    # --------------------------------------------------

    boundary_a = get_inner_boundary(
        mask_a,
        kernel_size=5,
    )

    boundary_b = get_inner_boundary(
        mask_b,
        kernel_size=5,
    )

    facing_a = (
        boundary_a
        & dilated_b
        & valid_depth
        & np.isfinite(height)
    )

    facing_b = (
        boundary_b
        & dilated_a
        & valid_depth
        & np.isfinite(height)
    )

    if (
        np.count_nonzero(facing_a) < minimum_depth_pixels
        or
        np.count_nonzero(facing_b) < minimum_depth_pixels
    ):
        return False

    height_a = float(
        np.median(height[facing_a])
    )

    height_b = float(
        np.median(height[facing_b])
    )

    # --------------------------------------------------
    # 3. Facing surfaces must have similar height
    # --------------------------------------------------

    edge_difference = abs(
        height_a - height_b
    )

    if (edge_difference > maximum_edge_height_difference):
        return False

    # --------------------------------------------------
    # 4. Check the physical gap between masks
    # --------------------------------------------------

    gap = (
        dilated_a
        & dilated_b
        & ~mask_a
        & ~mask_b
        & valid_depth
        & np.isfinite(height)
    )

    if np.count_nonzero(gap) >= minimum_depth_pixels:
        gap_height = float(np.median(height[gap]))
        surface_height = min(height_a, height_b,)

        height_drop = surface_height - gap_height

        # Physical valley -> probably different objects
        if (
            height_drop
            > maximum_gap_height_drop
        ):
            return False

    return True

def calculate_component_coverage(
    mask,
    point,
    core_foreground,
):
    mask = np.asarray(mask, dtype=bool)
    foreground = np.asarray(core_foreground, dtype=bool)

    component_count, component_labels = cv2.connectedComponents(
        foreground.astype(np.uint8),
        connectivity=8,
    )

    x, y = int(point.x), int(point.y)

    if not (
        0 <= x < foreground.shape[1]
        and 0 <= y < foreground.shape[0]
    ):
        return 0.0

    component_id = component_labels[y, x]

    # Prompt does not belong to foreground.
    if component_id == 0:
        return 0.0

    component_mask = component_labels == component_id
    component_area = np.count_nonzero(component_mask)

    if component_area == 0:
        return 0.0

    intersection = np.count_nonzero(
        mask & component_mask
    )

    return intersection / component_area

def calculate_mask_area_ratio(mask):
    """
    Fraction of the cropped SAM2 image occupied by the mask.

    0.10 = 10% of image
    0.75 = 75% of image
    """
    mask = np.asarray(mask, dtype=bool)

    mask_area = np.count_nonzero(mask)
    image_area = mask.size

    if image_area == 0:
        return 0.0

    return mask_area / image_area