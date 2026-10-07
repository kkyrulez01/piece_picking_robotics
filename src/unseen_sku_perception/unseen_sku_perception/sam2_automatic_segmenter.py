import numpy as np
import torch
import cv2

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from .points_generation import PositivePoint
from .sam2_segmenter import SamMaskResult

def load_automatic_mask_generator(
    checkpoint_path,
    config_path,
    point_grids,
    points_per_batch=16,
    predicted_iou_threshold=0.80,
    stability_threshold=0.90,
    minimum_region_area=50,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 automatic generator on: {device}")

    model = build_sam2(
        config_path,
        checkpoint_path,
        device=device,
    )

    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=None,
        point_grids=point_grids,
        points_per_batch=points_per_batch,
        pred_iou_thresh=predicted_iou_threshold,
        stability_score_thresh=stability_threshold,
        box_nms_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=minimum_region_area,
        output_mode="binary_mask",
    )

    return generator, device

def generate_and_filter_automatic_masks(
    generator,
    device,
    image_rgb,
    core_foreground,
    support_foreground,
    valid_depth,
    minimum_mask_area,
    minimum_foreground_ratio,
    minimum_valid_depth_ratio,
):
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                raw_masks = generator.generate(image_rgb)
        else:
            raw_masks = generator.generate(image_rgb)

    filtered_masks = []

    component_count, component_labels = cv2.connectedComponents(
        core_foreground.astype(np.uint8),
        connectivity=8,
    )

    for annotation in raw_masks:
        mask = annotation["segmentation"].astype(bool)

        mask_area = np.count_nonzero(mask)
        if mask_area < minimum_mask_area:
            continue

        point_x, point_y = annotation["point_coords"][0]

        point_x = int(np.clip(
            round(point_x),
            0,
            core_foreground.shape[1] - 1,
        ))

        point_y = int(np.clip(
            round(point_y),
            0,
            core_foreground.shape[0] - 1,
        ))

        component_id = int(component_labels[point_y, point_x])

        # Prompt must come from the reliable core mask.
        if not core_foreground[point_y, point_x]:
            continue

        # Validate the larger SAM mask against the expanded mask.
        support_overlap = np.count_nonzero(mask & support_foreground)
        valid_mask_area = np.count_nonzero(mask & valid_depth)
        valid_depth_ratio = (valid_mask_area / max(mask_area, 1))
        foreground_ratio = support_overlap / max(valid_mask_area, 1)

        if (foreground_ratio < minimum_foreground_ratio):
            continue

        if (valid_depth_ratio < minimum_valid_depth_ratio):
            continue

        # Keep your existing code that creates and appends
        # SamMaskResult below this point.
        mask_result = SamMaskResult(
            mask=mask,
            sam_score=float(annotation["predicted_iou"]),
            # stability_score=float(annotation["stability_score"]),
            point=PositivePoint(
                x=point_x,
                y=point_y,
                score=float(annotation["predicted_iou"]),
                component_id=component_id,
            ),
        )

        filtered_masks.append(mask_result)

    return filtered_masks, raw_masks