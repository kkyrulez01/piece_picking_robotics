#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import cv2
from helper_functions import load_config

from points_generation import build_uniform_rgb_point_grid
from remove_background import extract_core_foreground, load_depth, create_support_foreground
from output_utils import (
    extract_image_index,
    print_depth_statistics,
    save_binary_mask,
    save_prompt_debug,
    save_sam_outputs,
    save_automatic_grid_debug,
    save_raw_sam_masks,
    save_full_sam_overlay
)
from mask_postprocessing import (
    suppress_contained_masks, 
    merge_same_object_masks,
    calculate_component_coverage,
    calculate_mask_area_ratio,
)
from depth_utils import filter_masks_by_depth, calculate_height_map
from sam2_automatic_segmenter import generate_and_filter_automatic_masks, load_automatic_mask_generator
from sam2_segmenter import remove_duplicate_masks

DEFAULT_CONFIG_FILE = "/home/support/unseen_sku_ws/src/unseen_sku_perception/config/automatic_sam2.yaml"
DEFAULT_SCENE = "/home/support/unseen_sku_ws/data/depth_images/depth_0006.tiff"
DEFAULT_RGB = "/home/support/unseen_sku_ws/data/images/image_0006.png"

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run RGB-D foreground and SAM2 segmentation."
    )

    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG_FILE,
        help="Perception pipeline YAML configuration."
    )

    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help="Current depth image, for example depth_0001.tiff",
    )

    parser.add_argument(
        "--rgb",
        default=DEFAULT_RGB,
        help="Registered RGB image, for example image_0001.png",
    )

    return parser.parse_args()

def load_rgb(path):
    if path is None:
        return None

    rgb_path = Path(path).expanduser()
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not read RGB image: {rgb_path}")

    return image

def create_workspace_roi(image_shape, roi_config):
    image_height, image_width = image_shape[:2]

    if not roi_config.get("enabled", True):
        return np.ones((image_height, image_width),dtype=bool,)

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

    roi_mask = np.zeros((image_height, image_width),dtype=bool)
    roi_mask[y_min:y_max, x_min:x_max] = True

    return roi_mask

def main():
    args = parse_arguments()
    config = load_config(args.config_file)

    input_config = config["inputs"]
    output_config = config["outputs"]
    foreground_config = config["foreground"]
    support_config = config["support_foreground"]
    grid_config = config["prompt_grid"]
    sam2_config = config["sam2"]
    filtering_config = config["mask_filtering"]
    roi_config = config["roi"]
    containment_config = config["containment"]

    background = load_depth(input_config["background"])
    scene = load_depth(args.scene)
    bgr_image = load_rgb(args.rgb)
    roi_mask = create_workspace_roi(image_shape=scene.shape, roi_config=roi_config)

    image_index = extract_image_index(args.scene)
    output_root = Path(output_config["output_directory"]).expanduser()
    rgb_path = Path(args.rgb).expanduser()
    camera_name = rgb_path.parent.parent.parent.name # Mechmind / Realsense
    output_dir = output_root / camera_name / f"results_{image_index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get foreground
    result = extract_core_foreground(
        background=background,
        scene=scene,
        threshold=foreground_config["threshold"],
        minimum_area=foreground_config["minimum_area"],
        roi_mask=roi_mask
    )

    # ---------------------------------------------------------
    # DEBUG: keep full-resolution foreground before ROI masking
    # ---------------------------------------------------------
    full_core_foreground = result.core_foreground.astype(bool).copy()

    full_foreground_path = (output_dir / f"foreground_full_raw_{image_index}.png")
    save_binary_mask(full_core_foreground, full_foreground_path)

    # Create support foreground
    support_foreground = create_support_foreground(
        core_foreground=result.core_foreground,
        height=result.height,
        valid_depth=result.valid_depth,
        support_height_threshold=support_config["height_threshold"],
        dilation_size=support_config["dilation_size"],
    )

    support_foreground &= roi_mask

    # Crop ROI for SAM2
    x_min = int(roi_config["x_min"])
    x_max = int(roi_config["x_max"])
    y_min = int(roi_config["y_min"])
    y_max = int(roi_config["y_max"])

    cropped_bgr_image = bgr_image[y_min:y_max, x_min:x_max,].copy()
    cropped_core_foreground = result.core_foreground[y_min:y_max, x_min:x_max,].copy()
    cropped_valid_depth = result.valid_depth[y_min:y_max, x_min:x_max,].copy()
    cropped_height = result.height[y_min:y_max, x_min:x_max,].copy()
    cropped_support_foreground = support_foreground[y_min:y_max, x_min:x_max,].copy()

    # Extract points for point grids from resulting foreground
    debug_points, point_grids = (
        build_uniform_rgb_point_grid(
            core_foreground=cropped_core_foreground,
            height=cropped_height,
            valid_depth=cropped_valid_depth,
            spacing=grid_config["spacing"],
            minimum_boundary_distance=grid_config["minimum_boundary_distance"],
            anchor_points_per_component=3
        )
    )

    print("Foreground SAM2 points generated:", len(debug_points),)

    # Save foreground
    foreground_path = output_dir / f"foreground_cropped_{image_index}.png"
    save_binary_mask(cropped_core_foreground, foreground_path)

    # Save support foreground
    support_path = output_dir / f"support_foreground_{image_index}.png"
    save_binary_mask(cropped_support_foreground, support_path)

    print_depth_statistics(background, scene, result)
    print(f"Saved foreground mask: {foreground_path}")

    # Load SAM2 automatic mask generator
    checkpoint_path = Path(sam2_config["checkpoint"]).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint_path}"
        )

    generator, device = load_automatic_mask_generator(
        checkpoint_path=str(checkpoint_path),
        config_path=sam2_config["model_config"],
        point_grids=point_grids,
        points_per_batch=sam2_config["points_per_batch"],
        predicted_iou_threshold=sam2_config["predicted_iou_threshold"],
        stability_threshold=sam2_config["stability_threshold"],
        minimum_region_area=sam2_config["minimum_region_area"],
    )

    # Debugging automatic grid
    grid_debug_path = output_dir / (f"automatic_grid_{image_index}.png")

    save_automatic_grid_debug(
        image_bgr=cropped_bgr_image,
        core_foreground=cropped_core_foreground,
        generator=generator,
        output_path=grid_debug_path,
    )

    print(f"Saved automatic grid: {grid_debug_path}")

    image_rgb = cv2.cvtColor(cropped_bgr_image, cv2.COLOR_BGR2RGB)

    # Automatic mask generation and depth filtering
    depth_filtered_masks, raw_masks = (
        generate_and_filter_automatic_masks(
            generator=generator,
            device=device,
            image_rgb=image_rgb,
            core_foreground=cropped_core_foreground,
            support_foreground=cropped_support_foreground,
            valid_depth=cropped_valid_depth,
            minimum_mask_area=filtering_config["minimum_mask_area"],
            minimum_foreground_ratio=filtering_config["minimum_foreground_ratio"],
            minimum_valid_depth_ratio=filtering_config["minimum_valid_depth_ratio"],
        )
    )
    # ---------------------------------------------------------
    # DEBUG: save SAM2 masks BEFORE our depth/post filtering
    # ---------------------------------------------------------
    save_raw_sam_masks(
        image=cropped_bgr_image,
        masks=raw_masks,
        output_directory=output_dir,
    )


    print(f"Raw automatic SAM2 masks: {len(raw_masks)}")
    print(f"Masks after depth filtering: {len(depth_filtered_masks)}")

    if not depth_filtered_masks:
        raise RuntimeError(
            "No SAM2 masks passed depth filtering. "
            "Try lowering the foreground ratio, stability "
            "threshold or predicted-IoU threshold."
        )

    # Reject masks covering most of the cropped workspace
    maximum_mask_area_ratio = 0.60

    workspace_filtered_masks = []
    for index, mask_result in enumerate(
        depth_filtered_masks,
        start=1,
    ):
        area_ratio = calculate_mask_area_ratio(mask_result.mask)
        print(
            f"Mask {index}: "
            f"image_area_ratio={area_ratio:.3f}"
        )

        if area_ratio > maximum_mask_area_ratio:
            print(
                f"  -> rejected: mask covers "
                f"{area_ratio * 100:.1f}% of workspace"
            )
            continue

        workspace_filtered_masks.append(mask_result)

    depth_filtered_masks = workspace_filtered_masks

    # Rank masks by physical foreground coverage
    for index, mask_result in enumerate(depth_filtered_masks, start=1):
        coverage = calculate_component_coverage(
            mask_result.mask,
            point=mask_result.point,
            core_foreground=cropped_core_foreground,
        )

        print(
            f"Mask {index}: "
            f"coverage={coverage:.3f}"
        )

    depth_filtered_masks.sort(
        key=lambda mask_result: calculate_component_coverage(
            mask_result.mask,
            mask_result.point,
            cropped_core_foreground,
        ),
        reverse=True,
    )

    # The automatic generator already performs duplicate
    # suppression, but this additional mask-IoU check removes
    # duplicates that remain after depth filtering.
    deduplicated_masks = remove_duplicate_masks(
        results=depth_filtered_masks,
        iou_threshold=filtering_config["duplicate_iou_threshold"],
    )

    merged_masks = merge_same_object_masks(
        results=deduplicated_masks,
        height=cropped_height,
        valid_depth=cropped_valid_depth,
        dilation_size=9,
        maximum_edge_height_difference=0.015,
        minimum_edge_depth_pixels=5
    )

    print(f"Masks after final deduplication: {len(deduplicated_masks)}")
    print(f"Masks after merging: {len(merged_masks)}")
    
    if containment_config["enabled"]:
        accepted_masks = suppress_contained_masks(
            results=merged_masks,
            containment_threshold=containment_config["containment_threshold"],
            maximum_area_ratio=containment_config["maximum_area_ratio"]
        )
    else:
        accepted_masks = merged_masks

    # Save prompt visualization
    accepted_prompt_points = [mask_result.point for mask_result in accepted_masks]

    prompt_path = output_dir / f"automatic_sam2_prompts_{image_index}.png"

    save_prompt_debug(
        core_foreground=cropped_core_foreground,
        points=accepted_prompt_points,
        output_path=prompt_path,
        rgb_image=cropped_bgr_image,
    )

    print(f"Saved accepted SAM2 prompts: {prompt_path}")

    # Save masks and combined overlay
    save_sam_outputs(
        rgb_image=cropped_bgr_image,
        accepted_masks=accepted_masks,
        output_dir=output_dir,
        image_index=image_index,
    )

    # Full 1280 x 720 overlay
    full_overlay_path = output_dir / f"sam2_overlay_full_{image_index}.png"

    save_full_sam_overlay(
        full_bgr_image=bgr_image,
        mask_results=accepted_masks,
        x_min=x_min,
        y_min=y_min,
        output_path=full_overlay_path,
    )

    print(f"Saved {len(accepted_masks)} accepted SAM2 masks")
    print("Automatic SAM2 processing complete.")

if __name__ == "__main__":
    main()
