#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2

from remove_background import extract_core_foreground, load_depth
from output_utils import *
from points_generation import generate_positive_points, find_uncovered_foreground
from sam2_segmenter import generate_sam_masks, load_sam2_predictor, remove_duplicate_masks

DEFAULT_BACKGROUND = "/home/support/unseen_sku_ws/data/depth_images/background_box.tiff"
DEFAULT_OUTPUT_DIR = "/home/support/unseen_sku_ws/outputs"
DEFAULT_CHECKPOINT = "/home/support/unseen_sku_ws/external/sam2/checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Subtract an empty background image and generate an automatic" \
            "positive point for SAM2."
        )
    )

    parser.add_argument(
        "--background",
        default=DEFAULT_BACKGROUND,
        help="Empty background depth .npy or .tiff"
    )  
    parser.add_argument(
        "--scene",
        required=True,
        help="Loaded-tote depth, for example depth_0001.tiff",
    )
    parser.add_argument(
        "--rgb",
        required=True,
        help="Registered RGB image for point visualization",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Minimum foreground height in the depth image's units",
    )
    parser.add_argument(
        "--minimum-area",
        type=int,
        default=800,
        help="Minimum connected-component area in pixels",
    )
    parser.add_argument(
        "--residual-minimum-area",
        type=int,
        default=100,
        help="Minimum uncovered foreground area after SAM2",
    )
    parser.add_argument(
        "--maximum-rounds",
        type=int,
        default=3,
        help="Maximum number of iterative SAM2 prompt rounds",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=0.65,
        help="Preference for points far from a foreground boundary",
    )
    parser.add_argument(
        "--height-weight",
        type=float,
        default=0.35,
        help="Preference for points on high/exposed surfaces",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="SAM2 checkpoint path",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="SAM2 model configuration",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.75,
        help="Duplicate-mask IoU threshold",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for the mask and point-debug image",
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

def main():
    args = parse_arguments()

    background = load_depth(args.background)
    scene = load_depth(args.scene)
    rgb_image = load_rgb(args.rgb)

    # Get foreground
    result = extract_core_foreground(
        background=background,
        scene=scene,
        threshold=args.threshold,
        minimum_area=args.minimum_area
    )

    # Generate prompt points
    points = generate_positive_points(
        core_foreground=result.core_foreground,
        height=result.height,
        valid_depth=result.valid_depth,
        distance_weight=args.distance_weight,
        height_weight=args.height_weight
    )

    if not points:
        raise RuntimeError("No valid automatic positive point found")

    image_index = extract_image_index(args.scene)
    output_dir = Path(args.output_dir).expanduser()
    mask_path = output_dir / f"foreground_mask_{image_index}.png"
    prompt_path = output_dir / f"automatic_prompt_{image_index}.png"

    save_binary_mask(result.core_foreground, mask_path)

    save_prompt_debug(
        core_foreground=result.core_foreground,
        points=points,
        output_path=prompt_path,
        rgb_image=rgb_image,
    )

    print_depth_statistics(background, scene, result)
    print(f"Automatic positive points generated: {len(points)}")

    for point_number, point in enumerate(points, start=1):
        print(
            f"Point {point_number}: "
            f"x={point.x}, "
            f"y={point.y}, "
            f"score={point.score:.3f}, "
            f"component={point.component_id}"
        )

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint_path}"
        )

    image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    predictor, device = load_sam2_predictor(
        checkpoint_path=str(checkpoint_path),
        config_path=args.config
    )

    candidate_masks = generate_sam_masks(
        predictor=predictor,
        image_rgb=image_rgb,
        points=points,
        device=device
    )

    # accepted_masks = remove_duplicate_masks(
    #     results=candidate_masks,
    #     iou_threshold=args.duplicate_iou_threshold
    # )

    accepted_masks = candidate_masks

    print(f"SAM2 candidate masks: {len(candidate_masks)}")
    print(f"SAM2 masks after deduplication: {len(accepted_masks)}")

    save_sam_outputs(
        rgb_image=rgb_image,
        accepted_masks=accepted_masks,
        output_dir=output_dir,
        image_index=image_index
    )

if __name__ == "__main__":
    main()