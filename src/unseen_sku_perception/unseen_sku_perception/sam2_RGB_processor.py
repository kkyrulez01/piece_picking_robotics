#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

from .helper_functions import load_config, load_rgb
from .points_generation import (
    PositivePoint, 
    build_uniform_rgb_point_grid,
    resolve_workspace_roi
)
from .remove_background import load_depth
from .output_utils import (
    extract_image_index,
    save_binary_mask,
    save_prompt_debug,
    save_sam_outputs,
    save_automatic_grid_debug,
    save_raw_sam_masks,
)
from .mask_postprocessing import (
    suppress_contained_masks,
    suppress_union_masks_by_depth, 
    calculate_mask_area_ratio
)
from .depth_utils import calculate_height_map, filter_masks_by_depth
from .sam2_automatic_segmenter import load_automatic_mask_generator
from .sam2_segmenter import SamMaskResult, remove_duplicate_masks

DEFAULT_CONFIG_FILE = "/home/support/unseen_sku_ws/src/unseen_sku_perception/config/sam2.yaml"

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
        required=True,
        help="Current depth image, for example depth_0001.tiff",
    )

    parser.add_argument(
        "--rgb",
        required=True,
        help="Registered RGB image, for example image_0001.png",
    )

    parser.add_argument(
        "--log-images",
        action="store_true",
        default=None,
        help="Save SAM2 masks and diagnostic images for debugging."
    )

    return parser.parse_args()

class Sam2Processor:
    """
    Complete RGB-D + SAM2 perception pipeline.

    This class contains no ROS-specific code.

    It accepts:
        scene_depth
        bgr_image

    and returns:
        list[SamMaskResult]

    The returned masks remain in memory and can therefore
    be used directly by ROS without saving/reloading PNGs.

    If log_images=True, the same debug images used by the
    offline pipeline are also saved.
    """
    def __init__(
        self, 
        config_file=DEFAULT_CONFIG_FILE, 
        log_images=None
    ):
        self.config_file = Path(config_file).expanduser().resolve()
        self.log_images = bool(log_images)

        self.config = load_config(self.config_file)
        self.input_config = self.config["inputs"]
        self.output_config = self.config["outputs"]
        self.rgb_prompt_config = self.config["rgb_prompt"]
        self.depth_filter_config = self.config["depth_filter"]
        self.sam2_config = self.config["sam2"]
        self.filtering_config = self.config["mask_filtering"]
        self.roi_config = self.config["roi"]
        self.containment_config = self.config["containment"]

        if log_images is None:
            self.log_images = bool(self.output_config.get("log_images", False))
        else:
            self.log_images = bool(log_images)
    
        self.background = load_depth(self.input_config["background"])

        self.checkpoint_path = Path(self.sam2_config["checkpoint"]).expanduser().resolve()
        if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"SAM2 checkpoint not found: {self.checkpoint_path}"
                )

        self.generator = None
        self.device = None

    # ========================================================
    # SAM2 generator
    # ========================================================
    def _ensure_generator(self, point_grids):
        """
        Create the SAM2 generator once.

        On subsequent scenes only update the prompt grid.
        """
        if self.generator is None:
            print("Loading SAM2 automatic mask generator...")

            (
                self.generator,
                self.device,
            ) = load_automatic_mask_generator(
                checkpoint_path=str(self.checkpoint_path),
                config_path=self.sam2_config["model_config"],
                point_grids=point_grids,
                points_per_batch=self.sam2_config["points_per_batch"],
                predicted_iou_threshold=self.sam2_config["predicted_iou_threshold"],
                stability_threshold=self.sam2_config["stability_threshold"],
                minimum_region_area=self.sam2_config["minimum_region_area"],
            )

            print("SAM2 automatic mask generator ready.")

        else:
            # The automatic mask generator already stores
            # point_grids, which is also used by the existing
            # grid-debug utility.
            self.generator.point_grids = (point_grids)
        
    # ========================================================
    # Debug output
    # ========================================================
    def _get_output_directory(self, scene_index):
        """
        Return:
            outputs/
                ↳scene_0035/
                    ↳sam2/
        """
        output_root = Path(self.output_config["output_directory"]).expanduser().resolve()
        output_directory = output_root / f"scene_{scene_index:04d}" / "sam2"
        output_directory.mkdir(parents=True, exist_ok=True)

        return output_directory

    def _save_debug_outputs(
        self,
        scene_index,
        bgr_image,
        elevated_depth_mask,
        roi_mask,
        raw_masks,
        accepted_masks
    ):
        """
        Save all SAM2 diagnostic images for one scene.
        """
        output_directory = self._get_output_directory(scene_index)
        index_string = f"{scene_index:04d}"

        # Save elevated depth mask
        save_binary_mask(
            elevated_depth_mask, 
            output_directory / f"depth_elevated_roi_{index_string}.png"
        )

        # Save raw SAM2 masks before filtering
        save_raw_sam_masks(
            image=bgr_image,
            masks=raw_masks,
            output_directory=output_directory
        )

        # Save accepted prompt points
        accepted_prompt_points = [result.point for result in accepted_masks]
        save_prompt_debug(
            core_foreground=roi_mask,
            points=accepted_prompt_points,
            output_path=output_directory / f"automatic_sam2_prompts_{index_string}.png",
            rgb_image=bgr_image,
        )

        # Save prompt grid
        save_automatic_grid_debug(
            image_bgr=bgr_image,
            core_foreground=roi_mask,
            generator=self.generator,
            output_path=output_directory / f"automatic_grid_{index_string}.png",
        )

        # Save final masks and overlay
        save_sam_outputs(
            rgb_image=bgr_image,
            accepted_masks=accepted_masks,
            output_dir=output_directory,
            image_index=index_string
        )

        print(f"Debug images saved to: {output_directory}")

    # ========================================================
    # Main processing pipeline
    # ========================================================
    def process(self, scene_depth, bgr_image, scene_index=None):
        """
        Process one RGB-D scene.

        Parameters
        ----------
        scene_depth : np.ndarray
            Registered depth image in metres.
        bgr_image : np.ndarray
            Registered BGR image.
        scene_index : int or None
            Scene identifier. Required when image logging
            is enabled.

        Returns
        -------
        list[SamMaskResult]
            Final accepted masks.
        """
        scene_depth = np.asarray(scene_depth)
        bgr_image = np.asarray(bgr_image)

        if scene_depth.ndim != 2:
            raise ValueError(
                "scene_depth must have shape (height, width)."
            )

        if (bgr_image.ndim != 3 or bgr_image.shape[2] != 3):
            raise ValueError(
                "bgr_image must have shape (height, width, 3)."
            )

        if (self.log_images and scene_index is None):
            raise ValueError(
                "scene_index is required when log_images=True."
            )

        # Height map used for filtering only, not for prompting
        # SAM2 prompt generation is still RGB-only
        height, valid_depth = calculate_height_map(
            background_depth=self.background,
            scene_depth = scene_depth
        )

        # Get workspace ROI
        x_min, x_max, y_min, y_max = resolve_workspace_roi(
            image_shape=bgr_image.shape,
            roi_config=self.roi_config
        )
    
        image_height, image_width = bgr_image.shape[:2]
    
        if height.shape != (image_height, image_width):
            raise ValueError(
                "Depth and RGB dimensions must match.\n"
                f"RGB: {image_width}x{image_height}\n"
                f"Depth: {height.shape[1]}x{height.shape[0]}"
            )

        roi_mask = np.zeros(
            (image_height, image_width),
            dtype=bool,
        )
        roi_mask[y_min:y_max, x_min:x_max,] = True

        # Elevated-depth mask
        object_height_threshold = float(self.depth_filter_config["object_height_threshold"])
        
        elevated_depth_mask = (
            roi_mask
            & valid_depth
            & np.isfinite(height)
            & (height >= object_height_threshold)
        )

        # RGB-only SAM2 point grid
        prompt_spacing = int(self.rgb_prompt_config.get("spacing", 16))
        prompt_margin = int(self.rgb_prompt_config.get("margin", 15))
    
        debug_points, point_grids = build_uniform_rgb_point_grid(
            image_shape=bgr_image.shape,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            spacing=prompt_spacing,
            margin=prompt_margin,
        )
    
        print(
            f"RGB SAM2 points generated: {len(debug_points)} "
            f"(spacing={prompt_spacing}, margin={prompt_margin})"
        )

        # Initialize/reuse SAM2 generator
        self._ensure_generator(point_grids)

        # SAM2 RGB inference
        image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_masks = self.generator.generate(image_rgb)
            else:
                raw_masks = self.generator.generate(image_rgb)
    
        print(f"Raw RGB SAM2 masks: {len(raw_masks)}")

        # Basic RGB mask filtering
        maximum_mask_area_ratio = float(self.rgb_prompt_config.get("maximum_mask_area_ratio", 0.60))
        minimum_mask_area = int(self.filtering_config.get("minimum_mask_area", 0))
    
        rgb_masks = []
    
        for index, annotation in enumerate(raw_masks, start=1):
            mask = annotation["segmentation"].astype(bool)
            mask &= roi_mask
            mask_area = int(np.count_nonzero(mask))
            mask_inside_roi = mask[y_min:y_max, x_min:x_max]
            area_ratio = calculate_mask_area_ratio(mask_inside_roi)
    
            point_x, point_y = annotation["point_coords"][0]
            point_x = int(round(point_x))
            point_y = int(round(point_y))
    
            sam_score = float(annotation["predicted_iou"])
            stability = float(annotation["stability_score"])
    
            print(
                f"SAM mask {index}: "
                f"area={mask_area}, "
                f"area_ratio={area_ratio:.3f}, "
                f"pred_iou={sam_score:.3f}, "
                f"stability={stability:.3f}"
            )

            # Small and big mask rejection
            if mask_area < minimum_mask_area:
                print(
                    f"  -> rejected: area {mask_area} < "
                    f"{minimum_mask_area} pixels"
                )
                continue

            if area_ratio > maximum_mask_area_ratio:
                print(
                    "  -> rejected: giant workspace mask "
                    f"({area_ratio * 100:.1f}% of ROI)"
                )
                continue

            rgb_masks.append(
                SamMaskResult(
                    mask=mask,
                    sam_score=sam_score,
                    point=PositivePoint(
                        x=point_x, 
                        y=point_y,
                        score=sam_score,
                        component_id=-1,
                    ),
                )
            )

        print(f"Masks after RGB geometric filtering: {len(rgb_masks)}")

        # Depth shadow/background rejection
        depth_filtered_masks = filter_masks_by_depth(
            results=rgb_masks,
            height=height,
            valid_depth=valid_depth,
            object_height_threshold=object_height_threshold,
            minimum_elevated_ratio=float(self.depth_filter_config.get("minimum_elevated_ratio", 0.30)),
            minimum_valid_depth_ratio=float(self.depth_filter_config.get("minimum_valid_depth_ratio", 0.50,)),
            erosion_size=int(self.depth_filter_config.get("erosion_size", 5))
        )
    
        print(f"Masks after depth shadow rejection: {len(depth_filtered_masks)}")

        # Duplicate-mask suppression
        deduplicated_masks = remove_duplicate_masks(
            results=depth_filtered_masks,
            iou_threshold=self.filtering_config["duplicate_iou_threshold"]
        )
    
        print(f"Masks after final deduplication: {len(deduplicated_masks)}")

        # Containment filtering
        if self.containment_config["enabled"]:
            # Union filter first
            union_filtered_masks = suppress_union_masks_by_depth(
                results=deduplicated_masks,
                height=height,
                valid_depth=valid_depth,
                containment_threshold=self.containment_config.get("containment_threshold", 0.9),
                minimum_child_area_ratio=self.containment_config.get("minimum_child_area_ratio", 0.15),
                maximum_child_iou=self.containment_config.get("maximum_child_iou", 0.30),
                minimum_height_difference=self.containment_config.get("minimum_height_difference", 0.010),
                minimum_valid_pixels=self.containment_config.get("minimum_valid_pixels", 20)
            )
            # Then suppress
            accepted_masks = suppress_contained_masks(
                results=union_filtered_masks,
                containment_threshold=self.containment_config["containment_threshold"],
                maximum_area_ratio=self.containment_config["maximum_area_ratio"],
            )
        else:
            accepted_masks = deduplicated_masks
    
        print(f"Masks after containment filtering: {len(accepted_masks)}")

        # Optional debug images
        if self.log_images:
            self._save_debug_outputs(
                scene_index=int(scene_index),
                bgr_image=bgr_image,
                elevated_depth_mask=elevated_depth_mask,
                roi_mask=roi_mask,
                raw_masks=raw_masks,
                accepted_masks=accepted_masks,
            )

        return accepted_masks

def main():
    args = parse_arguments()

    processor = Sam2Processor(
        config_file=args.config_file,
        log_images=args.log_images,
    )

    scene_depth = load_depth(args.scene)
    bgr_image = load_rgb(args.rgb)
    scene_index = int(extract_image_index(args.scene))
    accepted_masks = processor.process(
            scene_depth=scene_depth,
            bgr_image=bgr_image,
            scene_index=scene_index,
        )
    

    print()
    print("SAM2 processing complete.")

    print(f"Final object masks: {len(accepted_masks)}")


if __name__ == "__main__":
    main()