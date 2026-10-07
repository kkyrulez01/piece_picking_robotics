from pathlib import Path

import cv2
import numpy as np

def extract_image_index(scene_path):
    """Extract 0001 from a filename such as depth_0001.tiff, without regex."""
    stem = Path(scene_path).expanduser().stem
    image_index = stem.rsplit("_", maxsplit=1)[-1]

    if not image_index.isdigit():
        raise ValueError(
            "Scene filename must end with a numeric index, "
            "for example depth_0001.tiff"
        )

    return image_index

def find_next_index(directories_and_patterns):
    highest_index = 0

    for directory, pattern in directories_and_patterns:
        directory = Path(directory)

        for path in directory.glob(pattern):
            try:
                index = int(
                    extract_image_index(path)
                )

                highest_index = max(
                    highest_index,
                    index,
                )

            except ValueError:
                continue

    return highest_index + 1

def save_binary_mask(mask, output_path):
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_uint8 = mask.astype(np.uint8) * 255

    if not cv2.imwrite(str(output_path), mask_uint8):
        raise RuntimeError(f"Could not save mask: {output_path}")

def save_prompt_debug(core_foreground, points, output_path, rgb_image=None):
    """
    Save the generated point on RGB when registered, otherwise on the mask.
    """
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rgb_image is None:
        base = core_foreground.astype(np.uint8) * 255
        debug_image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        if rgb_image.shape[:2] != core_foreground.shape:
            raise ValueError(
                "RGB and depth-mask dimensions do not match: "
                f"rgb={rgb_image.shape[:2]}, depth={core_foreground.shape}"
            )

        debug_image = rgb_image.copy()

    for point_number, point in enumerate(points, start=1):
        cv2.circle(
            debug_image,
            (point.x, point.y),
            8,
            (0, 0, 255),
            thickness=-1,
        )
        cv2.putText(
            debug_image,
            f"P{point_number}",
            (point.x + 10, max(point.y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), debug_image):
        raise RuntimeError(f"Could not save prompt debug image: {output_path}")

def print_depth_statistics(background, scene, result):
    valid_background = np.isfinite(background) & (background > 0)
    valid_scene = np.isfinite(scene) & (scene > 0)

    print(f"Depth dimensions: {scene.shape}")
    print(
        "Background median depth:",
        float(np.median(background[valid_background])),
    )
    print(
        "Scene median depth:",
        float(np.median(scene[valid_scene])),
    )

    if np.any(result.valid_depth):
        print(
            "Maximum measured height:",
            float(np.max(result.height[result.valid_depth])),
        )

    print(f"Components retained: {result.kept_components}")

def save_sam_outputs(rgb_image, accepted_masks, output_dir, image_index):
    """
    Save individual SAM2 masks and one combined coloured overlay.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = rgb_image.copy()

    # BGR colours for OpenCV.
    colours = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]

    for mask_number, result in enumerate(accepted_masks, start=1):
        mask = result.mask.astype(bool)

        if mask.shape != rgb_image.shape[:2]:
            raise ValueError(
                "SAM2 mask and RGB dimensions do not match: "
                f"mask={mask.shape}, "
                f"rgb={rgb_image.shape[:2]}"
            )

        mask_uint8 = mask.astype(np.uint8) * 255

        mask_path = output_dir / f"sam2_mask_{image_index}_{mask_number:02d}.png"

        if not cv2.imwrite(str(mask_path), mask_uint8):
            raise RuntimeError(
                f"Could not save SAM2 mask: {mask_path}"
            )

        colour = np.asarray(colours[(mask_number -1 ) % len(colours)], dtype=np.float32)

        original_pixels = overlay[mask].astype(np.float32)
        overlay[mask] = (0.55 * original_pixels + 0.45 * colour).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay,
            contours,
            contourIdx=-1,
            color=tuple(int(value) for value in colour),
            thickness=2,
        )

        point_x = result.point.x
        point_y = result.point.y

        cv2.circle(
            overlay,
            (point_x, point_y),
            7,
            (0, 0, 255),
            thickness=-1,
        )

        cv2.putText(
            overlay,
            (f"M{mask_number} {result.sam_score:.3f}"),
            (point_x + 10, max(point_y - 10, 20),),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        print(
            f"Mask {mask_number}: "
            f"SAM2 score= {result.sam_score:.3f}, "
            f"point=({point_x}, {point_y})"
        )
        print(f"Saved SAM2 mask: {mask_path}")

    overlay_path = output_dir / f"sam2_overlay_{image_index}.png"

    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(
            f"Could not save SAM2 overlay: {overlay_path}"
        )

    print(f"Saved SAM2 overlay: {overlay_path}")

def save_automatic_grid_debug(
    image_bgr,
    core_foreground,
    generator,
    output_path,
):
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_image = image_bgr.copy()

    height, width = core_foreground.shape

    normalized_grid = generator.point_grids[0]

    for normalized_x, normalized_y in normalized_grid:
        point_x = min(
            int(round(normalized_x * width)),
            width - 1,
        )

        point_y = min(
            int(round(normalized_y * height)),
            height - 1,
        )

        if core_foreground[point_y, point_x]:
            # Green: grid point lies on depth foreground.
            colour = (0, 255, 0)
        else:
            # Red: grid point lies on background.
            colour = (0, 0, 255)

        cv2.circle(
            debug_image,
            (point_x, point_y),
            4,
            colour,
            thickness=-1,
        )

    if not cv2.imwrite(
        str(output_path),
        debug_image,
    ):
        raise RuntimeError(
            f"Could not save grid debug image: {output_path}"
        )

def save_raw_sam_masks(
    image,
    masks,
    output_directory,
):
    """
    Save SAM2 masks immediately after generator.generate(),
    before custom post-processing.

    masks:
        List returned by SAM2AutomaticMaskGenerator.generate().
    """

    output_directory = Path(output_directory)
    raw_directory = output_directory / "raw_sam_masks"
    raw_directory.mkdir(parents=True, exist_ok=True)

    print(f"Raw SAM2 masks before custom filtering: {len(masks)}")

    # Save one image per mask
    for index, mask_result in enumerate(masks):
        mask = mask_result["segmentation"]
        mask_uint8 = mask.astype(np.uint8) * 255

        cv2.imwrite(
            str(raw_directory / f"mask_{index:03d}.png"),
            mask_uint8,
        )

        print(
            f"Mask {index:03d}: "
            f"area={mask_result.get('area', -1)}, "
            f"pred_iou={mask_result.get('predicted_iou', -1):.3f}, "
            f"stability={mask_result.get('stability_score', -1):.3f}"
        )

    # -------------------------------------------------------
    # Create an overlay showing every raw SAM2 mask
    # -------------------------------------------------------

    overlay = image.copy()

    rng = np.random.default_rng(12345)
    for index, mask_result in enumerate(masks):
        mask = mask_result["segmentation"]

        color = rng.integers(
            0,
            256,
            size=3,
            dtype=np.uint8,
        )

        colored = np.zeros_like(overlay)
        colored[:] = color

        overlay[mask] = cv2.addWeighted(
            overlay[mask],
            0.5,
            colored[mask],
            0.5,
            0,
        )

        # Label mask at its centroid
        ys, xs = np.where(mask)

        if len(xs) > 0:
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))

            cv2.putText(
                overlay,
                f"M{index}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(
        str(output_directory / "raw_sam_masks_overlay.png"),
        overlay,
    )

    print(f"Saved raw SAM2 masks to: {raw_directory}")