#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def run_prediction(predictor, point):
    point_coords = np.asarray([point], dtype=np.float32)
    point_labels = np.asarray([1], dtype=np.int32)

    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
        else:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

    best_index = int(np.argmax(scores))

    return masks[best_index].astype(bool), float(scores[best_index])


def draw_mask(image, mask, point, score):
    result = image.copy()

    colour = np.asarray([0, 255, 0], dtype=np.float32)
    original_pixels = result[mask].astype(np.float32)

    result[mask] = (
        0.55 * original_pixels + 0.45 * colour
    ).astype(np.uint8)

    point_x, point_y = point

    cv2.circle(
        result,
        (point_x, point_y),
        8,
        (0, 0, 255),
        thickness=-1,
    )

    cv2.putText(
        result,
        f"SAM2 score: {score:.3f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to an RGB image",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "~/unseen_sku_ws/external/sam2/"
            "checkpoints/sam2.1_hiera_tiny.pt"
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
    )

    parser.add_argument(
        "--output",
        default="~/unseen_sku_ws/outputs/sam2_click_result.png",
    )

    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_path = Path(args.output).expanduser()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Loading SAM2 model...")

    model = build_sam2(
        args.config,
        str(checkpoint_path),
        device=device,
    )

    predictor = SAM2ImagePredictor(model)

    with torch.inference_mode():
        predictor.set_image(image_rgb)

    print("Model ready.")
    print("Left-click an object to segment it.")
    print("Press Q or Escape to exit.")

    window_name = "SAM2 click test"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image_bgr)

    def mouse_callback(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        point = (x, y)

        print(f"Selected pixel: x={x}, y={y}")

        mask, score = run_prediction(
            predictor,
            point,
        )

        result = draw_mask(
            image_bgr,
            mask,
            point,
            score,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(str(output_path), result)

        print(f"Mask score: {score:.4f}")
        print(f"Mask area: {np.count_nonzero(mask)} pixels")
        print(f"Saved result: {output_path}")

        cv2.imshow(window_name, result)

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()