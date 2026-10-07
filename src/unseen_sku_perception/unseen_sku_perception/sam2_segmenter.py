from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class SamMaskResult:
    mask: np.ndarray
    sam_score: float
    point: Any

def load_sam2_predictor(checkpoint_path,config_path):  
    # Device agnostic code 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 on: {device}")

    model = build_sam2(
        config_path,
        checkpoint_path,
        device=device,
    )

    predictor = SAM2ImagePredictor(model)

    return predictor, device

def generate_sam_masks(predictor, image_rgb, points, device):
    results = []

    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    with torch.inference_mode():
        with autocast_context:
            # Compute the image embedding only once
            predictor.set_image(image_rgb)

            for point in points:
                point_coords = np.asarray([[point.x, point.y]], dtype=np.float32)
                point_labels = np.asarray([1], dtype=np.int32) # 1 indicates a foreground point

                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

                best_index = int(np.argmax(scores))
                best_mask = masks[best_index].astype(bool)

                # Requested positive point should appear inside selected mask
                if not best_mask[point.y, point.x]:
                    continue

                results.append(
                    SamMaskResult(
                        mask=best_mask,
                        sam_score=float(scores[best_index]),
                        point=point
                    )
                )

    return results

def calculate_mask_iou(mask_a, mask_b):
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)

    if union == 0:
        return 0.0

    return intersection / union

def remove_duplicate_masks(results, iou_threshold=0.75):
    # Process highest SAM2 score first
    sorted_results = sorted(results, key=lambda result: result.sam_score, reverse=True)

    accepted = []

    for candidate in sorted_results:
        duplicate = False

        for existing in accepted:
            iou = calculate_mask_iou(candidate.mask, existing.mask)

            if iou >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            accepted.append(candidate)

    return accepted