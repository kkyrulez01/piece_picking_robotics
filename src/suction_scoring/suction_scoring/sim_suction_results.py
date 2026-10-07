from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import cv2

def scores_to_colours(scores):
    """
    Convert suction scores into Viridis colours.

    The 1st and 99th percentiles are used as the displayed
    colour range so that individual outliers do not make the
    rest of the cloud appear as one colour.

    Returns:
        colours:
            Shape (N, 3), with values between 0 and 1.
        lower_score:
            Score represented by the lowest colour.
        upper_score:
            Score represented by the highest colour.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(
            f"scores must have shape (N,). Received {scores.shape}."
        )

    lower_score = float(np.percentile(scores, 1.0))
    upper_score = float(np.percentile(scores, 99.0))
    score_range = upper_score - lower_score

    if score_range <= 1e-8:
        normalized_scores = np.full(scores.shape, 0.5,dtype=np.float32)
    else:
        normalized_scores = (scores - lower_score) / score_range
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)

    colour_map = cm.get_cmap("viridis")

    colours = colour_map(normalized_scores)[:, :3]

    return colours.astype(np.float64), lower_score, upper_score

def create_score_coloured_cloud(xyz_m, normals, scores):
    """
    Create an Open3D point cloud coloured by suction score.
    """
    xyz_m = np.asarray(xyz_m, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    colours, lower_score, upper_score = scores_to_colours(scores)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_m.astype(np.float64))
    point_cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(colours)

    return point_cloud, lower_score, upper_score

def save_results(
    output_dir,
    input_cloud_path,
    prepared,
    scores,
    overwrite=False
):
    """
    Save valid scored samples as .npz and score-coloured .ply.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(input_cloud_path).stem
    output_stem = f"{input_stem}_sim_suction_scores"

    ply_path = output_dir / f"{output_stem}.ply"
    npz_path = output_dir / f"{output_stem}.npz"

    if not overwrite:
        existing_paths = [path for path in (ply_path, npz_path) if path.exists()]

        if existing_paths:
            existing_text = ", ".join(
                str(path) for path in existing_paths
            )

            raise FileExistsError(
                "Output already exists: "
                f"{existing_text}. Set output.overwrite to true to replace it."
            )

    valid_mask = prepared.valid_sample_mask
    valid_scores = scores[valid_mask]
    valid_xyz_cm = prepared.xyz_cm[valid_mask]

    # Return to ROS/Mech-Eye metre convention
    valid_xyz_m = (valid_xyz_cm / 100.0).astype(np.float32)

    valid_normals = prepared.normals[valid_mask].astype(np.float32)
    valid_source_indices = prepared.source_indices[valid_mask]

    coloured_cloud, colour_lower_score, colour_upper_score = create_score_coloured_cloud(xyz_m=valid_xyz_m,
                                                                                         normals=valid_normals,
                                                                                         scores=valid_scores)

    write_successful = o3d.io.write_point_cloud(str(ply_path),
                                                coloured_cloud,
                                                write_ascii=False,
                                                compressed=False,
                                                print_progress=False)

    if not write_successful:
        raise RuntimeError(f"Failed to save point cloud: {ply_path}")

    np.savez_compressed(
        npz_path,
        xyz_m=valid_xyz_m,
        normals=valid_normals,
        scores=valid_scores.astype(np.float32),
        source_indices=valid_source_indices,
        normalization_center_cm=prepared.normalization_center_cm,
        normalization_scale_cm=np.float32(prepared.normalization_scale_cm),
        coordinate_unit=np.asarray("m"),
        colour_lower_score=np.float32(colour_lower_score),
        colour_upper_score=np.float32(colour_upper_score)
    )

    return (ply_path, npz_path, coloured_cloud, colour_lower_score, colour_upper_score)

def save_top_candidates_debug_image(
    output_path,
    object_masks,
    top_candidates_by_object,
    source_image_path=None,
    mask_alpha=0.35,
    overwrite=False,
):
    """
    Save a 2D Debug image showing:
    - Every SAM2 object mask in a different colour.
    - The top suction candidates for each object.
    - Candidate object ID, rank and score.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Candidate debug image already exists: {output_path}"
        )

    object_masks = np.asarray(object_masks, dtype=bool)
    if object_masks.ndim != 3:
        raise ValueError(
            f"object_masks must have shape (K, H, W). Received {object_masks.shape}."
        )

    object_count, image_height, image_width = object_masks.shape

    if source_image_path is not None:
        source_image_path = Path(source_image_path).expanduser().resolve()

        debug_image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        if debug_image is None:
            raise RuntimeError(f"Failed to load source image: {source_image_path}")

        if debug_image.shape[:2] != (image_height,image_width,):
            raise ValueError(
                "Source image and SAM2 masks have different dimensions. "
                f"Image: {debug_image.shape[:2]}; "
                f"masks: {(image_height, image_width)}."
            )
        
    else:
        # Dark background when the original image is unavailable.
        debug_image = np.full(
            (image_height, image_width, 3),
            25,
            dtype=np.uint8,
        )

    # OpenCV uses BGR colour ordering.
    mask_colours = [
        (0, 255, 255),    # Yellow
        (255, 128, 0),    # Blue
        (0, 255, 0),      # Green
        (255, 0, 255),    # Magenta
        (0, 128, 255),    # Orange
        (255, 255, 0),    # Cyan
        (128, 0, 255),    # Pink
        (255, 0, 128),
    ]

    # Draw all object masks first.
    for mask_index in range(object_count):
        object_mask = object_masks[mask_index]
        colour = np.asarray(
            mask_colours[ mask_index % len(mask_colours)], dtype=np.float32,
        )

        current_pixels = debug_image[object_mask].astype(np.float32)
        blended_pixels = ((1.0 - mask_alpha) * current_pixels + mask_alpha * colour)

        debug_image[object_mask] = np.clip(
            blended_pixels,
            0,
            255,
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            object_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            debug_image,
            contours,
            contourIdx=-1,
            color=tuple(int(value) for value in colour),
            thickness=2,
        )

    outside_mask_count = 0

    # Draw the ranked candidate points.
    for object_id, candidates in (top_candidates_by_object.items()):
        mask_index = int(object_id) - 1

        if mask_index < 0 or mask_index >= object_count:
            raise ValueError(
                f"Object ID {object_id} does not map "
                f"to one of the {object_count} masks."
            )

        object_mask = object_masks[mask_index]
        object_colour = mask_colours[mask_index % len(mask_colours)]

        for candidate in candidates:
            u = int(candidate.uv[0])
            v = int(candidate.uv[1])

            if not (0 <= u < image_width and 0 <= v < image_height):
                raise ValueError(
                    f"Candidate UV {(u, v)} is outside "
                    f"the image dimensions "
                    f"{(image_height, image_width)}."
                )

            belongs_to_mask = bool(object_mask[v, u])
            if belongs_to_mask:
                marker_colour = object_colour
                status_text = ""

            else:
                # Red indicates an incorrect UV/mask mapping.
                marker_colour = (0, 0, 255)
                status_text = " OUTSIDE"
                outside_mask_count += 1

            # White outer circle keeps the marker visible.
            cv2.circle(
                debug_image,
                (u, v),
                radius=11,
                color=(255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            cv2.circle(
                debug_image,
                (u, v),
                radius=7,
                color=marker_colour,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            # Place the rank number inside the marker.
            cv2.putText(
                debug_image,
                str(candidate.rank),
                (u - 4, v + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            label = (
                f"O{object_id} "
                f"R{candidate.rank} "
                f"{candidate.ranking_score:.3f}"
                f"{status_text}"
            )

            text_origin = (min(u + 14, image_width - 180), max(v - 10, 20))

            # Black outline for readable text.
            cv2.putText(
                debug_image,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                debug_image,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    write_successful = cv2.imwrite(str(output_path), debug_image)
    if not write_successful:
        raise RuntimeError(
            f"Failed to save candidate debug image: {output_path}"
        )

    return output_path, outside_mask_count

def print_score_statistics(
    valid_scores,
    valid_xyz_m,
    valid_normals,
    valid_source_indices
):
    """
    Print score statistics and current highest-scoring point.
    """
    best_local_index = int(np.argmax(valid_scores))

    print()
    print("Suction score statistics")
    print("------------------------")
    print("Point count:", valid_scores.shape[0])
    print("Minimum:", float(np.min(valid_scores)))
    print("Maximum:", float(np.max(valid_scores)))
    print("Mean:", float(np.mean(valid_scores)))
    print("Median:", float(np.median(valid_scores)))
    print("Standard deviation:", float(np.std(valid_scores)))
    print("90th percentile:", float(np.percentile(valid_scores, 90)))
    print("95th percentile:", float(np.percentile(valid_scores, 95)))
    print("99th percentile:", float(np.percentile(valid_scores, 99)))
    print()
    print("Highest-scoring sampled point")
    print("-----------------------------")
    print("Score:", float(valid_scores[best_local_index]))
    print("XYZ in metres:", valid_xyz_m[best_local_index])
    print("Surface normal:", valid_normals[best_local_index])
    print("Original foreground index:", int(valid_source_indices[best_local_index]))
    print()

    print(
        "Note: this is only the highest network score. "
        "It has not yet passed suction-footprint, local "
        "surface-consistency, collision or reachability checks."
    )