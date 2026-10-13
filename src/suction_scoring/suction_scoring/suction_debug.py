"""Save per-scene Sim-Suction debug artifacts (no ROS node or inference here).

Consumes the already computed SuctionResult and SuctionCandidateArray.
Optional SceneInput masks allow a 2-D candidate overlay without an RGB image.

Outputs under output_root/scene_XXXX/sim_suction/:
  sim_suction_scores.npz  sampled, scored XYZ and raw model scores
  sim_suction_scores.ply  same sampled points, colored by relative score
  candidates.json         ranked candidates, fields, and timing/status metadata
  top_candidates.png      RGB/mask overlay of candidate pixels when possible

Note: scores.npz and scores.ply contain scored SAMPLED points, not the original
full-resolution foreground cloud. Debug saves do not change the ranking.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

_SCORE_FIELDS = (
    'sim_score', 'sim_score_normalized',
    'boundary_distance_px', 'boundary_distance_normalized', 'boundary_quality',
    'surface_variation', 'surface_quality',
    'footprint_support_ratio', 'footprint_point_count',
    'footprint_height_p90_m', 'footprint_quality',
    'clearance_valid', 'clearance_collision_count',
    'clearance_minimum_margin_m', 'ranking_score',
)

_PALETTE_BGR = (
    (60, 200, 60), (230, 150, 40), (70, 70, 240),
    (200, 100, 200), (40, 210, 220), (220, 180, 70),
)

def _finite_float(value):
    """
    Represent unavailable/NaN measurements as JSON null.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None

def _sampled_data(result):
    """
    Validate the arrays exposed by SimSuctionResult.
    """
    xyz = np.asarray(result.sampled_xyz_m, dtype=np.float32)
    scores = np.asarray(result.sampled_scores, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'Expected sampled_xyz_m (N,3), got {xyz.shape}')
    if scores.shape != (len(xyz),):
        raise ValueError('Sampled XYZ and scores must have the same length')
    if not np.isfinite(xyz).all() or not np.isfinite(scores).all():
        raise ValueError('Cannot save non-finite XYZ or Sim-Suction scores')
    return xyz, scores

def save_score_npz(path, result, scene_index, frame_id):
    """
    Save numerical scores and sampled points for offline analysis.
    """
    xyz, scores = _sampled_data(result)
    np.savez_compressed(
        path,
        xyz_m=xyz,
        sim_scores=scores,
        scene_index=np.uint32(scene_index),
        frame_id=np.asarray(frame_id),
        valid_points=np.uint32(result.valid_points),
    )
    return path

def save_colored_ply(path, result):
    """
    Save a sampled 3-D cloud colored by the relative Sim-Suction score.
    """
    xyz, scores = _sampled_data(result)
    if len(scores) == 0:
        return None

    lo, hi = np.percentile(scores, [5, 95])
    if hi > lo:
        normalized = np.clip((scores - lo) / (hi - lo), 0.0, 1.0)
    else:
        normalized = np.full(len(scores), 0.5, dtype=np.float32)

    gray = np.rint(normalized * 255).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET).reshape(-1, 3)
    rgb = bgr[:, ::-1]

    with Path(path).open('w', encoding='ascii') as stream:
        stream.write('ply\nformat ascii 1.0\n')
        stream.write(f'element vertex {len(xyz)}\n')
        stream.write('property float x\nproperty float y\nproperty float z\n')
        stream.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        stream.write('end_header\n')
        for point, color in zip(xyz, rgb):
            stream.write(
                f'{float(point[0]):.7f} {float(point[1]):.7f} '
                f'{float(point[2]):.7f} '
                f'{int(color[0])} {int(color[1])} {int(color[2])}\n'
            )
    return path

def save_candidates_json(path, output, result):
    """
    Save all ranked candidates and scores in an easy-to-inspect format.
    """
    entries = []
    for candidate in output.candidates:
        row = {
            'object_id': int(candidate.object_id),
            'rank': int(candidate.rank),
            'position_m': [
                float(candidate.position.x),
                float(candidate.position.y),
                float(candidate.position.z),
            ],
            'normal': [
                float(candidate.normal.x),
                float(candidate.normal.y),
                float(candidate.normal.z),
            ],
            'uv': [int(candidate.pixel_u), int(candidate.pixel_v)],
            'source_index': int(candidate.source_index),
        }
        for field in _SCORE_FIELDS:
            value = getattr(candidate, field)
            if isinstance(value, (bool, np.bool_)):
                row[field] = bool(value)
            elif field in ('footprint_point_count', 'clearance_collision_count'):
                row[field] = int(value)
            else:
                row[field] = _finite_float(value)
        entries.append(row)

    document = {
        'scene_index': int(output.scene_index),
        'frame_id': str(output.header.frame_id),
        'status': str(output.status),
        'error_message': str(output.error_message),
        'geometry_is_mask_limited': bool(output.geometry_is_mask_limited),
        'processing_time_s': _finite_float(output.processing_time_s),
        'model_inference_time_s': _finite_float(result.inference_seconds),
        'pipeline_time_s': _finite_float(result.total_seconds),
        'sampled_point_count': int(result.valid_points),
        'candidates': entries,
    }
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
    return path

def save_candidate_overlay(path, output, scene=None, source_image_path=None):
    """
    Draw object masks and ranked candidates onto RGB (or mask-only) image.

    If the RGB image is absent, a mask-only image is used when scene is given.
    If neither an RGB image nor a SceneInput is available, no image is saved.
    """
    canvas = None
    if source_image_path is not None and Path(source_image_path).is_file():
        canvas = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)

    if scene is not None:
        masks = np.asarray(scene.masks, dtype=bool)
        if masks.ndim != 3 or len(masks) != len(scene.object_ids):
            raise ValueError('Invalid scene masks/object_ids for debug image')
        height, width = masks.shape[1:]
        if canvas is None or canvas.shape[:2] != (height, width):
            canvas = np.full((height, width, 3), 25, dtype=np.uint8)
        for index, mask in enumerate(masks):
            color = np.asarray(_PALETTE_BGR[index % len(_PALETTE_BGR)])
            canvas[mask] = (
                0.65 * canvas[mask] + 0.35 * color
            ).astype(np.uint8)
    elif canvas is None:
        return None

    height, width = canvas.shape[:2]
    color_for_id = {}
    if scene is not None:
        color_for_id = {
            int(obj_id): _PALETTE_BGR[i % len(_PALETTE_BGR)]
            for i, obj_id in enumerate(scene.object_ids)
        }

    for candidate in output.candidates:
        u, v = int(candidate.pixel_u), int(candidate.pixel_v)
        if not (0 <= u < width and 0 <= v < height):
            continue
        color = color_for_id.get(
            int(candidate.object_id),
            _PALETTE_BGR[int(candidate.object_id) % len(_PALETTE_BGR)],
        )
        rank = int(candidate.rank)
        cv2.circle(canvas, (u, v), 10 if rank == 1 else 6, color, 2)
        label = f'O{int(candidate.object_id)} R{rank}'
        label_xy = (min(u + 12, max(0, width - 85)), max(18, v - 8))
        cv2.putText(canvas, label, label_xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, label_xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)

    if not cv2.imwrite(str(path), canvas):
        raise OSError(f'Could not write candidate image: {path}')
    return path

def write_suction_debug(
    output_root,
    scene_index,
    output,
    result,
    scene=None,
    source_image_path=None,
):
    """
    Write the artifacts for one completed scene, without changing its scores.

    Works with the current SuctionResult and SuctionCandidateArray.
    scene: optional SceneInput, used to draw SAM2 masks in top_candidates.png.
    """
    scene_index = int(scene_index)
    root = Path(output_root).expanduser().resolve()
    directory = root / f'scene_{scene_index:04d}' / 'sim_suction'
    directory.mkdir(parents=True, exist_ok=True)

    # RealSense RGB frames are saved here by the current camera pipeline.
    if source_image_path is None:
        source_image_path = (
            root.parent / 'data' / 'realsense' / 'scene' / 'images'
            / f'image_{scene_index:04d}.png'
        )

    files = {}
    files['scores_npz'] = save_score_npz(
        directory / 'sim_suction_scores.npz',
        result, scene_index, output.header.frame_id,
    )
    files['scores_ply'] = save_colored_ply(
        directory / 'sim_suction_scores.ply', result,
    )
    files['candidates_json'] = save_candidates_json(
        directory / 'candidates.json', output, result,
    )
    files['candidate_image'] = save_candidate_overlay(
        directory / 'top_candidates.png', output,
        scene=scene, source_image_path=source_image_path,
    )
    return {key: str(value) for key, value in files.items() if value is not None}