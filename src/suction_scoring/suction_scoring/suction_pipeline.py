"""ROS-independent Sim-Suction processing pipeline.

Based on the existing offline main.py: keep the same preprocessing,
network inference, mask association, scoring, spatial NMS, geometric filters,
and final ranking. Only file-based input/output is replaced by arrays.
"""
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import yaml

from .fusion_inputs import SceneInput
from .sim_suction_network import SimSuctionModel
from .sim_suction_preprocessing import prepare_sim_suction_cloud
from .suction_candidates import rank_candidates_per_mask, spatial_nms
from .suction_ranking import (
    normalize_sim_scores_per_object,
    calculate_weighted_ranking_scores,
)

from geometric_filters.boundary_filter import calculate_candidate_boundary_quality
from geometric_filters.surface_geometry import (
    build_object_surface_points,
    calculate_candidate_surface_geometry_quality,
)
from geometric_filters.footprint_filter import calculate_candidate_footprint_quality
from geometric_filters.clearance_filter import calculate_candidate_clearance_quality

@dataclass(frozen=True)
class SuctionResult:
    candidates_by_object: dict  # ORIGINAL ROS object_id -> ranked candidates
    valid_points: int
    sampled_xyz_m: np.ndarray
    sampled_scores: np.ndarray
    inference_seconds: float
    total_seconds: float
    geometry_is_mask_limited: bool

def load_sim_config(config_path):
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    # Does not have "input", "output" and "masks"
    required_sections = (
        "camera",
        "preprocessing",
        "model",
        "candidates",
        "geometric_filters",
        "ranking"
    )

    for section_name in required_sections:
        if not isinstance(config.get(section_name), dict):
            raise ValueError(
                f"Missing or invalid configuration section: '{section_name}'"
            )

    return config

class SuctionPipeline:
    def __init__(self, config):
        self.config = config
        model_config = config["model"]
        # Instantiate model
        self.model = SimSuctionModel(
            repository_path=model_config["repository_path"],
            checkpoint_path=model_config["checkpoint_path"],
            device=model_config.get("device", "cuda")
        )

    def process(self, scene: SceneInput, clearance_scene_xyz_m=None):
        """
        Infer once for the scene and rank candidates for every SAM2 mask.

        scene: point/mask union from the FusedObjectArray message.
        clearance_scene_xyz_m: OPTIONAL same-frame, same-capture FULL camera
            point cloud. At present, None uses only fused object points.
            Passing a full cloud does not replace MoveIt robot collision checks.
        """
        cfg = self.config
        camera_config = cfg["camera"]
        preprocessing_config = cfg["preprocessing"]
        model_config = cfg["model"]
        candidates_config = cfg["candidates"]
        geometric_filters_config = cfg["geometric_filters"]
        ranking_config = cfg["ranking"]

        camera_pos = np.asarray(cfg["camera"].get("position_m", [0,0,0]), dtype=np.float32)
        target_count = int(preprocessing_config.get("target_count", 5120))
        if target_count != 5120:
            raise ValueError("The pretrained Sim-Suction model requires exactly 5,120 points.")
    
        voxel_size_cm = preprocessing_config.get("voxel_size_cm", 0.2)
        if voxel_size_cm is not None:
            voxel_size_cm = float(voxel_size_cm)

        # Timer for recording time taken for 1 entire process
        total_time_start = time.perf_counter()

        # Preparing Sim Suction Cloud
        preprocessing_start_time = time.perf_counter()

        prepared = prepare_sim_suction_cloud(
            points_xyz_m=scene.xyz_m,
            camera_position_m=camera_pos,
            voxel_size_cm=voxel_size_cm,
            normal_radius_cm=float(preprocessing_config.get("normal_radius_cm", 1.5)),
            maximum_normal_neighbours=int(preprocessing_config.get("maximum_normal_neighbours", 30)),
            target_count=target_count,
            minimum_z_m=float(preprocessing_config.get("minimum_z_m", 0.0)),
            random_seed=int(preprocessing_config.get("random_seed", 42))
        )

        preprocessing_duration = time.perf_counter() - preprocessing_start_time

        # Inference using Sim-Suction model
        inference_start_time = time.perf_counter()

        raw_scores = np.asarray(self.model.predict_scores(prepared.features), dtype=np.float32)

        inference_duration = time.perf_counter() - inference_start_time
        if raw_scores.shape != (target_count,):
            raise RuntimeError(
                f"Expected {target_count} raw scores, "
                f"received shape {raw_scores.shape}."
            )
    
        if not np.isfinite(raw_scores).all():
            raise RuntimeError("Sim-Suction produced NaN or infinite scores.")
        
        valid = prepared.valid_sample_mask
        valid_xyz = (prepared.xyz_cm[valid] / 100.0).astype(np.float32)
        valid_normals = prepared.normals[valid]
        valid_filtered_src_idx = prepared.source_indices[valid]
        valid_uv = scene.uv[valid_filtered_src_idx]
        valid_scores = raw_scores[valid]
        object_masks = scene.masks  # (K,H,W): ranks use 1-based sequential IDs

        # Associate full-resolution foreground points with each SAM2 object.
        # These points are used for local surface geometry analysis.
        object_points_by_id = build_object_surface_points(
            foreground_xyz_m=scene.xyz_m,
            foreground_uv=scene.uv,
            object_masks=object_masks,
        )
    
        # Associate scored points with masks and rank them
        candidates_by_object = rank_candidates_per_mask(
            valid_xyz_m=valid_xyz,
            valid_normals=valid_normals,
            valid_scores=valid_scores,
            valid_source_indices=valid_filtered_src_idx,
            valid_uv=valid_uv,
            object_masks=object_masks,
            minimum_score=float(candidates_config.get("minimum_score",0.0)),
            maximum_candidates_per_object=len(valid_scores),
        )

        # Normalize the Sim-Suction scores for each object
        sim_normalization_config = ranking_config["sim_normalization"]
        candidates_by_object = normalize_sim_scores_per_object(
            candidates_by_object=candidates_by_object,
            lower_percentile=float(sim_normalization_config.get("lower_percentile", 5.0)),
            upper_percentile=float(sim_normalization_config.get("upper_percentile", 95.0))
        )

        # Apply boundary filter
        boundary_config = geometric_filters_config["boundary"]
        if bool(boundary_config.get("enabled", True)):
            (boundary_candidates,
            boundary_distance_maps, 
            normalized_boundary_distance_maps) = calculate_candidate_boundary_quality(
                candidates_by_object=candidates_by_object,
                object_masks=object_masks,
                exponent=float(boundary_config.get("exponent", 2.0))
            )
        else:
            boundary_candidates = candidates_by_object

        # Calculate score BEFORE spatial NMS
        final_ranking_config = final_ranking_config.get["final"]
        pre_nms_candidates = calculate_weighted_ranking_scores(
            candidates_by_object=boundary_candidates,
            sim_suction_weight=float(final_ranking_config.get("sim_suction_weight", 0.40)),
            boundary_weight=float(final_ranking_config.get("boundary_weight", 0.30)),
            surface_geometry_weight=0.0
        )

        nms_candidates_by_object = {}
        for sequential_id, candidates in pre_nms_candidates.items():
            nms_candidates_by_object[sequential_id] = spatial_nms(
                candidates=candidates,
                minimum_distance_m=float(candidates_config.get("nms_distance_m", 0.02)),
                maximum_candidates=int(candidates_config.get("maximum_nms_candidates", 10)),
            )

        # Surface geometry filter
        surface_config = geometric_filters_config["surface_geometry"]
        if bool(surface_config.get("enabled", True)):
            surface_candidates = calculate_candidate_surface_geometry_quality(
                candidates_by_object=nms_candidates_by_object,
                object_points_by_id=object_points_by_id,
                neighbourhood_radius_m=float(surface_config.get("neighbourhood_radius_m", 0.015)),
                minimum_neighbours=int(surface_config.get("minimum_neighbours",20)),
                maximum_surface_variation=float(surface_config.get("maximum_surface_variation",0.02)),
                exponent=float(surface_config.get("exponent",1.0)),
                missing_quality=float(surface_config.get("missing_quality", 0.5))
            )

        # Footprint filter
        footprint_config = geometric_filters_config["footprint"]
        if bool(footprint_config.get("enabled", True)):
            footprint_candidates = calculate_candidate_footprint_quality(
                candidates_by_object=surface_candidates,
                object_points_by_id=object_points_by_id,
                cup_radius_m=float(footprint_config.get("cup_radius_m", 0.010)),
                grid_resolution_m=float(footprint_config.get("grid_resolution_m", 0.002)),
                maximum_surface_deviation_m=float(footprint_config.get("maximum_surface_deviation_m", 0.003))
            )

        # Clearance filter
        clearance_config = geometric_filters_config["clearance"]
        masked_only = clearance_scene_xyz_m is None
        clearance_xyz = scene.xyz_m if masked_only else np.asarray(
            clearance_scene_xyz_m, dtype=np.float32)
        if bool(clearance_config.get("enabled", True)):
            clearance_candidates = calculate_candidate_clearance_quality(
                candidates_by_object=footprint_candidates,
                scene_points_m=clearance_xyz,
                camera_position_m=camera_pos,
                tool_radius_m=float(clearance_config.get("tool_radius_m",0.020)),
                approach_height_m=float(clearance_config.get("approach_height_m",0.060)),
                minimum_clearance_height_m=float(clearance_config.get("minimum_clearance_height_m",0.008)),
                safety_margin_m=float(clearance_config.get("safety_margin_m",0.003)),
                reject_collisions=bool(clearance_config.get("reject_collisions",False)),
            )

        # Calculate FINAL score
        final_ranked_candidates = calculate_weighted_ranking_scores(
            candidates_by_object=clearance_candidates,
            sim_suction_weight=float(final_ranking_config.get("sim_suction_weight", 0.40)),
            boundary_weight=float(final_ranking_config.get("boundary_weight", 0.30)),
            surface_geometry_weight=float(final_ranking_config.get("surface_geometry_weight", 0.40)),
            footprint_weight=float(final_ranking_config.get("footprint_weight", 0.30))
        )

        maximum = int(candidates_config.get("maximum_final_candidates", 5))
        # Helpers use sequential 1..K mask IDs. Restore original ROS object IDs
        # only on the published output; do not alter internal score algorithms.
        selected = {
            int(original_id): list(final_ranked_candidates.get(i, []))[:maximum]
            for i, original_id in enumerate(scene.object_ids, start=1)
        }

        return SuctionResult(
            candidates_by_object=selected,
            valid_points=int(np.count_nonzero(valid)),
            sampled_xyz_m=valid_xyz,
            sampled_scores=valid_scores,
            inference_seconds=inference_duration,
            total_seconds=time.perf_counter() - total_time_start,
            geometry_is_mask_limited=masked_only,
        )

