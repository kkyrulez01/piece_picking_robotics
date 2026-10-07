import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from .sam2_helios_foreground_filter import foreground_combined_sam2_intersect

from .sim_suction_network import SimSuctionModel
from .sim_suction_preprocessing import prepare_sim_suction_cloud
from .sim_suction_results import print_score_statistics, save_results, save_top_candidates_debug_image
from .suction_candidates import load_sam2_masks, rank_candidates_per_mask, spatial_nms
from .suction_ranking import normalize_sim_scores_per_object, calculate_weighted_ranking_scores

from geometric_filters.boundary_filter import calculate_candidate_boundary_quality
from geometric_filters.surface_geometry import build_object_surface_points, calculate_candidate_surface_geometry_quality
from geometric_filters.footprint_filter import calculate_candidate_footprint_quality
from geometric_filters.clearance_filter import calculate_candidate_clearance_quality

DEFAULT_REPOSITORY_PATH = "/home/support/unseen_sku_ws/external/Sim-Suction-API"
DEFAULT_CHECKPOINT_PATH = "/home/support/unseen_sku_ws/external/Sim-Suction-API/Sim-Suction-Pointnet/models/MV_PCL_1550_500.model"
DEFAULT_CONFIG_PATH = "/home/support/unseen_sku_ws/src/sim_suction_test/config/sim_suction_config.yaml"

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=("Run offline Sim-Suction inference using a YAML config file.")
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to Sim-Suction YAML config file."
    )

    return parser.parse_args()

def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

        required_sections = (
            "input",
            "camera",
            "preprocessing",
            "model",
            "output",
            "masks",
            "candidates",
            "geometric_filters",
            "ranking"
        )

        for section_name in required_sections:
            if not isinstance(config.get(section_name), dict):
                raise ValueError(
                    f"Missing or invalid configuration section: '{section_name}'"
                )

    return config, config_path

def extract_scene_index(cloud_path):
    """
    Extract the numeric scene index from filenames such as:
        foreground_0006.npz
    """
    stem = Path(cloud_path).stem

    try:
        index_text = stem.rsplit("_", maxsplit=1)[1]
        return int(index_text)

    except (IndexError, ValueError) as error:
        raise ValueError(
            "Unable to extract scene index from cloud filename: "
            f"{cloud_path}. Expected a name such as "
            "'foreground_0006.npz'."
        ) from error
    
def main():
    args = parse_arguments()
    config, config_path = load_config(args.config)

    input_config = config["input"]
    camera_config = config["camera"]
    preprocessing_config = config["preprocessing"]
    model_config = config["model"]
    output_config = config["output"]
    masks_config = config["masks"]
    candidates_config = config["candidates"]

    geometric_filters_config = config["geometric_filters"]
    boundary_config = geometric_filters_config["boundary"]
    surface_config = geometric_filters_config["surface_geometry"]
    footprint_config = geometric_filters_config["footprint"]
    clearance_config = geometric_filters_config["clearance"]

    ranking_config = config["ranking"]
    sim_normalization_config = ranking_config["sim_normalization"]
    final_ranking_config = ranking_config["final"]

    cloud_path = Path(input_config["cloud_path"]).expanduser().resolve()
    if not cloud_path.exists():
        raise FileNotFoundError(f"Input foreground cloud not found: {cloud_path}")

    camera_position_m = np.asarray(
        camera_config.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float32,
    )

    target_count = int(preprocessing_config.get("target_count", 5120))
    if target_count != 5120:
        raise ValueError("The pretrained Sim-Suction model requires exactly 5,120 points.")

    voxel_size_cm = preprocessing_config.get("voxel_size_cm", 0.2)
    if voxel_size_cm is not None:
        voxel_size_cm = float(voxel_size_cm)

    print(f"Configuration: {config_path}")
    print(f"Input cloud:   {cloud_path}")
    print()

    # Timer for recording time taken for 1 entire process
    total_start_time = time.perf_counter()

    # Timer for recording time taken for loading pointcloud and uv
    load_start_time = time.perf_counter()

    # Load foreground NPZ
    with np.load(cloud_path, allow_pickle=False) as foreground_data:
        if "uv" not in foreground_data:
            raise KeyError(f"Foreground file does not contain UV: {cloud_path}")

        xyz = np.asarray(foreground_data["xyz"])
        rgb = None
        uv = np.asarray(foreground_data["uv"])

        if "rgb" in foreground_data:
            try:
                rgb = np.asarray(foreground_data["rgb"])
            except ValueError as error:
                if "Object arrays cannot be loaded" in str(error):
                    print(
                        "RGB is stored as an object array; ignoring RGB for Sim-Suction."
                    )
                    rgb = None
                else:
                    raise

        xyz_realsense = (
            np.asarray(foreground_data["xyz_realsense"]) 
            if "xyz_realsense" in foreground_data 
            else None
        )

        scene_indices = (
            np.asarray(foreground_data["scene_indices"])
            if "scene_indices" in foreground_data
            else None
        )

    original_foreground_count = len(xyz)

    # Load SAM2 masks
    scene_index = extract_scene_index(cloud_path)
    mask_root_directory = Path(masks_config["root_directory"]).expanduser().resolve()
    mask_directory = mask_root_directory / "realsense" / f"results_RGB_{scene_index:04d}"
    object_masks, mask_paths = load_sam2_masks(
        mask_directory=mask_directory,
        scene_index=scene_index
    )

    # Sim-Suction output directory
    sim_suction_root_dir = Path(output_config["directory"])
    sim_suction_output_dir = sim_suction_root_dir / f"scene_{scene_index:04d}" / "sim_suction"
    sim_suction_output_dir.mkdir(parents=True,exist_ok=True)

    # Helios2 foreground intersection with SAM2 masks
    intersected = foreground_combined_sam2_intersect(
        xyz=xyz,
        uv=uv,
        object_masks=object_masks,
        rgb=rgb,
        xyz_realsense=xyz_realsense,
        scene_indices=scene_indices,
        dilation_size=int(masks_config.get("union_dilation_size",5)),
    )
    foreground_xyz_m = np.asarray(intersected["xyz"], dtype=np.float32)
    foreground_uv = np.asarray(intersected["uv"])
    foreground_original_indices = np.asarray(intersected["source_indices"], dtype=np.int64)
    foreground_rgb = intersected.get("rgb")
    foreground_xyz_realsense = intersected.get("xyz_realsense")
    foreground_scene_indices = intersected.get("scene_indices")

    load_duration = time.perf_counter() - load_start_time

    print()
    print(f"Original Helios foreground: {original_foreground_count}")
    print(f"Filtered object foreground: {len(foreground_xyz_m)}")
    print("Input coordinate unit:      metres")

    # Clean, downsample, estimate normals, sample/pad and normalize
    # Timer for recording time taken for entire pre-processing
    preprocessing_start_time = time.perf_counter()

    prepared = prepare_sim_suction_cloud(
        points_xyz_m=foreground_xyz_m,
        camera_position_m=camera_position_m,
        voxel_size_cm=voxel_size_cm,
        normal_radius_cm=float(preprocessing_config.get("normal_radius_cm", 1.5)),
        maximum_normal_neighbours=int(preprocessing_config.get("maximum_normal_neighbours", 30)),
        target_count=target_count,
        minimum_z_m=float(preprocessing_config.get("minimum_z_m", 0.0)),
        random_seed=int(preprocessing_config.get("random_seed", 42))
    )

    preprocessing_duration = time.perf_counter() - preprocessing_start_time

    valid_mask = prepared.valid_sample_mask
    valid_point_count = int(np.count_nonzero(valid_mask))
    padded_point_count = target_count - valid_point_count

    print(f"Model input shape:        {prepared.features.shape}")
    print(f"Valid sampled points:     {valid_point_count}")
    print(f"Padded points:            {padded_point_count}")
    print(
        "Normalization centre:   "
        f"{prepared.normalization_center_cm} cm"
    )
    print(
        "Normalization scale:    "
        f"{prepared.normalization_scale_cm:.6f} cm"
    )

    # Load the pretrained Sim-Suction model
    model = SimSuctionModel(
        repository_path=model_config["repository_path"],
        checkpoint_path=model_config["checkpoint_path"],
        device=model_config.get("device", "cuda")
    )

    # Run inference
    # Timer for recording time  taken for inference
    inference_start_time = time.perf_counter()

    scores = model.predict_scores(prepared.features)
    scores = np.asarray(scores, dtype=np.float32)

    inference_duration = time.perf_counter() - inference_start_time

    if scores.shape != (target_count,):
        raise RuntimeError(
            f"Expected {target_count} scores, "
            f"received shape {scores.shape}."
        )

    if not np.isfinite(scores).all():
        raise RuntimeError("Sim-Suction produced NaN or infinite scores.")

    # Remove scores associated with padded points.
    valid_scores = scores[valid_mask]
    valid_xyz_m = (prepared.xyz_cm[valid_mask] / 100.0).astype(np.float32)
    valid_normals = prepared.normals[valid_mask]
    valid_filtered_source_indices = prepared.source_indices[valid_mask]
    valid_uv = foreground_uv[valid_filtered_source_indices]
    valid_original_source_indices = foreground_original_indices[valid_filtered_source_indices]

    # Associate full-resolution foreground points with each SAM2 object.
    # These points are used for local surface geometry analysis.
    object_points_by_id = build_object_surface_points(
        foreground_xyz_m=foreground_xyz_m,
        foreground_uv=foreground_uv,
        object_masks=object_masks,
    )

    # Associate scored points with masks and rank them
    candidates_by_object = rank_candidates_per_mask(
        valid_xyz_m=valid_xyz_m,
        valid_normals=valid_normals,
        valid_scores=valid_scores,
        valid_source_indices=valid_filtered_source_indices,
        valid_uv=valid_uv,
        object_masks=object_masks,
        minimum_score=float(candidates_config.get("minimum_score",0.0)),
        # maximum_candidates_per_object=int(candidates_config.get("maximum_candidates_per_object",100)),
        maximum_candidates_per_object=len(valid_scores),
    )

    # Normalize the Sim-Suction scores for each object
    candidates_by_object = normalize_sim_scores_per_object(
        candidates_by_object=candidates_by_object,
        lower_percentile=float(sim_normalization_config.get("lower_percentile", 5.0)),
        upper_percentile=float(sim_normalization_config.get("upper_percentile", 95.0))
    )

    # Apply boundary filter
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
    pre_nms_candidates = calculate_weighted_ranking_scores(
        candidates_by_object=boundary_candidates,
        sim_suction_weight=float(final_ranking_config.get("sim_suction_weight", 0.40)),
        boundary_weight=float(final_ranking_config.get("boundary_weight", 0.30)),
        surface_geometry_weight=0.0
    )

    nms_candidates_by_object = {}
    # Spatial NMS uses pre_nms_candidates
    for object_id, object_candidates in pre_nms_candidates.items():
        print(f"Object {object_id}: {len(object_candidates)} total candidates")

        if not object_candidates:
            print(" No candidate passed the minimum score.")
            continue

        # Apply spatial NMS
        selected_nms = spatial_nms(
            candidates=object_candidates,
            minimum_distance_m=float(config["candidates"].get("nms_distance_m", 0.02)),
            maximum_candidates=int(candidates_config.get("maximum_nms_candidates", 10))
        )

        nms_candidates_by_object[object_id] = selected_nms

        # Without spatial NMS
        # selected = object_candidates[:5]

    # Surface geometry filter
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
    else:
        surface_candidates = nms_candidates_by_object

    # Footprint filter
    if bool(footprint_config.get("enabled", True)):
        footprint_candidates = calculate_candidate_footprint_quality(
            candidates_by_object=surface_candidates,
            object_points_by_id=object_points_by_id,
            cup_radius_m=float(footprint_config.get("cup_radius_m", 0.010)),
            grid_resolution_m=float(footprint_config.get("grid_resolution_m", 0.002)),
            maximum_surface_deviation_m=float(footprint_config.get("maximum_surface_deviation_m", 0.003))
        )
    else:
        footprint_candidates=surface_candidates

    # Clearance filter
    if bool(clearance_config.get("enabled", True)):
        clearance_candidates = calculate_candidate_clearance_quality(
            candidates_by_object=footprint_candidates,
            scene_points_m=foreground_xyz_m,
            camera_position_m=camera_position_m,
            tool_radius_m=float(clearance_config.get("tool_radius_m",0.020)),
            approach_height_m=float(clearance_config.get("approach_height_m",0.060)),
            minimum_clearance_height_m=float(clearance_config.get("minimum_clearance_height_m",0.008)),
            safety_margin_m=float(clearance_config.get("safety_margin_m",0.003)),
            reject_collisions=bool(clearance_config.get("reject_collisions",False)),
        )
        
    else:
        clearance_candidates = footprint_candidates
    
    # Calculate FINAL score
    final_ranked_candidates = calculate_weighted_ranking_scores(
        candidates_by_object=clearance_candidates,
        sim_suction_weight=float(final_ranking_config.get("sim_suction_weight", 0.40)),
        boundary_weight=float(final_ranking_config.get("boundary_weight", 0.30)),
        surface_geometry_weight=float(final_ranking_config.get("surface_geometry_weight", 0.40)),
    )

    # Top candidates
    # For now take top 3 highest scoring points in each object mask
    # Later these candidates will be tested in rank order using footprint,
    # normal-consistency and collision checks.
    top_candidates_by_object = {}
    top_candidates = []
    maximum_final_candidates = int(candidates_config.get("maximum_final_candidates", 5))

    print()
    print("Top candidates by object")
    print("------------------------")

    for object_id, object_candidates in final_ranked_candidates.items():
        selected = object_candidates[:maximum_final_candidates]
        top_candidates_by_object[object_id] = selected

        # Flat list that can later be passed to visualization.
        top_candidates.extend(selected)
        print(
            f"Object {object_id}: "
            f"{len(selected)} final candidates"
        )
        for candidate in selected:
            print(f"  Rank: {candidate.rank}")
            print(f"  Sim-Suction score: {candidate.sim_score:.6f}")
            print(f"  Normalized Sim-Suction score: {candidate.sim_score_normalized:.6f}")
            print(f"  Boundary distance: {candidate.boundary_distance_px:.2f} px")
            print(f"  Normalized boundary distance: {candidate.boundary_distance_normalized:.3f}")
            print(f"  Boundary quality: {candidate.boundary_quality:.3f}")
            if candidate.surface_variation is not None:
                print(f"  Surface variation: {candidate.surface_variation:.6f}")
            print(f"  Surface quality: {candidate.surface_quality:.3f}")
            print(f"  Footprint support: {candidate.footprint_support_ratio:.3f}")
            print(f"  Footprint points: {candidate.footprint_point_count}")
            if candidate.footprint_height_p90_m is not None:
                print(f"  Footprint height P90: {candidate.footprint_height_p90_m * 1000.0:.2f} mm")
            print(f"  Footprint quality: {candidate.footprint_quality:.3f}")
            print(f"  Clearance valid: {candidate.clearance_valid}")
            print(f"  Clearance collision points: {candidate.clearance_collision_count}")
            if candidate.clearance_minimum_margin_m is not None:
                print(f"  Minimum clearance margin: {candidate.clearance_minimum_margin_m * 1000.0:.2f} mm")
            else:
                print("  Minimum clearance margin: no observed obstacle")
            print(f"  Ranking score: {candidate.ranking_score:.3f}")
            print(f"  XYZ: {candidate.xyz_m} m")
            print(f"  Normal: {candidate.normal}")
            print(f"  UV: {candidate.uv}")
        print("------------------------")

    candidate_debug_path = sim_suction_output_dir / f"{cloud_path.stem}_top5_candidates.png"

    candidate_debug_path, outside_mask_count = save_top_candidates_debug_image(
        output_path=candidate_debug_path,
        object_masks=object_masks,
        top_candidates_by_object=top_candidates_by_object,
        source_image_path=masks_config.get("source_image_path"),
        overwrite=bool(output_config.get("overwrite", False)),
    )

    print()
    print("Candidate visualization")
    print("-----------------------")
    print(f"Saved: {candidate_debug_path}")
    print("Candidates outside assigned mask:", outside_mask_count)

    if outside_mask_count > 0:
        print(
            "WARNING: At least one candidate does not lie inside its assigned " \
            "SAM2 mask. Check UV registration and source-index mapping."
        )

    # Print score statistics
    print_score_statistics(
        valid_scores=valid_scores,
        valid_xyz_m=valid_xyz_m,
        valid_normals=valid_normals,
        valid_source_indices=valid_original_source_indices,
    )

    # Save the numerical results and coloured point cloud.
    (ply_path,
     npz_path,
     coloured_cloud,
     colour_lower_score,
     colour_upper_score
    ) = save_results(output_dir=sim_suction_output_dir,
                     input_cloud_path=cloud_path,
                     prepared=prepared,
                     scores=scores,
                     overwrite=bool(output_config.get("overwrite", False)))

    # Total time taken for 1 whole process
    total_duration = time.perf_counter() - total_start_time

    print()
    print("Output")
    print("------")
    print(f"Numerical results: {npz_path}")
    print(f"Coloured cloud:    {ply_path}")
    print(
        "Displayed colour range: "
        f"{colour_lower_score:.6f} to "
        f"{colour_upper_score:.6f}"
    )

    print()
    print("Timing")
    print("------")
    print(f"Cloud loading:         {load_duration:.3f} seconds")
    print(f"Cloud preprocessing:   {preprocessing_duration:.3f} seconds")
    print(f"Model inference:     {inference_duration:.3f} seconds")
    print(f"Total:                 {total_duration:.3f} seconds")

    if bool(output_config.get("visualize", True)):
        o3d.visualization.draw_geometries(
            [coloured_cloud],
            window_name = "Sim-Suction scores"
        )

if __name__ == "__main__":
    main()