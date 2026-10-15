"""
Suction candidate class, to be shared with other filtering modules.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SuctionCandidate:
    object_id: int
    rank: int

    # Sim-Suction score
    sim_score: float
    
    # Final ranking score
    ranking_score: float

    xyz_m: np.ndarray
    normal: np.ndarray
    uv: np.ndarray

    source_index: int
    valid_index: int

    # Normalized Sim-Suction score
    sim_score_normalized: float | None = None

    # Geometric information
    # Boundary
    boundary_distance_px: float | None = None
    boundary_distance_normalized: float | None = None
    boundary_quality: float | None = None

    # Surface geometry
    surface_neighbour_count: int | None = None
    surface_variation: float | None = None
    surface_quality: float | None = None

    # Suction footprint
    footprint_support_ratio: float | None = None
    footprint_point_count: int | None = None
    footprint_height_p90_m: float | None = None
    footprint_quality: float | None = None

    # Clearance
    clearance_valid: bool | None = None
    clearance_collision_count: int | None = None
    clearance_minimum_margin_m: float | None = None

