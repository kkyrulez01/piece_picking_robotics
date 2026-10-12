#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneOutputPaths:
    scene_index: str

    root: Path
    sam2: Path
    mask_fusion: Path
    sim_suction: Path

def get_scene_output_paths(
    scene_index,
    output_root="/home/support/unseen_sku_ws/outputs",
    create=True,
):
    """
    Build all output directories for one scene.

    Example:
        scene_index = 37

    produces:
        /outputs/scene_0037/
            ↳ sam2/
            ↳ mask_fusion/
            ↳ sim_suction/
    """
    index = f"{int(scene_index):04d}"
    scene_root = Path(output_root).expanduser() / f"scene_{index}"

    paths = SceneOutputPaths(
        scene_index=index,
        root=scene_root,
        sam2=scene_root / "sam2",
        mask_fusion=scene_root / "mask_fusion",
        sim_suction=scene_root / "sim_suction",
    )

    if create:
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.sam2.mkdir(parents=True, exist_ok=True)
        paths.mask_fusion.mkdir(parents=True, exist_ok=True)
        paths.sim_suction.mkdir(parents=True, exist_ok=True)

    return paths