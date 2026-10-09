#!/usr/bin/env python3
"""Orchestrate existing Helios projection and SAM2 mask-fusion algorithms.

This is a pure NumPy/OpenCV processor, independent of ROS 2.
`project_to_image.py` and `mask_pointcloud_fusion.py` retain the algorithms.
`mask_fusion_node.py` handles ROS message transport.
"""

from pathlib import Path
from collections.abc import Mapping
import numpy as np

from .project_to_image import(
    load_extrinsic,
    load_realsense_intrinsics,
    project_helios_to_image
)
from .mask_pointcloud_fusion import (
    associate_projected_points_with_masks,
    save_fused_objects
)

class MaskFusionProcessor:
    """Project Helios XYZ to RGB pixels and associate points with SAM2 masks.

    Args:
        helios_to_realsense_file: NPZ with T_helios2realsense (4x4).
        realsense_calibration_file: NPZ with camera_matrix (3x3).
        erosion_size: Mask erosion kernel width; 1 disables erosion.
        z_buffer: Retain only the nearest Helios point per RGB pixel.
                  Defaults to False, matching the offline CLI's default.
        save_debug: Save projected and per-object NPZs on each scene.
        output_root: Directory containing scene_XXXX/mask_fusion/.

    """
    def __init__(
        self,
        helios_to_realsense_file,
        realsense_calibration_file,
        erosion_size=1,
        z_buffer=False,
        save_debug=False,
        output_root="~/unseen_sku_ws/outputs"
    ):
        self.erosion_size = int(erosion_size)
        self.z_buffer = bool(z_buffer)
        self.save_debug = bool(save_debug)
        self.output_root = Path(output_root).expanduser().resolve()

        self.transform = load_extrinsic(Path(helios_to_realsense_file).expanduser().resolve())
        self.camera_matrix, self.dist_coeffs = load_realsense_intrinsics(
            Path(realsense_calibration_file).expanduser().resolve()
        )

    @staticmethod
    def _prepare_masks(object_masks):
        """
        Return (ordered object IDs, K x H x W boolean masks).
        """
        if isinstance(object_masks, Mapping):
            object_ids = [int(obj_id) for obj_id in object_masks]
            masks = [np.asarray(mask, dtype=bool) for mask in object_masks.values()]
            if len(set(object_ids)) != len(object_ids):
                raise ValueError("Duplicate SAM2 object IDs")
        else:
            masks_array = np.asarray(object_masks, dtype=bool)
            if masks_array.ndim != 3:
                raise ValueError(
                    "object_masks must be a dict {object_id: (H,W) mask} "
                    "or a (K,H,W) array"
                )
            object_ids = list(range(1, len(masks_array) + 1))
            masks = list(masks_array)

        if not masks:
            return [], None

        first_shape = masks[0].shape
        if len(first_shape) != 2 or min(first_shape) < 1:
            raise ValueError(f"Invalid SAM2 mask dimensions: {first_shape}")
        if any(mask.shape != first_shape for mask in masks):
            raise ValueError("All SAM2 object masks must share image dimensions")

        return object_ids, np.stack(masks, axis=0)

    def process(self, helios_xyz, object_masks, scene_index=None, rgb_image=None):
        """Fuse one Helios cloud and one full-size SAM2 mask set.

        Args:
            helios_xyz: (H,W,3) or (N,3) Helios XYZ in metres.
            object_masks: {object_id: bool mask (H,W)} or stack (K,H,W).
            scene_index: Scene number used only for optional debug files.
            rgb_image: Optional RGB (H,W,3) array for projected color logging.

        Returns:
            dict mapping original SAM2 object IDs to dictionaries with
            helios_xyz, realsense_xyz, uv, and source_indices.

        All 'helios_xyz' output coordinates remain in the Helios optical frame.
        """
        if self.save_debug and scene_index is None:
            raise ValueError("scene_index is required when save_debug=True")

        xyz = np.asarray(helios_xyz, dtype=np.float64)
        if xyz.ndim not in (2, 3) or xyz.shape[-1] != 3:
            raise ValueError(f"Expected Helios XYZ (N,3) or (H,W,3), got {xyz.shape}")
        helios_xyz_flat = xyz.reshape(-1, 3)

        object_ids, masks = self._prepare_masks(object_masks)
        if masks is None:
            return {}

        image_shape = masks.shape[1:3]
        if rgb_image is not None:
            rgb_image = np.asarray(rgb_image)
            if rgb_image.shape != (*image_shape, 3):
                raise ValueError(
                    f"RGB image must have shape {(*image_shape, 3)}, "
                    f"got {rgb_image.shape}"
                )

        # 1. Project Helios2 pointcloud onto RGB image
        rs_xyz, u, v, source_indices = project_helios_to_image(
            helios_xyz_m=helios_xyz_flat,
            transform=self.transform,
            camera_matrix=self.camera_matrix,
            image_shape=image_shape,
            z_buffer=self.z_buffer,
        )

        # 2. Then associate the projected points with the masks
        points_by_object = associate_projected_points_with_masks(
            object_masks=masks,
            helios_xyz_flat=helios_xyz_flat,
            rs_xyz=rs_xyz,
            u=u,
            v=v,
            source_indices=source_indices,
            erosion_size=self.erosion_size,
        )

        # Add object_id and turn points_by_object to a dictionary
        points_by_object = {
            object_id: points_by_object[index]
            for index, object_id in enumerate(object_ids, start=1)
        }

        # For debugging
        if self.save_debug:
            output_directory = (
                self.output_root / f"scene_{int(scene_index):04d}" / "mask_fusion"
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            self._save_projected_points(
                output_directory=output_directory,
                scene_index=int(scene_index),
                rs_xyz=rs_xyz,
                u=u,
                v=v,
                source_indices=source_indices,
                rgb_image=rgb_image,
            )
            # Reuse existing save_fused_objects(), with original NPZ fields.
            save_fused_objects(points_by_object, output_directory)

        return points_by_object

    @staticmethod
    def _save_projected_points(
        output_directory,
        scene_index,
        rs_xyz,
        u,
        v,
        source_indices,
        rgb_image=None
    ):
        """
        Write the correspondence NPZ format created by project_to_image.py.
        """
        data = {
            "xyz_realsense": np.asarray(rs_xyz, dtype=np.float32),
            "u": np.asarray(u, dtype=np.int32),
            "v": np.asarray(v, dtype=np.int32),
            "source_indices": np.asarray(source_indices, dtype=np.int64),
            "xyz_unit": np.array("m"),
        }
        if rgb_image is not None:
            data["rgb"] = rgb_image[data["v"], data["u"]]

        output_path = (
            Path(output_directory)
            / f"helios_projected_to_realsense_rgb_{scene_index:04d}.npz"
        )
        np.savez(output_path, **data)
        
        return output_path
