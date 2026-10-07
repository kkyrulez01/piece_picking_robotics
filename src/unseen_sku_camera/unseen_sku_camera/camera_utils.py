#!/usr/bin/env python3

from pathlib import Path

import yaml
import numpy as np

def load_config(config_path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8",) as file:
        config = yaml.safe_load(file)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML dictionary."
        )

    return config

def load_camera_config(
    config_path,
    section_name,
):
    """
    Load one camera configuration section.

    Example:

        config = load_camera_config(
            "helios2.yaml",
            "helios2_capture",
        )
    """

    config = load_config(config_path)
    camera_config = config.get(section_name)

    if not isinstance(camera_config, dict,):
        raise ValueError(
            f"Missing or invalid "
            f"'{section_name}' section "
            f"in configuration file."
        )

    return camera_config

def find_next_index(
    output_directory,
    prefix,
    suffix=".npz",
):
    """
    Find the next available numeric index.

    Example:
        scene_0001.npz
        scene_0002.npz

    returns:
        3
    """

    output_directory = (Path(output_directory) .expanduser())

    pattern = (f"{prefix}_*{suffix}")

    highest_index = 0
    for path in output_directory.glob(pattern):
        try:
            index = int(path.stem.rsplit("_", 1,)[1])
            highest_index = max(highest_index, index,)

        except (IndexError, ValueError,):
            continue

    return highest_index + 1

def save_debug_npz(
    output_directory,
    prefix,
    data,
):
    """
    Save one debug NPZ file using the next
    available numeric index.

    Parameters
    ----------
    output_directory : str or Path
        Directory where the NPZ will be saved.

    prefix : str
        Filename prefix, for example "scene".

    data : dict
        Data to pass to np.savez().

    Returns
    -------
    index : int
        Assigned file index.

    output_path : Path
        Saved NPZ path.
    """

    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True,)

    index = find_next_index(
        output_directory=output_directory,
        prefix=prefix,
        suffix=".npz",
    )

    output_path = output_directory / f"{prefix}_{index:04d}.npz"
    np.savez(output_path, **data)

    return index, output_path

def load_world_to_camera_calibration(calibration_path, translation_scale=0.001):
    """
    Load a camera calibration NPZ containing:
        R_world2cam
        t_world2cam

    The stored translation is expected in millimetres.

    Returns
    -------
    R_world2cam : ndarray (3, 3)

    t_world2cam_m : ndarray (3,)
        Translation converted to metres.
    """

    calibration_path = Path(calibration_path).expanduser().resolve()

    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}"
        )

    with np.load(calibration_path, allow_pickle=False,) as data:
        required_keys = (
            "R_world2cam",
            "t_world2cam",
        )

        for key in required_keys:
            if key not in data.files:
                raise KeyError(
                    f"{calibration_path} does not "
                    f"contain '{key}'. "
                    f"Keys: {data.files}"
                )

        rotation = np.asarray(data["R_world2cam"], dtype=np.float64)
        translation_mm = np.asarray(data["t_world2cam"], dtype=np.float64).reshape(3)

    if rotation.shape != (3, 3):
        raise ValueError("R_world2cam must have shape (3, 3).")

    translation_m = translation_mm * translation_scale

    return rotation, translation_m

def invert_rigid_transform(rotation, translation,):
    """
    Invert:
        p_target = R * p_source + t

    Returns the inverse:
        p_source = R_inv * p_target + t_inv
    """

    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)

    rotation_inverse = rotation.T
    translation_inverse = -rotation_inverse @ translation

    return rotation_inverse, translation_inverse

def load_camera_to_world_calibration(calibration_path):
    """
    Load an eye-to-hand calibration and return
    camera -> world/base transformation.

    Stored calibration convention:
        p_camera = R_world2cam @ p_world + t_world2cam

    Returns
    -------
    R_cam2world : ndarray (3, 3)

    t_cam2world_m : ndarray (3,)
    """
    R_world2cam, t_world2cam = load_world_to_camera_calibration(calibration_path)

    return invert_rigid_transform(R_world2cam, t_world2cam,)

def build_transform_matrix(rotation, translation,):
    """
    Construct a 4x4 homogeneous transform.
    """

    transform = np.eye(4, dtype=np.float64,)

    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform

def load_helios_to_realsense_transform(calibration_path):
    """
    Load the direct Helios2 -> RealSense transform.

    Expected keys:
        T_helios2realsense

    or:
        R_helios2realsense
        t_helios2realsense

    Translation is already stored in metres.
    """

    calibration_path = Path(calibration_path).expanduser().resolve()

    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: "
            f"{calibration_path}"
        )

    with np.load(calibration_path, allow_pickle=False,) as data:
        if "T_helios2realsense" in data.files:
            transform = np.asarray(
                data["T_helios2realsense"],
                dtype=np.float64,
            )

        else:
            rotation = np.asarray(data["R_helios2realsense"], dtype=np.float64,)
            translation = np.asarray(data["t_helios2realsense"], dtype=np.float64,).reshape(3)

            transform = build_transform_matrix(rotation, translation,)

    if transform.shape != (4, 4):
        raise ValueError(
            "Direct calibration transform must have shape (4, 4)."
        )

    return transform

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Converts a 3x3 rotation matrix to quarternion [x, y, z, w].
        """

        # Get the matrix trace (1 + M_00 + M_11 + M_22)
        trace = np.trace(R)

        # Case 1: If trace is positive
        if trace > 0.0:
            S = 2.0 * np.sqrt(1.0 + trace)

            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S

        # Case 2: M_00 is th largest diagonal element
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0

            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S

        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0

            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S

        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0

            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion) # Normalize the quarternion

        return quaternion