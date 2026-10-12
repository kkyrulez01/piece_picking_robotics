import numpy as np
from pathlib import Path

def make_transform(R, t):
    T = np.eye(4)
    # [rows, columns]
    T[:3, :3] = R # 3x3 Rotation matrix, R
    T[:3, 3] = t.reshape(3) # 3x1 translation vector, t
    return T

def load_world_to_camera(calibration_path) :
    """
    Load calibration .npz file
    """
    calibration_path = Path(calibration_path)

    with np.load(calibration_path) as calibration:
        required = {"R_world2cam", "t_world2cam"}

        rotation = calibration["R_world2cam"]
        translation = calibration["t_world2cam"]

    return make_transform(rotation, translation)

def compute_helios_to_realsense(
    helios_calibration_path,
    realsense_calibration_path,
):
    """
    Compute T_helios2realsense.

    T_helios2realsense = T_world2realsense @ inverse(T_world2helios)
    """
    t_world2realsense = load_world_to_camera(realsense_calibration_path)
    t_world2helios = load_world_to_camera(helios_calibration_path)

    t_helios2realsense = t_world2realsense @ np.linalg.inv(t_world2helios)
    # Currently units are in millimeters, convert to metres
    t_helios2realsense[:3, 3] *= 0.001

    return t_helios2realsense

def save_extrinsic(output_path, transform):
    """
    Save the t_helios2realsense transform.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    np.savez(
        output_path,
        T_helios2realsense = transform,
        R_helios2realsense = rotation,
        t_helios2realsense=translation
    )

def main():
    transform = compute_helios_to_realsense(
        helios_calibration_path="/home/support/unseen_sku_ws/tools/camera_calibration/calibration_data/helios2_260922/helios2_260922.npz",
        realsense_calibration_path="/home/support/unseen_sku_ws/tools/camera_calibration/calibration_data/realsense_D455f_260922/realsense_D455f_260922.npz"
    )

    output_dir = "/home/support/unseen_sku_ws/tools/camera_calibration/calibration_data/helios_to_realsense_260922"
    output_file = output_dir + "/helios_to_realsense_260922"
    save_extrinsic(
        output_path=output_file,
        transform=transform
    )

    print("Helios2 -> RealSense transform (translation in metres):")
    print(transform)
    print()
    print("Translation [m]:", transform[:3, 3])
    print(
        "Camera-origin separation [m]:",
        float(np.linalg.norm(transform[:3, 3])),
    )
    print(f"Saved: {Path(output_file).resolve()}")

if __name__ == "__main__":
    main()