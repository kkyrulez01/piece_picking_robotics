import numpy as np
import open3d as o3d
import argparse
from pathlib import Path

def load_raw_pointcloud(file_path):
    file_path = Path(file_path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Point-cloud file not found: {file_path}")

    rgb = None
    frame_id = "unknown"

    if file_path.suffix.lower() == ".npy":
        point_data = np.load(file_path, allow_pickle=False)

    elif file_path.suffix.lower() == ".npz":
        with np.load(file_path, allow_pickle=False) as data:
            if "xyz" not in data:
                raise KeyError(
                    f"{file_path} does not contain an 'xyz' array. "
                    f"Available keys: {list(data.files)}"
                )

            point_data = data["xyz"]

            if "rgb" in data:
                rgb = data["rgb"]

            if "frame_id" in data:
                frame_id = str(data["frame_id"])

    else:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            "Expected .npy or .npz."
        )

    point_data = np.asarray(point_data, dtype=np.float32)

    if point_data.ndim == 2 and point_data.shape[1] >= 3:
        xyz = point_data[:, :3]

    elif point_data.ndim == 3 and point_data.shape[2] >= 3:
        xyz = point_data[..., :3].reshape(-1, 3)

    else:
        raise ValueError(
            "Expected point-cloud shape (N, 3), (N, 6), "
            "(H, W, 3), or (H, W, 6). "
            f"Received {point_data.shape}."
        )

    valid = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 2] > 0.0)
    )

    xyz = xyz[valid]

    colours = None

    if rgb is not None:
        flattened_rgb = np.asarray(rgb).reshape(-1, 3)

        if flattened_rgb.shape[0] != valid.shape[0]:
            raise ValueError(
                "RGB and XYZ do not contain the same number of points."
            )

        colours = flattened_rgb[valid].astype(np.float64) / 255.0

    if xyz.shape[0] == 0:
        raise ValueError("Point cloud contains no valid points.")

    print("File:", file_path)
    print("Stored shape:", point_data.shape)
    print("Valid points:", xyz.shape[0])
    print("Frame:", frame_id)
    print("Minimum XYZ in metres:", xyz.min(axis=0))
    print("Maximum XYZ in metres:", xyz.max(axis=0))
    print("Median XYZ in metres:", np.median(xyz, axis=0))

    return xyz, colours

def visualize_pointcloud(xyz, colours=None, point_size=2.0):
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(
        xyz.astype(np.float64)
    )

    if colours is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colours)
    else:
        point_cloud.paint_uniform_color([0.1, 0.7, 1.0])

    visualizer = o3d.visualization.Visualizer()

    window_created = visualizer.create_window(
        window_name="Raw point cloud",
        width=1280,
        height=720,
    )

    if not window_created:
        raise RuntimeError(
            "Open3D could not create a window. "
            "Check the DISPLAY and OpenGL configuration."
        )

    visualizer.add_geometry(point_cloud)

    render_options = visualizer.get_render_option()
    render_options.point_size = float(point_size)
    render_options.background_color = np.asarray(
        [0.05, 0.05, 0.05],
        dtype=np.float64,
    )

    # Display from approximately the camera viewpoint.
    view_control = visualizer.get_view_control()
    view_control.set_lookat(np.mean(xyz, axis=0))
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_up([0.0, -1.0, 0.0])
    view_control.set_zoom(0.7)

    print()
    print("Opening Open3D window.")
    print("Close the window to exit.")

    visualizer.run()
    visualizer.destroy_window()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize a .npy or .npz point cloud."
    )

    parser.add_argument(
        "--cloud",
        required=True,
        help="Path to a .npy or .npz point cloud.",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    xyz, colours = load_raw_pointcloud(args.cloud)

    visualize_pointcloud(
        xyz=xyz,
        colours=colours,
        point_size=args.point_size,
    )
    
if __name__ == "__main__":
    main()