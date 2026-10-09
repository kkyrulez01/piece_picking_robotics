from pathlib import Path
import argparse

import numpy as np
import cv2
import open3d as o3d

RGB_PATH = Path("/home/support/unseen_sku_ws/data/realsense/scene/images")
OUTPUT_ROOT = Path("/home/support/unseen_sku_ws/outputs")

def visualize_3d(paths):
    """
    Visualize fused Helios point clouds in Open3D.
    """
    geometries = []

    for path in paths:
        with np.load(path) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float64,)

        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]

        print(f"{path.name}: {len(xyz)} points")

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)

        geometries.append(cloud)

    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    )

    geometries.append(coordinate_frame)
    o3d.visualization.draw_geometries(geometries, window_name="SAM2 + Helios fusion")

def visualize_on_rgb(
    paths,
    rgb_path,
    output_path=None,
    maximum_points_per_object=10000
):
    """
    Draw fused Helios points onto the RealSense RGB image.

    Each object is drawn with a different colour.
    """
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR,)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")

    image_height, image_width = image.shape[:2]

    # BGR colours for OpenCV.
    colours = [
        (0, 0, 255),       # red
        (0, 255, 0),       # green
        (255, 0, 0),       # blue
        (0, 255, 255),     # yellow
        (255, 0, 255),     # magenta
        (255, 255, 0),     # cyan
        (0, 128, 255),     # orange
        (255, 128, 0),
        (128, 0, 255),
        (128, 255, 0),
    ]

    for object_index, path in enumerate(paths):
        with np.load(path) as data:
            if "uv" not in data.files:
                raise KeyError(
                    f"{path} does not contain 'uv'."
                )

            uv = np.asarray(data["uv"], dtype=np.int32,)

            if "object_id" in data.files:
                object_id = int(data["object_id"])
            else:
                object_id = object_index + 1

        if len(uv) == 0:
            print(
                f"Object {object_id}: no projected points"
            )
            continue

        # Safety check.
        valid = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < image_width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < image_height)
        )

        uv = uv[valid]

        colour = colours[object_index % len(colours)]

        # Avoid drawing an excessive number of
        # circles if a mask contains many points.
        step = max(1, len(uv) // maximum_points_per_object,)

        for u, v in uv[::step]:
            cv2.circle(
                image,
                (int(u), int(v)),
                1,
                colour,
                -1,
            )

        # Add object label around the centre of its projected points.
        if len(uv) > 0:
            centre_u = int(np.median(uv[:, 0]))
            centre_v = int(np.median(uv[:, 1]))

            cv2.putText(
                image,
                f"Object {object_id}",
                (centre_u, centre_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        print(
            f"Object {object_id}: "
            f"{len(uv)} projected RGB points"
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True,)

        cv2.imwrite(str(output_path), image)
        print(f"Saved RGB fusion overlay: {output_path}")

    cv2.imshow(
        "Helios points on RealSense RGB",
        image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize fused SAM2 + Helios point clouds "
            "for a scene."
        )
    )

    parser.add_argument(
        "index",
        type=int,
        help=(
            "Scene index. "
            "Example: 35 or 0035"
        ),
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    index = args.index
    index_string = f"{index:04d}"

    fused_directory = OUTPUT_ROOT / f"fused_{index_string}"
    rgb_path = RGB_PATH / f"image_{index_string}.png"
    overlay_output = fused_directory / f"helios_mask_fusion_overlay_{index_string}.png"
    
    paths = sorted(fused_directory.glob("object_*.npz"))
    if not paths:
        raise RuntimeError(
            f"No object NPZ files found in "
            f"{fused_directory}"
        )

    print()
    print("RGB projection")
    print("--------------")

    visualize_on_rgb(
        paths=paths,
        rgb_path=rgb_path,
        output_path=overlay_output,
    )

    print()
    print("3D visualization")
    print("----------------")

    visualize_3d(paths=paths)

if __name__ == "__main__":
    main()