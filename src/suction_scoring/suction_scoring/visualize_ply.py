import argparse
import open3d as o3d

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cloud",
        required=True,
        help="Path to the PLY point cloud.",
    )
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(args.cloud)
    if cloud.is_empty():
        raise RuntimeError(
            f"Failed to load point cloud: {args.cloud}"
        )

    print("Points:", len(cloud.points))
    print("Has colours:", cloud.has_colors())
    print("Has normals:", cloud.has_normals())

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="PLY point cloud",
        point_show_normal=True,
    )

if __name__ == "__main__":
    main()