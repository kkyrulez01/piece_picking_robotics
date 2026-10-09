#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import open3d as o3d


FUSED_DIRECTORY = Path(
    "/home/support/unseen_sku_ws/outputs/fused_0036"
)

OUTPUT_DIRECTORY = (
    FUSED_DIRECTORY / "cluster_analysis"
)


# ============================================================
# DBSCAN settings
#
# Assumes XYZ is in metres.
# ============================================================

DBSCAN_EPS = 0.010       # 10 mm
DBSCAN_MIN_POINTS = 20

# A cluster must contain at least this fraction of all
# non-noise points to be treated as a meaningful split.
MIN_CLUSTER_FRACTION = 0.10

# Also prevent tiny absolute clusters from being considered.
MIN_CLUSTER_POINTS = 100


def load_xyz(path):
    with np.load(path) as data:

        print(
            f"{path.name} keys:",
            list(data.keys()),
        )

        if "xyz" not in data:
            return None

        xyz = np.asarray(
            data["xyz"],
            dtype=np.float64,
        )

    xyz = xyz.reshape(-1, 3)

    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.any(xyz != 0, axis=1)
    )

    return xyz[valid]


def run_dbscan(xyz):

    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        xyz
    )

    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=False,
        )
    )

    return cloud, labels


def analyse_clusters(labels):

    valid_labels = labels[
        labels >= 0
    ]

    if len(valid_labels) == 0:
        return []

    unique_labels, counts = np.unique(
        valid_labels,
        return_counts=True,
    )

    total_clustered_points = counts.sum()

    clusters = []

    for label, count in zip(
        unique_labels,
        counts,
    ):
        fraction = (
            count / total_clustered_points
        )

        meaningful = (
            count >= MIN_CLUSTER_POINTS
            and
            fraction >= MIN_CLUSTER_FRACTION
        )

        clusters.append(
            {
                "label": int(label),
                "points": int(count),
                "fraction": float(fraction),
                "meaningful": meaningful,
            }
        )

    clusters.sort(
        key=lambda x: x["points"],
        reverse=True,
    )

    return clusters


def save_cluster_plys(
    xyz,
    labels,
    clusters,
    output_dir,
    stem,
):
    mask_output_dir = (
        output_dir / stem
    )

    mask_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for cluster in clusters:

        if not cluster["meaningful"]:
            continue

        label = cluster["label"]

        cluster_xyz = xyz[
            labels == label
        ]

        cloud = o3d.geometry.PointCloud()

        cloud.points = (
            o3d.utility.Vector3dVector(
                cluster_xyz
            )
        )

        path = (
            mask_output_dir
            / f"cluster_{label:02d}.ply"
        )

        o3d.io.write_point_cloud(
            str(path),
            cloud,
        )


def create_debug_cloud(
    xyz,
    labels,
):
    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )

    colours = np.zeros(
        (len(xyz), 3),
        dtype=np.float64,
    )

    valid_labels = labels >= 0

    if np.any(valid_labels):

        max_label = labels[
            valid_labels
        ].max()

        # Different deterministic colours for
        # different clusters.
        for label in range(
            max_label + 1
        ):
            mask = labels == label

            # Simple HSV-based colour generation
            hue = (
                label * 0.61803398875
            ) % 1.0

            import colorsys

            colour = colorsys.hsv_to_rgb(
                hue,
                0.8,
                1.0,
            )

            colours[mask] = colour

    # Noise = grey
    colours[
        labels < 0
    ] = [0.3, 0.3, 0.3]

    cloud.colors = (
        o3d.utility.Vector3dVector(
            colours
        )
    )

    return cloud


def main():

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    npz_paths = sorted(
        FUSED_DIRECTORY.glob("*.npz")
    )

    if not npz_paths:
        raise RuntimeError(
            f"No NPZ files found in "
            f"{FUSED_DIRECTORY}"
        )

    print()
    print(
        f"Found {len(npz_paths)} NPZ files."
    )

    print()
    print(
        "========================================"
    )
    print("3D SPLIT ANALYSIS")
    print(
        "========================================"
    )

    for path in npz_paths:

        print()
        print(
            "----------------------------------------"
        )

        print(path.name)

        xyz = load_xyz(path)

        if xyz is None:
            print(
                "Skipped: no 'xyz' key."
            )
            continue

        if len(xyz) < DBSCAN_MIN_POINTS:
            print(
                f"Skipped: only {len(xyz)} points."
            )
            continue

        print(
            "Valid XYZ points:",
            len(xyz),
        )

        cloud, labels = run_dbscan(
            xyz
        )

        clusters = analyse_clusters(
            labels
        )

        noise_count = np.count_nonzero(
            labels < 0
        )

        meaningful_clusters = [
            c
            for c in clusters
            if c["meaningful"]
        ]

        print(
            "Noise points:",
            noise_count,
        )

        for cluster in clusters:

            status = (
                "KEEP"
                if cluster["meaningful"]
                else "SMALL"
            )

            print(
                f"Cluster {cluster['label']:2d}: "
                f"{cluster['points']:6d} points "
                f"({cluster['fraction'] * 100:5.1f}%) "
                f"{status}"
            )

        print()

        if len(meaningful_clusters) == 0:

            print(
                "RESULT: no reliable 3D cluster"
            )

        elif len(meaningful_clusters) == 1:

            print(
                "RESULT: KEEP AS ONE OBJECT"
            )

        else:

            print(
                f"RESULT: POSSIBLE SPLIT "
                f"({len(meaningful_clusters)} objects)"
            )

        save_cluster_plys(
            xyz,
            labels,
            clusters,
            OUTPUT_DIRECTORY,
            path.stem,
        )

        # Save a coloured debug cloud
        debug_cloud = create_debug_cloud(
            xyz,
            labels,
        )

        debug_path = (
            OUTPUT_DIRECTORY
            / f"{path.stem}_clusters.ply"
        )

        o3d.io.write_point_cloud(
            str(debug_path),
            debug_cloud,
        )


if __name__ == "__main__":
    main()