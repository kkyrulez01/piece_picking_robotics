#!/usr/bin/env python3

import numpy as np
import torch

from sim_suction_network import SimSuctionModel


REPOSITORY_PATH = (
    "/home/support/unseen_sku_ws/"
    "external/Sim-Suction-API"
)

CHECKPOINT_PATH = (
    "/home/support/unseen_sku_ws/"
    "external/Sim-Suction-API/"
    "Sim-Suction-Pointnet/models/"
    "MV_PCL_1550_500.model"
)


def create_test_features():
    random_generator = np.random.default_rng(7)

    xyz = random_generator.normal(
        size=(5120, 3)
    ).astype(np.float32)

    # Normalize XYZ into a unit sphere, following the
    # repository's inference preprocessing.
    xyz -= np.mean(
        xyz,
        axis=0,
        keepdims=True,
    )

    maximum_radius = np.max(
        np.linalg.norm(
            xyz,
            axis=1,
        )
    )

    xyz /= maximum_radius

    normals = random_generator.normal(
        size=(5120, 3)
    ).astype(np.float32)

    normal_lengths = np.linalg.norm(
        normals,
        axis=1,
        keepdims=True,
    )

    normals /= np.maximum(
        normal_lengths,
        1e-8,
    )

    return np.concatenate(
        (xyz, normals),
        axis=1,
    )


def main():
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    model = SimSuctionModel(
        repository_path=REPOSITORY_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
    )

    features = create_test_features()
    scores = model.predict_scores(features)

    print("Points scored:", len(scores))
    print("Minimum score:", float(scores.min()))
    print("Maximum score:", float(scores.max()))
    print("Mean score:", float(scores.mean()))
    print("All scores finite:", np.isfinite(scores).all())

    assert scores.shape == (5120,)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)

    print("Pretrained Sim-Suction inference passed.")


if __name__ == "__main__":
    main()