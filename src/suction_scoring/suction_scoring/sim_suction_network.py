from pathlib import Path
import sys

import numpy as np
import torch

class SimSuctionModel:
    def __init__(self, repository_path, checkpoint_path, device="cuda"):
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.pointnet_path = self.repository_path / "Sim-Suction-Pointnet"
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not self.pointnet_path.exists():
            raise FileNotFoundError(
                "Sim-Suction-Pointnet directory not found: "
                f"{self.pointnet_path}"
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if str(self.pointnet_path) not in sys.path:
            sys.path.insert(0, str(self.pointnet_path))

        # Check if CUDA available
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not accessible.")

        self.device = torch.device(device)

        # Import after adding Sim-Suction-Pointnet to sys.path.
        # torch has already been imported, so libc10.so is loaded.
        from sim_suction_model.sim_suction_pointnet import ScoreNet

        # Instantiate model in evaluation mode
        self.model = ScoreNet(training=False).to(self.device)
        self._load_checkpoint()
        self.model.eval()

    def _load_checkpoint(self):
        loaded_checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        if isinstance(loaded_checkpoint, torch.nn.Module):
            state_dict = loaded_checkpoint.state_dict()

        elif (isinstance(loaded_checkpoint, dict) and "state_dict" in loaded_checkpoint):
            state_dict = loaded_checkpoint["state_dict"]

        elif isinstance(loaded_checkpoint, dict):
            state_dict = loaded_checkpoint

        else:
            raise TypeError("Unsupported checkpoint type.")

        # Remove prefixes produced by nn.DataParallel.
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned_state_dict[key] = value

        self.model.load_state_dict(cleaned_state_dict,strict=True)
        print("Sim-Suction checkpoint loaded successfully.")

    def predict_scores(self, features):
        """
        Run Sim-Suction on an N x 6 array:
            [x, y, z, normal_x, normal_y, normal_z]
        
        x,y,z should be normalized. Normals should be unit vectors.
        """
        features = np.asarray(features, dtype=np.float32)

        # Point cloud data should be a 2D Matrix (N x 6)
        if features.ndim != 2:
            raise ValueError("Features should be a 2D array of size N x 6.")

        # Point cloud columns should have 6 channels (x,y,z,n_x,n_y,n_z)
        if features.shape[1] != 6:
            raise ValueError(f"Expected features with shape N x 6, received {features.shape}")

        if features.shape[0] < 5120:
            raise ValueError(
                "Sim-Suction input must contain at least "
                "5,120 points after sampling or padding."
            )

        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or infinite values.")

        input_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            _, output_scores = self.model.extrat_featurePN2(
                input_tensor.transpose(1, 2).contiguous()
            )

        scores = output_scores[0, :, 0].detach().cpu().numpy()

        return scores