from pathlib import Path
import cv2
import yaml

def load_config(path):
    config_path = Path(path).expanduser()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8",) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {config_path}")

    return config

def load_rgb(path):
    """
    Load a BGR image for offline testing.
    """

    rgb_path = Path(path).expanduser().resolve()
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(
            f"Could not read RGB image: {rgb_path}"
        )

    return image