import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        default="/home/support/unseen_sku_ws/data/mecheye_pointclouds",
    )

    return parser.parse_known_args

def main():
    args = parse_arguments()

    data = np.load(args.file)

    print("Keys:", data.files)
    print("XYZ:", data["xyz"].shape, data["xyz"].dtype)
    print("RGB:", data["rgb"].shape, data["rgb"].dtype)
    print("Frame:", data["frame_id"])