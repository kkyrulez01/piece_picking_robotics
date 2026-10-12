import os
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R
from calibration_board import Board
from calibration_folder_manager import CalibrationFolderManager

# Need to pass in the calibration folder path as positional arg when running script
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("calibration_folder_path")
args = parser.parse_args()  # <-- parse the command line arguments
calibration_folder_path = args.calibration_folder_path
folder = CalibrationFolderManager(calibration_folder_path)

# get image size for the algorithm later
list_of_imgpoints, _, _, filenames, image_paths, _ = folder.get_calibration_data()
im = cv2.imread(image_paths[0])
height, width, channels = im.shape

exclude = []
# exclude = [6,7,8,9,10,11,21,22,23,24,25,26,36,37,38,39,40,41] #rotations
# exclude = [6,7,8,9,21,22,23,24,36,37,38,39] #rotations
# exclude = [6,7,8,9,10,11,36,37,38,39,40,41] #rotations

BDB6 = Board()
Board_coords = BDB6.compute_board_coordinates()
list_of_objpoints_board = np.stack([Board_coords] * len(list_of_imgpoints)) # N*20*3

# convert to np.float 32 for the function
list_of_objpoints_board = [np.array(pts, dtype=np.float32) for pts in list_of_objpoints_board]
list_of_imgpoints  = [np.array(pts, dtype=np.float32) for pts in list_of_imgpoints]

# First calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    list_of_objpoints_board, list_of_imgpoints, (width, height),
    None, None, flags=None, criteria = None
)
# Print results
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.flatten())
print("\n")


# Save calibration results
np.savez(os.path.join(calibration_folder_path,"camera_parameters.npz"), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("\ncamera_parameters.npz saved")
