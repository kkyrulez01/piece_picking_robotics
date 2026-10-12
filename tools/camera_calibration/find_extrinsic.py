import os
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R

from calibration_board import Board
from calibration_folder_manager import CalibrationFolderManager
from common import calculate_mean_reprojection_error

# Need to pass in the calibration folder path as positional arg when running script
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("calibration_folder_path")
args = parser.parse_args()  # <-- parse the command line arguments
calibration_folder_path = args.calibration_folder_path


calibration_image_folder_path = os.path.join(calibration_folder_path, "images")
reprojection_results = os.path.join(calibration_folder_path, "reprojections")
if not os.path.exists(reprojection_results):
    os.mkdir(reprojection_results)

# load camera parameters
data = np.load(os.path.join(calibration_folder_path, "camera_parameters.npz"), allow_pickle=True)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

# load initial Rz acute angle and translation
data = np.load(os.path.join(calibration_folder_path, "board_alignment.npz"), allow_pickle=True)
acute_angle = data['Rz_acute']
t_board2tcp = data['t_board2tcp']

BDB6 = Board()
folder = CalibrationFolderManager(calibration_folder_path)
            

def sort_list_by_error(errors, to_be_sorted, reverse=False):
    sorted_pairs = sorted(zip(errors, to_be_sorted), key=lambda x: x[0], reverse=reverse)
    sorted_errors = [v for v, _ in sorted_pairs]
    sorted_items = [item for _, item  in sorted_pairs]
    return sorted_errors, sorted_items

def solvepnp_with_reprojection_error(objpoints, imgpoints, rvec_init=None, tvec_init=None, method=None):
    
    if (method is None) and (rvec_init is None) and (tvec_init is None):
        success, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, camera_matrix, dist_coeffs, 
                                           None, None, useExtrinsicGuess=False,  flags=cv2.SOLVEPNP_ITERATIVE)
    
    if (method is None) and (rvec_init is not None) and (tvec_init is not None):
        success, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, camera_matrix, dist_coeffs, 
                                           rvec_init, tvec_init, useExtrinsicGuess=True,  flags=cv2.SOLVEPNP_ITERATIVE)
    if method == 'ransac':
        success, rvec, tvec, inliers = cv2.solvePnPRansac(objpoints, imgpoints, camera_matrix, dist_coeffs)
        
    reprojection_error = calculate_mean_reprojection_error(objpoints, imgpoints, rvec, tvec, camera_matrix, dist_coeffs)

    return success, rvec, tvec, reprojection_error
      

def find_sorted_rvecs_tvecs(objpoints_list, imgpoints_list, filenames, rvec_init, tvec_init, method):
    """objpoints_list: M x N x 3, where M is the number of calibrating image
       imgpoints_list: M x N x 2
    """
    
    errors = []
    rvecs_baseframe = []
    tvecs_baseframe = []

    for (objpoints_baseframe, imgpoints) in zip(objpoints_list, imgpoints_list):
        
        success, rvec_pnp, tvec_pnp, reprojection_error = solvepnp_with_reprojection_error(objpoints_baseframe, imgpoints, 
                                                                                           rvec_init, tvec_init, method=method)
    
        rvecs_baseframe.append(rvec_pnp)
        tvecs_baseframe.append(tvec_pnp)

        errors.append(reprojection_error)


    sorted_errors, sorted_tvecs = sort_list_by_error(errors, tvecs_baseframe)
    _ ,            sorted_rvecs = sort_list_by_error(errors, rvecs_baseframe)

    # display error
    # for error, filename, rvec, tvec in zip(sorted_errors, sorted_text_filenames, sorted_rvecs, sorted_tvecs):
    #     print(filename, error)
    #     print(f"rvec: {rvec.flatten()}, tvec: {tvec.flatten()}")
    
    return sorted_rvecs, sorted_tvecs, sorted_errors


list_of_imgpoints, list_of_depths, list_of_tcp_pose, filenames, image_paths, image_depth_paths = folder.get_calibration_data()

# compute total_errors when using: each list below has 4 datapoints, 1 for each offset
total_errors = []
rvec_bests = []
tvec_bests = []
list_of_list_of_objpoints_base = []
quadrant_offsets = [0, 90, 180, 270]

for angle in quadrant_offsets:
    
    # recompute the objpoints in robot coordinates for each quadrant: [0,90,180,270] to determine the best reprojection
    list_of_objpoints_base = []
    
    # calculate R_board2tcp
    R_board2tcp = R.from_euler('z', acute_angle + angle, degrees=True).as_matrix()  # 3x3 rotation matrix
    
    for imgpoints, tcp_pose, filename, image_path, image_depth_path in zip(list_of_imgpoints, list_of_tcp_pose, filenames, image_paths, image_depth_paths):
        
        # calculate R_tcp2base and t_tcp2base
        x, y, z, Rx, Ry, Rz = tcp_pose
        R_tcp2base = R.from_euler('xyz', [Rx, Ry, Rz]).as_matrix()
        t_tcp2base = np.array([[x], [y], [z]]) 

        # compute the board circles in terms robot base coordinates
        objpoints_base = BDB6.compute_worldpoints(R_tcp2base @ R_board2tcp, t_tcp2base + R_tcp2base @ t_board2tcp)
        list_of_objpoints_base.append(objpoints_base)

    # convert to np.float 32 for the function
    list_of_objpoints_base = [np.array(pts, dtype=np.float32) for pts in list_of_objpoints_base]
    list_of_imgpoints  = [np.array(pts, dtype=np.float32) for pts in list_of_imgpoints]

    # solve pnp for each of the image using objpoints_base and determine the best extrinsic
    sorted_rvecs, sorted_tvecs, sorted_errors = find_sorted_rvecs_tvecs(list_of_objpoints_base, list_of_imgpoints, filenames, None, None, None)
    rvec_best = sorted_rvecs[0]
    tvec_best = sorted_tvecs[0]
    
    rvec_bests.append(rvec_best)
    tvec_bests.append(tvec_best)
    
    # use the rvec_best, tvec_best to get the total reprojection error across all poses
    total_error = 0
    for object_points, image_points in zip(list_of_objpoints_base, list_of_imgpoints):
        mean_error =  calculate_mean_reprojection_error(object_points, image_points, rvec_best, tvec_best, camera_matrix, dist_coeffs)
        total_error = total_error + mean_error

    total_errors.append(total_error)
    list_of_list_of_objpoints_base.append(list_of_objpoints_base)
    #print(f"tcp2board Rz: {acute_angle} + {angle} -> total error: {total_error}\n\n")


# Print results

# Best Extrinsic
_, sorted_rvec_bests = sort_list_by_error(total_errors, rvec_bests)
_, sorted_tvec_bests = sort_list_by_error(total_errors, tvec_bests)
_, sorted_list_of_list_of_objpoints_base = sort_list_by_error(total_errors, list_of_list_of_objpoints_base)
_, sorted_quadrant_offsets = sort_list_by_error(total_errors, quadrant_offsets)

rvec_best = sorted_rvec_bests[0]
tvec_best = sorted_tvec_bests[0]
list_of_objpoints_base = sorted_list_of_list_of_objpoints_base[0]

R_world2cam  = R.from_rotvec(rvec_best.flatten()).as_matrix()  
t_world2cam = tvec_best
print("R_world2cam:\n", R_world2cam)
# print("t_world2cam:\n", t_world2cam.flatten() )

# Rz 
Rz_angle =  acute_angle + sorted_quadrant_offsets[0]
_, quadrant_offsets = sort_list_by_error(total_errors, quadrant_offsets)
print(f"Rz: {Rz_angle}")

# Save Extrinsic
np.savez(os.path.join(calibration_folder_path,"extrinsic.npz"), Rz_angle=Rz_angle, R_world2cam=R_world2cam, t_world2cam=t_world2cam, t_board2tcp=t_board2tcp)
print("\nextrinsic.npz saved.")
