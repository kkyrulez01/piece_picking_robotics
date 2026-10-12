import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import math
import matplotlib.pyplot as plt

from calibration_board import Board
from calibration_folder_manager import CalibrationFolderManager
from common import image_to_cam_coordinate

# Need to pass in the calibration folder path as positional arg when running script
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("calibration_folder_path")
args = parser.parse_args()  # <-- parse the command line arguments
calibration_folder_path = args.calibration_folder_path


# load camera parameters
data = np.load(os.path.join(calibration_folder_path, "camera_parameters.npz"), allow_pickle=True)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

# load (base2cam ,tcp2board) rotation matrices and translation vectors
data = np.load(os.path.join(calibration_folder_path, "extrinsic.npz"), allow_pickle=True)
Rz_angle = data['Rz_angle']
R_world2cam = data['R_world2cam']
t_world2cam = data['t_world2cam']
t_board2tcp = data['t_board2tcp']
# Invert world-to-camera transform from solvePnP
R_cam2world = R_world2cam.T

# check shape:
# print(Rz_angle.shape)
# print(R_world2cam.shape)
# print(t_world2cam.shape)
# print(t_board2tcp.shape)

# convert to x0 for initialisation
rvec_cam2world = cv2.Rodrigues(R_cam2world)[0].flatten()  # shape (3,)
t_world2cam =  t_world2cam.flatten()
Rz_angle_rad = np.deg2rad(Rz_angle)
t_board2tcp = t_board2tcp.flatten()
x0 = np.concatenate([rvec_cam2world, t_world2cam, [Rz_angle_rad], t_board2tcp])

BDB6 = Board()
folder = CalibrationFolderManager(calibration_folder_path)

#exclude = [6,7,8,9, 21,22,23,24, 36,37,38,39]
# exclude = [36,37,38,39]
exclude =  []
list_of_imgpoints, list_of_depths, list_of_tcp_pose, filenames, image_paths, image_depth_paths = folder.get_calibration_data(exclude=exclude)

def compute_xyz_residuals(x):
    
    # Extrinsic
    rvec_cam2world = x[0:3]
    
    t_world2cam = x[3:6].reshape(3, 1)
    R_cam2world = cv2.Rodrigues(rvec_cam2world)[0]
    R_world2cam =  R_cam2world.T
    t_cam2world = -R_cam2world @ t_world2cam

    # TCP-to-board
    Rz_angle_rad = x[6]
    R_board2tcp = R.from_euler('z', Rz_angle_rad, degrees=False).as_matrix()
    t_board2tcp = np.array(x[7:10]).reshape(3,1)
    
    errors_reproj = []
    for imgpoints, depths, tcp_pose, in zip(list_of_imgpoints, list_of_depths, list_of_tcp_pose):
        
        # Compute R_tcp2base, t_tcp2base from pose data
        x, y, z, Rx, Ry, Rz = tcp_pose
        R_tcp2base = R.from_euler('xyz', [Rx, Ry, Rz]).as_matrix()
        t_tcp2base = np.array([[x], [y], [z]]) 
        
        # print(R_tcp2base.shape)
        # print(R_board2tcp.shape)
        # print(t_tcp2base.shape)
        # print(t_board2tcp.shape)
        
        objpoints_base = BDB6.compute_worldpoints(R_tcp2base @ R_board2tcp,  t_tcp2base + R_tcp2base @ t_board2tcp)
        
        for idx, (imgpoint, depth, objpoint_base) in enumerate(zip(imgpoints, depths, objpoints_base)):

            if not np.isnan(depth):
                
                if np.isinf(depth):
                    print("infinite depth value found")
                    continue
                
                if np.isclose(depth, 0.0, atol=1e-8):
                    print("depth value 0 found")
                    continue
       
                X_cam = image_to_cam_coordinate(imgpoint, depth, camera_matrix, dist_coeffs)
                X_world = R_cam2world @ X_cam + t_cam2world
                #X_cam = R_world2cam @ X_world + t_world2cam
                                
                if X_world is None:
                    print(depth)
                    print("Invalid world point computed")
                
                x1, y1, z1 =  X_world
                x2, y2, z2 = objpoint_base
               
                # N,3 array
                errors_reproj.append([x1-x2, y1-y2, z1-z2])
                
                # N,2 array
                #errors_reproj.append([x1-x2, y1-y2])

    
    return np.array(errors_reproj).flatten()
                 
# Run the optimization
res = least_squares(
    compute_xyz_residuals,
    x0, 
    args=(),
    method='lm',
    max_nfev=10000,               
    xtol=1e-10, ftol=1e-10, gtol=1e-12,
    verbose=2  
)

# Print the optimised parameters
print("\n Results:")

Rz_angle_opt = np.rad2deg(res.x[6])
print(f"Rz angle:{Rz_angle_opt}")

t_board2tcp_opt = res.x[7:10].reshape(3,1)
print(f"Translations tcp2board: {t_board2tcp_opt}")

t_world2cam_opt = res.x[3:6].reshape(3,1)
print(f"Translations world2cam: {t_world2cam_opt}")

R_cam2world_opt = R.from_rotvec(res.x[0:3]).as_matrix() 
print("Rotation cam2world:")
print(R_cam2world_opt)

t_cam2world_opt = -R_cam2world_opt @ t_world2cam_opt
print(f"Translations cam2world: {t_cam2world_opt}")

R_world2cam_opt  = R_cam2world_opt.T
print("Rotation world2cam:")
print(R_world2cam_opt)


# Save optimised parameters
np.savez(os.path.join(calibration_folder_path,"optimised.npz"), 
         Rz_angle=Rz_angle_opt, 
         R_world2cam=R_world2cam_opt, 
         t_world2cam=t_world2cam_opt, 
         t_board2tcp=t_board2tcp_opt,
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)

print("\noptimised.npz saved")