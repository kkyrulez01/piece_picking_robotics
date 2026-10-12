import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from calibration_board import Board
from calibration_folder_manager import CalibrationFolderManager
from common import calculate_mean_reprojection_error, image_to_cam_coordinate

# Need to pass in the calibration folder path as positional arg when running script
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("calibration_folder_path")
parser.add_argument("reprojection_folder_name")
parser.add_argument("extrinsic_name")

args = parser.parse_args()  # <-- parse the command line arguments
calibration_folder_path = args.calibration_folder_path
reprojection_folder_name = args.reprojection_folder_name
extrinsic_name = args.extrinsic_name

reprojection_folder_path = os.path.join(calibration_folder_path, reprojection_folder_name)
calibration_image_folder_path = os.path.join(calibration_folder_path, "images")
if not os.path.exists(reprojection_folder_path):
    os.mkdir(reprojection_folder_path)


# load (base2cam ,tcp2board) rotation matrices and translation vectors 
data = np.load(os.path.join(calibration_folder_path, extrinsic_name), allow_pickle=True)

Rz_angle = data['Rz_angle']
t_board2tcp = data['t_board2tcp']

R_world2cam = data['R_world2cam']
t_world2cam = data['t_world2cam']

R_cam2world = R_world2cam.T
t_cam2world = - R_cam2world @ t_world2cam

R_board2tcp = R.from_euler('z', Rz_angle, degrees=True).as_matrix()  # 3x3 rotation matrix

# check value:
# print(Rz_angle)
# print(R_world2cam)
# print(t_world2cam)
# print(t_board2tcp)

# # check shape:
# print(Rz_angle.shape)
# print(R_world2cam.shape)
# print(t_world2cam.shape)
# print(t_board2tcp.shape)

# load camera parameters
data = np.load(os.path.join(calibration_folder_path, "camera_parameters.npz"), allow_pickle=True)
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

BDB6 = Board()
folder = CalibrationFolderManager(calibration_folder_path)

list_of_imgpoints, list_of_depths, list_of_tcp_pose, filenames, image_paths, image_depth_paths = folder.get_calibration_data()
list_of_objpoints_base = []

errors_xyz_reproj_actual = []
errors_xyz_reproj_true = []

x_errors_actual = []
y_errors_actual = []
z_errors_actual = []

for imgpoints, depths, tcp_pose, filename, image_path, image_depth_path in zip(list_of_imgpoints, list_of_depths, list_of_tcp_pose, filenames, image_paths, image_depth_paths):
        
    # Compute base coordinates
    x, y, z, Rx, Ry, Rz = tcp_pose
    R_tcp2base = R.from_euler('xyz', [Rx, Ry, Rz]).as_matrix()
    t_tcp2base = np.array([[x], [y], [z]]) 
    objpoints_base = BDB6.compute_worldpoints(R_tcp2base @ R_board2tcp,  t_tcp2base + R_tcp2base @ t_board2tcp)
    list_of_objpoints_base.append(objpoints_base)
    
    for idx, (imgpoint, depth, objpoint_base) in enumerate(zip(imgpoints, depths, objpoints_base)):
        
        if not np.isnan(depth):
            # if np.isinf(depth):
            #     print("infinie value found")

            # use sensor depth
            cam_point_estimated = image_to_cam_coordinate(imgpoint, depth, camera_matrix, dist_coeffs)
            world_point_estimated = R_cam2world @ cam_point_estimated + t_cam2world

            x1, y1, z1 =  world_point_estimated.flatten()
            x2, y2, z2 = objpoint_base
            distance_xyz = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            
            errors_xyz_reproj_actual.append(distance_xyz)
            x_errors_actual.append(np.sqrt((x2 - x1)**2))
            y_errors_actual.append(np.sqrt((y2 - y1)**2))
            z_errors_actual.append(np.sqrt((z2 - z1)**2))

            
            # use true depth(objpoints z)
            objpoint_base =  objpoint_base.reshape(3, 1) 
            X_cam = R_world2cam @ objpoint_base + t_world2cam
            depth = X_cam[-1][0]
            
            cam_point_estimated = image_to_cam_coordinate(imgpoint, depth, camera_matrix, dist_coeffs)
            world_point_estimated = R_cam2world @ cam_point_estimated + t_cam2world
            x1, y1, z1 =  world_point_estimated
            x2, y2, z2 = objpoint_base
            distance_xyz = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            
            errors_xyz_reproj_true.append(distance_xyz)
                
            
# Plot histogram
ticks = np.arange(0, 50, 1)
plt.xticks(ticks)
plt.hist(errors_xyz_reproj_actual, bins=50, edgecolor='black')
plt.title("Error Distribution")
plt.xlabel("Error Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Summary 
print("\nSummary:")
for percentile in [50,95,99,100]:
    print(f"{percentile}th percentile {round(np.percentile(errors_xyz_reproj_actual, percentile), 3)} mm")
print("Number of datapoints:", len(errors_xyz_reproj_actual))

print("\nSummary (by components):")
for percentile in [50,95,99,100]:
    print(f"\n{percentile}th percentile (x): {round(np.percentile(x_errors_actual, percentile), 3)} mm")
    print(f"{percentile}th percentile (y): {round(np.percentile(y_errors_actual, percentile), 3)} mm")
    print(f"{percentile}th percentile (z): {round(np.percentile(z_errors_actual, percentile), 3)} mm")

# print("\nSummary Using true Z:")
# for percentile in [50,95,99,100]:
#     print(f"{percentile}th percentile {round(np.percentile(errors_xyz_reproj_true, percentile), 3)} mm")
# print("Number of datapoints:", len(errors_xyz_reproj_true))

# Write the projected image for visualisation later
rvec_best, _ = cv2.Rodrigues(R_world2cam)
tvec_best    = t_world2cam
list_of_objpoints_base = [np.array(pts, dtype=np.float32) for pts in list_of_objpoints_base]

for imgpoints, objpoints_base, filename, image_path, _ in zip(list_of_imgpoints, list_of_objpoints_base, filenames, image_paths, image_depth_paths):
    
    # read image
    img = cv2.imread(image_path)
    
    reprojections, _ = cv2.projectPoints(objpoints_base, rvec_best, tvec_best, camera_matrix, dist_coeffs)
    mean_error = calculate_mean_reprojection_error(objpoints_base, imgpoints, rvec_best, tvec_best, camera_matrix, dist_coeffs)
    print(f"{filename} mean error: {mean_error}")
    #print(f"shape of reprojections: {reprojections.shape}")
    
    # visualise the reprojection using the extrinsics computed using camera calibrate
    for p in reprojections:
        cv2.circle(img, tuple(np.round(p[0]).astype(int)), 4, (255, 0, 0), -1) # Blue
    
    # show the index of each point
    for idx, element in enumerate(imgpoints):
        #print(element)
        x = element[0]
        y = element[1]
        cv2.putText(img, str(idx), (int(x), int(y)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=1)
        
        cv2.circle(img, tuple(np.round((x,y)).astype(int)), 4, (0, 255, 0), -1) # Green
    
    cv2.imwrite(os.path.join(reprojection_folder_path, filename + ".png"), img)
