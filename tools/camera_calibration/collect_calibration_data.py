import os
import time
from robot import Robot

# from camera_classes.mechmind import MechmindCamera as Camera
# from camera_classes.lucid import Helios2 as Camera
from camera_classes.realsense import D45X as Camera

from calibration_folder_manager import CalibrationFolderManager

camera_calibration_data_path = "/home/support/unseen_sku_ws/tools/camera_calibration/calibration_data"
calibration_folder_name = os.path.join(camera_calibration_data_path, "realsense_D455f_260922")

robot_port = 30001
robot_port_pose = 30001
pc_port = 30005
# Helios2: 192.168.1.130
# Mechmind: 192.168.1.230
# D45X: None
camera_ip = None
robot_ip = '192.168.1.122'
pc_ip = '192.168.1.100' 
x, y, z = -444.45, -100, 120                                                                       
# -1570, -25, -205 # for left camera: x, y, z = 100, -620, -100 for right camera:x, y, z = -180, 700, -100

delta_x, delta_y, delta_z, taper_xy = 0.180, 0.075, 0.05, 0.05
# delta_x, delta_y, delta_z, taper_xy = 0.15, 0.25, 0.2, 0.15
a, v = 0.05, 0.25
num_of_layers_pyramid = 3

TIME_DELAY_BTW_POSE =  1 # time delay between pose
x, y, z = x/1000, y/1000, z/1000
center = (x,y,z)
dRx_deg, dRy_deg, dRz_deg = 10, 10, 10  # rotation angles

robot = Robot(robot_ip)
manager = CalibrationFolderManager(calibration_folder_name)
cam = Camera(camera_ip)
cam.connect()


def compute_pyramid_points(center_x, center_y, center_z, delta_x, delta_y, delta_z, taper_xy):
    """
    delta are in metres.
    Computes a pyramid of 27 points organized in 3 layers.
    Each layer consists of a 3x3 grid with the same Z (height).

    Parameters:
    - center_x, center_y, center_z: coordinates of the center point of the bottom layer (in robot base axes)
    - delta_xy: spacing between adjacent points in X and Y direction
    - delta_z: spacing between layers (Z direction)

    Returns:
    - List of (X, Y, Z) tuples for all 27 points
    """
    points = []

    for layer in range(num_of_layers_pyramid):  # 3 layers
        z = center_z + layer * delta_z
        offset_x = delta_x * (1 - layer * taper_xy)  # Optional: reduce spread per layer
        offset_y = delta_y * (1 - layer * taper_xy)  # Optional: reduce spread per layer        for dx in [-offset, 0, offset]:
        for dx in [-offset_x, 0, offset_x]:
            x = center_x + dx
            for dy in [-offset_y, 0, offset_y]:
                y = center_y + dy
                points.append((x, y, z))
    
    return points

def generate_rotation_variations(Rx, Ry, Rz, dRx_deg, dRy_deg, dRz_deg):
    """
    Generate 6 rotation-only pose variations in radians.
    
    Args:
        Rx, Ry, Rz (float): Base rotations in radians.
        dRx_deg, dRy_deg, dRz_deg (float): Rotation deltas in degrees.
    
    Returns:
        List[List[float]]: Each pose as [Rx(rad), Ry(rad), Rz(rad)]
    """
    pi = 3.141592653589793

    def deg2rad(angle_deg):
        return angle_deg * (pi / 180)

    # Convert deltas to radians only
    dRx = deg2rad(dRx_deg)
    dRy = deg2rad(dRy_deg)
    dRz = deg2rad(dRz_deg)

    variations = [
        [Rx + dRx, Ry, Rz],
        [Rx - dRx, Ry, Rz],
        [Rx, Ry + dRy, Rz],
        [Rx, Ry - dRy, Rz],
        [Rx, Ry, Rz + dRz],
        [Rx, Ry, Rz - dRz],
    ]

    return variations


# Record the robot tcp pose at the beginning
robot_pose = robot.get_tcp_pose(pc_ip, pc_port, robot_port_pose)
Rx_st, Ry_st, Rz_st = robot_pose[3:]
print(f"Initial Rx, Ry, Rz = {Rx_st}, {Ry_st}, {Rz_st}")
# x_st, y_st, z_st = robot_pose[:3]
# print(f"Initial x,y,z = {x_st}, {y_st}, {z_st}")

# Move to starting position
robot.movel([*center, Rx_st, Ry_st, Rz_st], a, v, pc_ip, pc_port, robot_port) 

points = compute_pyramid_points(*center, delta_x, delta_y, delta_z, taper_xy)
rotated_poses = generate_rotation_variations(Rx_st, Ry_st, Rz_st, dRx_deg, dRy_deg, dRz_deg)

# Capture image and pose
for i, pt in enumerate(points, 1):
    
    # Translation
    x, y, z = pt[0], pt[1], pt[2]
    Rx, Ry, Rz = Rx_st, Ry_st, Rz_st
    p = [x, y, z, Rx, Ry, Rz]
    
    robot.movel(p, a, v, pc_ip, pc_port, robot_port)
    time.sleep(TIME_DELAY_BTW_POSE)

    filename, filename_depth = manager.get_next_filenames()

    # Take picture and save tcp pose from robot
    cam.capture_2d_image(os.path.join(calibration_folder_name, "images", filename))
    cam.capture_depth_map(os.path.join(calibration_folder_name, "images_depth", filename_depth))
    robot_pose = robot.get_tcp_pose(pc_ip, pc_port, robot_port_pose)
    manager.save_robot_pose(robot_pose, filename)
        
    # Rotations
    if pt[0] == center[0] and pt[1] == center[1]:
        
        for j, (Rx, Ry, Rz) in enumerate(rotated_poses,1):
            p = [pt[0], pt[1], pt[2], Rx, Ry, Rz]
            print(f"\n\nPoint {i}.{j}: {p}")
            
            robot.movel(p, a, v, pc_ip, pc_port, robot_port)
            time.sleep(TIME_DELAY_BTW_POSE)

            filename, filename_depth = manager.get_next_filenames()
            # Take picture and save 
            cam.capture_2d_image(os.path.join(calibration_folder_name, "images", filename))
            cam.capture_depth_map(os.path.join(calibration_folder_name, "images_depth", filename_depth))

            robot_pose = robot.get_tcp_pose(pc_ip, pc_port, robot_port_pose)
            manager.save_robot_pose(robot_pose, filename)
            
                

