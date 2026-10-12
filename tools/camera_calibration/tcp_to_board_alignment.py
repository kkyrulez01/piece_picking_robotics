import numpy as np
import cv2


class TCPtoBoardAlignment:
    def __init__(self, tcp_poses, imgpoints, objpoints, camera_matrix, dist_coeffs):
        """
        Compute the acute angle Rz offset between the TCP axis and the Board axis
        Parameters:
            tcp_poses : list of 4x4 np.array
                Absolute poses of TCP (robot end-effector to robot base)
            imgpoints : list of (Nx2) np.array
                2D image points per view (from detectCirclesGrid, etc.)
            objpoints : list of (Nx3) np.array
                Corresponding 3D object points per view (in board frame)
            camera_matrix : 3x3 np.array
                Intrinsic camera matrix
            dist_coeffs : (1x5) or (1x8) np.array
                Distortion coefficients
        """
        self.tcp_poses = tcp_poses
        self.imgpoints = imgpoints
        self.objpoints = objpoints
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        
        self.R_tcp2base_matrices = []
        self.R_tcp_motion = []    # Relative motions of the TCP (A_i)
        self.t_tcp_motion = []
        self.R_board_motion = []  # Relative motions of the board (B_i)
        self.t_board_motion = []
        
        # self.
        
    def convert_tcp_poses_to_matrices(self):
        """
        Convert a list of [X, Y, Z, Rx, Ry, Rz] into 4x4 transformation matrices.
        
        Parameters:
            tcp_poses : list of list or np.array
                Each element is [X, Y, Z, Rx, Ry, Rz], where:
                    - X, Y, Z are translations in mm
                    - Rx, Ry, Rz are rotation vector components in radians

        Returns:
            list of 4x4 np.array
                Homogeneous transformation matrices.
        """
        transformations = []
        for pose in self.tcp_poses:
            x, y, z, Rx, Ry, Rz = pose
            
            rotation = R.from_euler('xyz', [Rx, Ry, Rz])
            rot_matrix = rotation.as_matrix()

            # Build 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = [x, y, z]
            transformations.append(T)
        
        self.R_tcp2base_matrices =  transformations

    def compute_A_motions(self):
        """Compute relative TCP motions A_i from absolute TCP poses."""
        self.R_tcp_motion = []
        self.t_tcp_motion = []
        
        for i in range(len(self.tcp_poses) - 1):
            A_i = np.linalg.inv(self.R_tcp2base_matrices[i]) @ self.R_tcp2base_matrices[i + 1]
            R_A = A_i[:3, :3]
            t_A = A_i[:3, 3]
            self.R_tcp_motion.append(R_A)
            self.t_tcp_motion.append(t_A)

    def compute_B_motions(self):
        """Compute relative board motions B_i using solvePnP and 2D-3D correspondences."""
        cam_to_board_poses = []

        for imgp, objp in zip(self.imgpoints, self.objpoints):
            success, rvec, tvec = cv2.solvePnP(
                objp, imgp, self.camera_matrix, self.dist_coeffs
            )
            if not success:
                raise RuntimeError("solvePnP failed on one of the views.")
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            cam_to_board_poses.append(T)

        self.R_board_motion = []
        self.t_board_motion = []

        for i in range(len(cam_to_board_poses) - 1):
            B_i = np.linalg.inv(cam_to_board_poses[i]) @ cam_to_board_poses[i + 1]
            R_B = B_i[:3, :3]
            t_B = B_i[:3, 3]
            self.R_board_motion.append(R_B)
            self.t_board_motion.append(t_B)


    def find_offset(self):
        """Run OpenCV hand-eye calibration using the Tsai-Lenz method."""
        if not (self.R_tcp_motion and self.R_board_motion):
            raise RuntimeError("Must compute A and B motions before finding the offset.")

        R_board2tcp, t_board2tcp = cv2.calibrateHandEye(
            self.R_tcp_motion, self.t_tcp_motion,
            self.R_board_motion, self.t_board_motion,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        return R_board2tcp, t_board2tcp
    
        
    def to_acute_angle(self, angle):
        # Normalize the angle to be between 0 and 360
        angle = angle % 360

        # Determine the acute angle based on the quadrant
        if angle <= 90:
            return angle
        
        # Second Quadrant
        elif angle <= 180:
            return angle - 90
        # Third
        elif angle <= 270:
            return angle - 180
        # Fourth
        else:
            return angle - 270

        
        
if __name__ == "__main__":
    
    import pandas as pd
    import os
    import random
    from calibration_board import Board
    from scipy.spatial.transform import Rotation as R
    from calibration_folder_manager import CalibrationFolderManager

    
   # Need to pass in the calibration folder path as positional arg when running script
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_folder_path")
    args = parser.parse_args()  # <-- parse the command line arguments
    calibration_folder_path = args.calibration_folder_path

    manager = CalibrationFolderManager(calibration_folder_path)
    
    
    list_of_imgpoints, list_of_depths, list_of_tcp_pose, filenames, _, _ = manager.get_calibration_data()
    BDB6 = Board()
    Board_coords = BDB6.compute_board_coordinates()
    list_of_objpoints_board = np.stack([Board_coords] * len(list_of_imgpoints)) # N*20*3
    list_of_tcp_pose = np.array(list_of_tcp_pose)

    
    # load camera parameters
    data = np.load(os.path.join(calibration_folder_path, "camera_parameters.npz"), allow_pickle=True)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    # solve for R_board2tcp, t_board2tcp
    alignment_finder = TCPtoBoardAlignment(list_of_tcp_pose, list_of_imgpoints, list_of_objpoints_board, 
                                           camera_matrix, dist_coeffs)
    alignment_finder.convert_tcp_poses_to_matrices()
    alignment_finder.compute_A_motions()
    alignment_finder.compute_B_motions()
    R_board2tcp, t_board2tcp = alignment_finder.find_offset()
    
    # Extract the Rz value in rad and degree
    rot = R.from_matrix(R_board2tcp)
    roll, pitch,  yaw = rot.as_euler('xyz') 
    print(f"\nRx rotation: {roll} rad, {np.degrees(roll)} deg.")
    print(f"Ry rotation: {pitch} rad, {np.degrees(pitch)} deg.")
    print(f"Rz rotation: {yaw} rad, {np.degrees(yaw)} deg.")
    acute_angle = alignment_finder.to_acute_angle(np.degrees(yaw))
    print(f"\nOriginal Rz: {np.degrees(yaw)}° -> Acute: {acute_angle}°")
    print(f"Translations (mm): {t_board2tcp.flatten()}\n")
    
    # Save
    np.savez(os.path.join(calibration_folder_path,"board_alignment.npz"), Rz_acute=acute_angle, t_board2tcp=t_board2tcp)
    print("\nboard_alignment.npz saved.")
