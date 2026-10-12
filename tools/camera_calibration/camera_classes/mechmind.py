import cv2
import numpy as np
from mecheye.shared import show_error
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, print_camera_intrinsics, print_camera_info

class MechmindCamera:
    
    def __init__(self, camera_ip):
        self.camera = Camera()
        self.intrinsics = CameraIntrinsics()
        self.camera_ip =  camera_ip

    def capture_2d_image(self, filepath):
        frame_2d = Frame2D()
        show_error(self.camera.capture_2d(frame_2d))
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image2d = frame_2d.get_gray_scale_image()
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            image2d = frame_2d.get_color_image()

        cv2.imwrite(filepath, image2d.data())
        print("Capture and save the 2D image: {}".format(filepath))
        
    def capture_depth_map(self, filepath2):
        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d))

        depth_map = frame3d.get_depth_map()
        cv2.imwrite(filepath2, depth_map.data())
        print("Capture and save the depth map: {}".format(filepath2))
        
    def get_intrinsic_dist_coeffs(self):
        show_error(self.camera.get_camera_intrinsics(self.intrinsics))
        #print_camera_intrinsics(self.intrinsics)
        
        # dist_coeffs
        dist = self.intrinsics.texture.camera_distortion
        k1 = dist.k1
        k2 = dist.k2
        p1 = dist.p1
        p2 = dist.p2
        k3 = dist.k3

        dist_coeffs = np.array([[k1, k2, p1, p2, k3]])
        
        # camera_matrix
        camera_matrix = self.intrinsics.texture.camera_matrix
        fx = camera_matrix.fx
        fy = camera_matrix.fy
        cx = camera_matrix.cx
        cy = camera_matrix.cy
        
        camera_matrix = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,   1]
        ])
        
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        np.savez("camera_parameters_mecheye.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        
    def setup_camera_capture_settings(self):
        exposure_time_3d_sequence = [0.5,1,3] # in ms
        exposure_time_2d = 600 # Default 150.0
        
        # Obtain the name of the currently selected user set. 
        current_user_set = self.camera.current_user_set()
        error, user_set_name = current_user_set.get_name()
        show_error(error)
        print("\ncurrent_user_set: " + user_set_name)

        # Set the exposure times for acquiring depth information.
        error = current_user_set.set_float_array_value(
            Scanning3DExposureSequence.name, exposure_time_3d_sequence)
        show_error(error)
        error, exposure_sequence = current_user_set.get_float_array_value(
            Scanning3DExposureSequence.name)
        show_error(error)
        print("\nThe 3D scanning exposure multiplier: {}".format(
            len(exposure_sequence)))
        for i in exposure_sequence:
            print("3D scanning exposure time: {}".format(i))
        
        # Set the exposure times for 2D
        exposure_mode_2d = Scanning2DExposureMode.Value_Timed
        error = current_user_set.set_enum_value(
            Scanning2DExposureMode.name, exposure_mode_2d)
        show_error(error)
        error = current_user_set.set_float_value(
            Scanning2DExposureTime.name, exposure_time_2d)
        show_error(error)
        
        error, exposure_mode_2d = current_user_set.get_enum_value_string(
            Scanning2DExposureMode.name)
        show_error(error)
        error, exposure_time_2d = current_user_set.get_float_value(
            Scanning2DExposureTime.name)
        show_error(error)
        print("2D scanning exposure mode enum: {} \n2D scanning exposure time: {}".
              format(exposure_mode_2d, exposure_time_2d))


        # Save all the parameter settings to the currently selected user set.
        show_error(current_user_set.save_all_parameters_to_device())

        
    def connect(self):
        camera_infos = self.camera.discover_cameras()
        device_found = False
        
        if len(camera_infos) == 0:
            print("No Mech-Eye Industrial 3D Cameras found.")
            return
        
        for i in range(len(camera_infos)):
            print("Mech-Eye device index :", i)
            print_camera_info(camera_infos[i])
            # print(camera_infos[i].ip_address)
            
            if camera_infos[i].ip_address == self.camera_ip:
                device_found = True
                error_status = self.camera.connect(camera_infos[i])
                if not error_status.is_ok():
                    # show_error(error_status)
                    print(f"Camera: {self.camera_ip} found but connection failed. Check PC Ethernet Port IP.")
                else:
                    print(f"Connected to camera: {self.camera_ip}")
                    self.setup_camera_capture_settings()

                break
        
        if not device_found:
            print(f"Camera: {self.camera_ip} not found.")
                

if __name__ == "__main__":
    #cam =  MechmindCamera("192.168.100.215")
    cam =  MechmindCamera("192.168.100.205")
    # cam = MechmindCamera("192.168.1.101")
    cam.connect()
    cam.get_intrinsic_dist_coeffs()
    cam.capture_2d_image("mecheye_test.png")
    cam.capture_depth_map("mecheye_test_depth.tiff")
    