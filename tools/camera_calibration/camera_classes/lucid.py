#!/usr/bin/env python3

import cv2
import numpy as np

from arena_api.system import system

class Helios2:
    def __init__(self, camera_ip):
        self.camera = None
        self.camera_ip = camera_ip

        self.camera_matrix = None
        self.dist_coeffs = None

        self.z_scale_mm = None
        self.z_offset_mm = None

        # Used to capture_2d_image() and capture_depth_map()
        # refer to the SAME Helios frame.
        self.cached_depth_map = None

    def capture_frame(self):
        """
        Capture one Coord3D_ABCY16 frame.

        Returns:
            intensity_image: uint8 grayscale image
            depth_map: float32 Z depth in millimetres
        """
        image_buffer = self.camera.get_buffer()

        try:
            height = image_buffer.height
            width = image_buffer.width
            bytes_per_pixel = int(image_buffer.bits_per_pixel / 8)

            if bytes_per_pixel != 8:
                raise RuntimeError(
                    "Expected Coord3D_ABCY16 (8 bytes/pixel), "
                    f"but got {image_buffer.bits_per_pixel} bits/pixel."
                )

            # Arena SDK -> NumPy
            raw_bytes = np.ctypeslib.as_array(image_buffer.pdata, shape=(height, width, bytes_per_pixel)).copy()

        finally:
            # Important: return Arena buffer after copying its data.
            self.camera.requeue_buffer(image_buffer)

        # Coord3D_ABCY16:
        #
        # channel 0 = X
        # channel 1 = Y
        # channel 2 = Z
        # channel 3 = intensity
        #
        # each channel = uint16
        raw_u16 = raw_bytes.view(np.uint16).reshape(height,width,4)

        raw_z = raw_u16[:, :, 2]
        intensity_u16 = raw_u16[:, :, 3]

        # Convert raw Helios Z value to physical Z in mm
        depth_map = raw_z.astype(np.float32) * self.z_scale_mm + self.z_offset_mm

        # Keep invalid depth as 0 so your existing calibration
        # code can reject zero-depth points.
        depth_map[raw_z == 0] = 0.0

        # Convert Helios intensity to an 8-bit image.
        # This is useful for blob detection.
        intensity_image = cv2.normalize(
            intensity_u16,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        return intensity_image, depth_map

    # collect_calibration_data.py calls capture_2d_image() and
    # capture_depth_map()

    def capture_2d_image(self, filepath):
        intensity_image, depth_map = self.capture_frame()
        cv2.imwrite(filepath, intensity_image)

        # Cache depth from SAME frame.
        self.cached_depth_map = depth_map

        print("Capture and save the 2D image: {}".format(filepath))

    def capture_depth_map(self, filepath2):
        if self.cached_depth_map is None:
            _, self.cached_depth_map = self.capture_frame()

        cv2.imwrite(filepath2, self.cached_depth_map.astype(np.float32))
        print("Capture and save the depth map: {}".format(filepath2))

        # Do not accidentally reuse this frame next time.
        self.cached_depth_map = None

    def get_intrinsic_dist_coeffs(self):
        # Load Helios2.npz file containing camera intrinsics
        data = np.load("/home/support/unseen_sku_ws/tools/camera_calibration/manufacturer_camera_parameters/camera_parameters_helios2.npz")

        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        print("Camera Matrix:")
        print(self.camera_matrix)

        print("\nDistortion Coefficients:")
        print(self.dist_coeffs)

        return self.camera_matrix, self.dist_coeffs

    def setup_camera_capture_settings(self):
        nodemap = self.camera.nodemap

        # Required to obtain:
        #
        # X, Y, Z, intensity
        nodemap["PixelFormat"].value = "Coord3D_ABCY16"

        # Recommended GigE settings.
        try:
            self.camera.tl_stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
        except Exception as e:
            print("Could not enable StreamAutoNegotiatePacketSize:", e)

        try:
            self.camera.tl_stream_nodemap["StreamPacketResendEnable"].value = True
        except Exception as e:
            print("Could not enable StreamPacketResendEnable:", e)

        # Use CoordinateC = Z.
        coordinate_selector = nodemap["Scan3dCoordinateSelector"]
        coordinate_selector.value = "CoordinateC"

        self.z_scale_mm = float(nodemap["Scan3dCoordinateScale"].value)
        self.z_offset_mm = float(nodemap["Scan3dCoordinateOffset"].value)

        print("\nHelios2 capture settings:")
        print("Pixel format: Coord3D_ABCY16")
        print("Z scale (mm):", self.z_scale_mm)
        print("Z offset (mm):", self.z_offset_mm)

        # Print current ToF operating settings.
        try:
            print("Operating mode:", nodemap["Scan3dOperatingMode"].value)
        except Exception:
            pass

        try:
            print("Exposure:", nodemap["ExposureTimeSelector"].value)
        except Exception:
            pass

        # If you later want to force an exposure:
        #
        # nodemap["ExposureTimeSelector"].value = "Exp1000Us"
        #
        # LUCID also supports values such as Exp250Us
        # depending on the Helios model/mode.
    
    def connect(self):
        camera_infos = system.device_infos

        if len(camera_infos) == 0:
            print("No LUCID cameras found.")
            return

        print("LUCID cameras found:")

        selected_camera_info = None

        for i, camera_info in enumerate(camera_infos):
            print("\nLUCID device index:", i)
            print(camera_info)

            # Arena SDK versions may expose IP information with
            # slightly different dictionary keys.
            for key, value in camera_info.items():
                if "ip" in key.lower():
                    print(f"{key}: {value}")

                    if str(value) == str(self.camera_ip):
                        selected_camera_info = camera_info

        # If only one LUCID camera exists, simply use it.
        if selected_camera_info is None and len(camera_infos) == 1:
            selected_camera_info = camera_infos[0]
            print("\nOnly one LUCID camera found. Using this device.")

        if selected_camera_info is None:
            print(f"Camera: {self.camera_ip} not found.")
            return

        self.camera = system.create_device(device_infos=selected_camera_info)[0]

        print("\nConnected to LUCID camera:",
              selected_camera_info.get("model", "Unknown")
        )

        self.setup_camera_capture_settings()
        self.camera.start_stream()

if __name__ == "__main__":
    ip_address = "192.168.1.100" # Change to Helios IP address
    cam = Helios2(ip_address)

    cam.connect()
    cam.get_intrinsic_dist_coeffs()

    cam.capture_2d_image("helios2_test.png")
    cam.capture_depth_map("helios2_test_depth.tiff")
