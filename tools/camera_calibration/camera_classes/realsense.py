import cv2
import numpy as np
import pyrealsense2 as rs

class D45X:
    def __init__(self, camera_ip):
        # Kept only so this class can still be used by:
        #
        # cam = Camera(camera_ip)
        #
        # RealSense D45X cameras are USB cameras,
        # so camera_ip is not actually used.
        self.camera_ip = None

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.profile = None
        self.device = None

        self.width = 1280
        self.height = 720
        self.fps = 30

        self.camera_matrix = None
        self.dist_coeffs = None

        self.align = None
        self.depth_scale = None
        self.cached_depth_map = None

    def capture_frame(self):
        frames = self.pipeline.wait_for_frames()

        # Align depth into RGB/color coordinate system
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            raise RuntimeError("Failed to obtain RealSense color frame.")
        if not depth_frame:
            raise RuntimeError("Failed to obtain RealSense aligned depth frame.")

        image2d = np.asanyarray(color_frame.get_data()).copy()
        raw_depth = np.asanyarray(depth_frame.get_data()).copy()

        # RealSense Z16 raw value × scale -> metres.
        #
        # Convert to mm here to keep the saved depth representation
        # consistent with the existing calibration pipeline.
        depth_map = raw_depth.astype(np.float32) * self.depth_scale * 1000.0

        # RealSense uses zero for invalid depth
        depth_map[raw_depth == 0] = 0.0

        return image2d, depth_map

    def capture_2d_image(self, filepath):
        image2d, depth_map = self.capture_frame()

        cv2.imwrite(filepath, image2d)
        # Cache depth from the SAME RGB/depth frameset.
        self.cached_depth_map = depth_map

        print("Capture and save the 2D image: {}".format( filepath))

    def capture_depth_map(self, filepath2):
        if self.cached_depth_map is None:
            _, self.cached_depth_map = self.capture_frame()

        cv2.imwrite(filepath2, self.cached_depth_map.astype(np.float32))

        print("Capture and save the depth map: {}".format(filepath2))
        self.cached_depth_map = None

    def get_intrinsic_dist_coeffs(self):
        # For D455f
        data = np.load("/home/support/unseen_sku_ws/tools/camera_calibration/manufacturer_camera_parameters/camera_parameters_D455f.npz")
        # For D456
        # data = np.load("/home/support/unseen_sku_ws/tools/camera_calibration/manufacturer_camera_parameters/camera_parameters_D456.npz")

        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        print("Camera Matrix:")
        print(self.camera_matrix)

        print("\nDistortion Coefficients:")
        print(self.dist_coeffs)

        return self.camera_matrix, self.dist_coeffs

    def setup_camera_capture_settings(self):
        depth_sensor = self.device.first_depth_sensor()

        # RealSense depth scale converts raw Z16 values to metres.
        self.depth_scale = depth_sensor.get_depth_scale()

        print(
            "\nRealSense depth scale:", self.depth_scale, "metres/raw-unit"
        )

        # Enable depth auto exposure if supported.
        try:
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

                print("Depth auto exposure: ON")

        except Exception as e:
            print("Unable to configure depth auto exposure:", e)

        # Enable IR emitter if supported.
        try:
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)

                print("Depth emitter: ON")

        except Exception as e:
            print("Unable to configure emitter:", e)

        # Find RGB sensor.
        for sensor in self.device.query_sensors():
            try:
                sensor_name = sensor.get_info(rs.camera_info.name)

                if "RGB" in sensor_name:
                    if sensor.supports(rs.option.enable_auto_exposure):
                        sensor.set_option(rs.option.enable_auto_exposure, 1)

                        print( "RGB auto exposure: ON")

            except Exception:
                pass

    def connect(self):
        # RGB image
        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps
        )

        # Depth image
        self.config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps
        )

        try:
            self.profile = self.pipeline.start( self.config)

        except Exception as e:
            print("Failed to connect to RealSense camera: ", e)
            return

        self.device = self.profile.get_device()

        device_name = self.device.get_info(rs.camera_info.name)
        serial_number = self.device.get_info(rs.camera_info.serial_number)

        print( f"Connected to camera: {device_name}")
        print(f"Serial number: {serial_number}")

        self.setup_camera_capture_settings()

        # Depth will be projected into RGB image coordinates.
        self.align = rs.align(rs.stream.color)

        # Let automatic exposure settle.
        print("Waiting for RealSense exposure to settle...")

        for i in range(30):
            self.pipeline.wait_for_frames()

        print("RealSense ready.")

if __name__ == "__main__":
    cam = D45X(None)

    cam.connect()
    cam.get_intrinsic_dist_coeffs()
    cam.capture_2d_image("realsense_test.png")
    cam.capture_depth_map("realsense_test_depth.tiff")