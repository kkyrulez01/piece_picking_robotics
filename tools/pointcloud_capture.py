import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
import cv2

from cv_bridge import CvBridge

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
from mecheye_ros_interface.srv import (
    CaptureColorImage,
    CaptureDepthMap,
    CapturePointCloud,
    SetEnumParameter, 
    SetFloatParameter, 
    SetFloatArrayParameter
)

def unpack_rgb(points, field_name, height, width):
    packed = np.asarray(points[field_name]).reshape(-1) # Flatten array 

    if packed.dtype.kind == "f":
        packed = np.ascontiguousarray(packed.astype(np.float32), copy=False).view(np.uint32)

    else:
        packed = packed.astype(np.uint32, copy=False)

    # 31-24 -> Alpha/Pad, 23-16->Red, 15-8->Green, 7-0->Blue
    # Shift until only the desired channel left
    red = ((packed >> 16) & 255).astype(np.uint8)
    green = ((packed >> 8) & 255).astype(np.uint8)
    blue = ((packed) & 255).astype(np.uint8)

    return np.stack((red, green, blue), axis=1).reshape(height, width, 3)

def convert_cloud(message):
    available_fields = {field.name for field in message.fields}

    required_fields = {"x", "y", "z"} # Only need x,y,z fields
    if not required_fields.issubset(available_fields):
        raise RuntimeError(f"Pointcloud does not contain XYZ.")

    colour_field = None

    if "rgb" in available_fields:
        colour_field = "rgb"
    elif "rgba" in available_fields:
        colour_field = "rgba"

    required_fields = ["x","y","z"]
    if colour_field is not None:
        required_fields.append(colour_field) # Final list of required fields

    points = point_cloud2.read_points(
        message,
        field_names=required_fields,
        skip_nans=False,
    )

    points = np.asarray(points)
    if points.dtype.names is None:
        raise RuntimeError("Expected structured NumPy point-cloud array.")

    height = int(message.height)
    width = int(message.width)
    expected_points = height * width

    if points.size != expected_points:
        raise RuntimeError(
            f"Expected {expected_points} points but received {points.size}"
        )

    points = points.reshape(-1)
    xyz = np.stack((points["x"],
                    points["y"],
                    points["z"],),
                    axis=1).astype(np.float32)
    
    xyz = xyz.reshape(height,width,3)

    rgb = None
    if colour_field is not None:
        rgb = unpack_rgb(
            points,
            colour_field,
            height,
            width,
        )

    return xyz, rgb

def find_next_capture_index(
    image_directory,
    depth_directory,
    pointcloud_directory,
    scene_prefix="scene"
):
    """
    Find the next available capture index based on existing:

        image_XXXX.png
        depth_XXXX.tiff
        scene_XXXX.npz

    Example:
        image_0012.png
        depth_0012.tiff
        scene_0012.npz

    -> next index = 13
    """
    directories_and_patterns = [
        (Path(image_directory), "image_*.png"),
        (Path(depth_directory), "depth_*.tiff"),
        (Path(pointcloud_directory), f"{scene_prefix}_*.npz"),
    ]
    highest_index = 0

    for directory, pattern in directories_and_patterns:
        for path in directory.glob(pattern):
            try:
                index = int(path.stem.rsplit("_", 1)[1])
                highest_index = max(highest_index,index)

            except (IndexError, ValueError):
                continue

    return highest_index + 1

class MechEyeCapture(Node):
    def __init__(
        self,
        image_dir,
        depth_dir,
        pointcloud_dir,
        prefix,
    ):
        super().__init__("mecheye_capture")

        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.pointcloud_dir = pointcloud_dir
        self.prefix = prefix
        self.active_index = None

        self.bridge = CvBridge()
        self.waiting_for_color = False
        self.color_finished = False

        self.waiting_for_depth = False
        self.depth_finished = False

        self.waiting_for_cloud = False
        self.capture_finished = False

        self.capture_error = None

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.color_subscription = self.create_subscription(
            Image,
            "/mechmind/color_image",
            self.color_callback,
            image_qos,
        )

        self.depth_subscription = self.create_subscription(
            Image,
            "/mechmind/depth_map",
            self.depth_callback,
            image_qos,
        )

        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            "/mechmind/point_cloud",
            self.pointcloud_callback,
            cloud_qos,
        )

        # Clients
        self.color_capture_client = self.create_client(
            CaptureColorImage,
            "/capture_color_image",
        )

        self.depth_capture_client = self.create_client(
            CaptureDepthMap,
            "/capture_depth_map",
        )

        self.pointcloud_client = self.create_client(
            CapturePointCloud,
            "/capture_point_cloud",
        )

        # Enum client for Scan2DExposureMode
        self.set_enum_client = self.create_client(
            SetEnumParameter,
            "/set_enum_parameter",
        )

        # Float client for Scan2DExposureTime 
        self.set_float_client = self.create_client(
            SetFloatParameter,
            "/set_float_parameter",
        )

        # Float array client for Scan3DExposureSequence 
        self.set_float_array_client = self.create_client(
            SetFloatArrayParameter,
            "/set_float_array_parameter",
        )

    # Generic MechEye parameter service call
    def call_parameter_service(
        self,
        client,
        request,
        description,
        timeout_sec=10.0,
    ):
        while not client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise RuntimeError(
                    f"ROS stopped while waiting for {description} service."
                )
            self.get_logger().info(f"Waiting for {description} service...")

        future = client.call_async(request)
        deadline = time.monotonic() + timeout_sec

        while (
            rclpy.ok()
            and not future.done()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        if not rclpy.ok():
            raise RuntimeError(f"ROS stopped while setting {description}.")

        if not future.done():
            raise TimeoutError(f"Timed out while setting {description}.")

        response = future.result()
        if response is None:
            raise RuntimeError(f"{description} service returned no response.")

        if (hasattr(response, "error_code") and response.error_code != 0):
            raise RuntimeError(
                f"Failed to set {description}: "
                f"error_code={response.error_code}, "
                f"error_description="
                f"{getattr(response, 'error_description', '')}"
            )

        self.get_logger().info(f"{description} configured successfully.")

    def set_2d_exposure_mode_timed(self):
        """
        Set Scan2DExposureMode to Timed so that
        Scan2DExposureTime can be configured.
        """

        request = SetEnumParameter.Request()
        request.name = "Scan2DExposureMode"
        request.value = "Timed"

        self.call_parameter_service(
            self.set_enum_client,
            request,
            "2D exposure mode",
        )

        self.get_logger().info(
            "2D exposure mode: Timed"
        )

    # Set 2D Exposure time
    def set_2d_exposure(self, exposure_ms):
        exposure_ms = float(exposure_ms) 

        # Scan2DExposureTime is only available in Timed mode.
        self.set_2d_exposure_mode_timed()

        exposure_request = SetFloatParameter.Request()
        exposure_request.name = "Scan2DExposureTime"
        exposure_request.value = exposure_ms

        self.call_parameter_service(
            self.set_float_client,
            exposure_request,
            "2D exposure time",
        )

        self.get_logger().info(f"2D exposure time: {exposure_ms:.3f} ms")

    # Set 3D Exposure Sequence
    def set_3d_exposure(self, exposure_sequence_ms):
        """
        Set Mech-Eye 3D exposure sequence.

        A single value:
            [5.0]

        means one 3D exposure of 5 ms.

        Multiple values:
            [3.0, 6.0, 12.0]

        configure a multi-exposure sequence.
        """

        exposure_sequence_ms = [float(value) for value in exposure_sequence_ms]
        if not exposure_sequence_ms:
            raise ValueError("At least one 3D exposure value must be provided.")

        if any(value <= 0 for value in exposure_sequence_ms):
            raise ValueError(
                "All 3D exposure values must be greater than 0 ms."
            )

        request = SetFloatArrayParameter.Request()
        request.name = "Scan3DExposureSequence"
        request.array = exposure_sequence_ms

        self.call_parameter_service(
            self.set_float_array_client,
            request,
            "3D exposure sequence",
        )

        self.get_logger().info(f"3D exposure sequence: {exposure_sequence_ms} ms")

    def configure_exposure(self, exposure_2d=None, exposure_3d=None):
        """
        Configure exposure values before capture.
        """

        if exposure_2d is not None:
            self.set_2d_exposure( exposure_2d)
        if exposure_3d is not None:
            self.set_3d_exposure(exposure_3d)

    def color_callback(self, message):
        if not self.waiting_for_color:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            output_path = self.image_dir / f"image_{self.active_index:04d}.png"

            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Failed to save RGB image: {output_path}")

            print(f"Saved RGB:   {output_path}")

        except Exception as error:
            self.capture_error = error
            self.get_logger().error(f"RGB capture failed: {error!r}")

        finally:
            self.waiting_for_color = False
            self.color_finished = True

    def depth_callback(self, message):
        if not self.waiting_for_depth:
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough",)
            depth = np.asarray(depth, dtype=np.float32,)
            output_path = self.depth_dir / f"depth_{self.active_index:04d}.tiff"

            if not cv2.imwrite(str(output_path), depth):
                raise RuntimeError(f"Failed to save depth image: {output_path}")

            valid_depth = np.isfinite(depth) & (depth > 0.0)

            print(f"Saved depth: {output_path}")

        except Exception as error:
            self.capture_error = error
            self.get_logger().error(f"Depth capture failed: {error!r}")

        finally:
            self.waiting_for_depth = False
            self.depth_finished = True

    def pointcloud_callback(self, message):
        if not self.waiting_for_cloud:
            return

        self.get_logger().info(f"Received point cloud: {message.width} x {message.height}")

        try:
            xyz, rgb = convert_cloud(message)

            output_path = self.pointcloud_dir / f"{self.prefix}_{self.active_index:04d}.npz"
            save_data = {
                "xyz": xyz,
                "height": np.int32(message.height),
                "width": np.int32(message.width),
                "frame_id": np.asarray(message.header.frame_id),
                "stamp_sec": np.int64(message.header.stamp.sec),
                "stamp_nanosec": np.uint32(message.header.stamp.nanosec),
            }

            if rgb is not None:
                save_data["rgb"] = rgb

            # Uncompressed saving is faster during capture.
            np.savez(output_path, **save_data)

            valid_z = np.isfinite(xyz[..., 2]) & (xyz[..., 2] > 0)

            print(f"Saved: {output_path}")
            print(f"XYZ shape: {xyz.shape}")

            if rgb is not None:
                print(f"RGB shape: {rgb.shape}")

            print(
                "Valid points:",
                int(np.count_nonzero(valid_z)),
            )

            if np.any(valid_z):
                print(
                    "Median Z:",
                    float(np.median(xyz[..., 2][valid_z])),
                )

        except Exception as error:
            self.capture_error = error
            self.get_logger().error(f"Point-cloud conversion or saving failed: {error!r}")

        finally:
            self.waiting_for_cloud = False
            self.capture_finished = True

    def capture_and_wait(
        self,
        client,
        request,
        waiting_attribute,
        finished_attribute,
        description,
        timeout_sec=30.0,
    ):
        while not client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise RuntimeError(f"ROS stopped while waiting for {description} service.")
            self.get_logger().info(f"Waiting for {description} service...")

        self.capture_error = None

        setattr(self, waiting_attribute, True)
        setattr(self, finished_attribute, False)

        future = client.call_async(request)
        deadline = time.monotonic() + timeout_sec

        response = None

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self,timeout_sec=0.1,)

            if self.capture_error is not None:
                setattr(self, waiting_attribute, False)
                raise self.capture_error

            if future.done() and response is None:
                response = future.result()

                if response is None:
                    raise RuntimeError(
                        f"{description} service returned no response."
                    )

                if (hasattr(response, "error_code") and response.error_code != 0):
                    raise RuntimeError(
                        f"{description} failed: "
                        f"{response.error_code}, "
                        f"{getattr(response, 'error_description', '')}"
                    )

            if (response is not None and getattr(self, finished_attribute)):
                return

        setattr(self, waiting_attribute, False)

        raise TimeoutError(f"Timed out waiting for {description}.")

    def capture_color(self):
        self.capture_and_wait(
            client=self.color_capture_client,
            request=CaptureColorImage.Request(),
            waiting_attribute="waiting_for_color",
            finished_attribute="color_finished",
            description="color image",
        )


    def capture_depth(self):
        self.capture_and_wait(
            client=self.depth_capture_client,
            request=CaptureDepthMap.Request(),
            waiting_attribute="waiting_for_depth",
            finished_attribute="depth_finished",
            description="depth image",
        )


    def capture_pointcloud(self):
        self.capture_and_wait(
            client=self.pointcloud_client,
            request=CapturePointCloud.Request(),
            waiting_attribute="waiting_for_cloud",
            finished_attribute="capture_finished",
            description="point cloud",
        )

    def capture_once(self):
        self.active_index = find_next_capture_index(
            image_directory=self.image_dir,
            depth_directory=self.depth_dir,
            pointcloud_directory=self.pointcloud_dir,
            scene_prefix=self.prefix
        )

        print()
        print(f"Capture set {self.active_index:04d}")
        print("----------------")

        try:
            print("1/3 Capturing RGB image...")
            self.capture_color()

            print("2/3 Capturing depth image...")
            self.capture_depth()

            print("3/3 Capturing point cloud...")
            self.capture_pointcloud()

            print(f"Completed capture set {self.active_index:04d}")

        finally:
            self.active_index = None

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-directory",
        default="/home/support/unseen_sku_ws/data/images",
    )
    
    parser.add_argument(
        "--depth-directory",
        default="/home/support/unseen_sku_ws/data/depth_images",
    )
    
    parser.add_argument(
        "--pointcloud-directory",
        default="/home/support/unseen_sku_ws/data/mecheye_pointclouds",
    )

    parser.add_argument(
        "--prefix",
        default="scene",
        help="For example: background or scene",
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously request and save point clouds.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Delay in seconds between captures.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of captures. Zero means unlimited.",
    )

    # Exposure arguments
    parser.add_argument(
        "--exposure-2d",
        type=float,
        default=None, # 250.0
        metavar="MS",
        help=(
            "2D exposure time in milliseconds. "
            "Example: --exposure-2d 40.0"
        )
    )

    parser.add_argument(
        "--exposure-3d",
        type=float,
        nargs="+",
        default=None, # [5.0, 10.0, 70.0]
        metavar="MS",
        help=(
            "3D exposure sequence in milliseconds. "
            "Examples: --exposure-3d 5.0 or --exposure-3d 3.0 6.0 12.0"
        ),
    )

    return parser.parse_known_args()

def main():
    # Define arguments
    args, ros_arguments = parse_arguments()

    # Output directories
    image_directory = Path(args.image_directory).expanduser()
    depth_directory = Path(args.depth_directory).expanduser()
    pointcloud_directory = Path(args.pointcloud_directory).expanduser()

    image_directory.mkdir(parents=True, exist_ok=True)
    depth_directory.mkdir(parents=True, exist_ok=True)
    pointcloud_directory.mkdir(parents=True, exist_ok=True,)

    rclpy.init(args=ros_arguments)

    # Create node
    node = MechEyeCapture(
        image_dir=image_directory,
        depth_dir=depth_directory,
        pointcloud_dir=pointcloud_directory,
        prefix=args.prefix,
    )

    try:
        node.configure_exposure(
            exposure_2d=args.exposure_2d,
            exposure_3d=args.exposure_3d,
        )
        
        if args.continuous:
            captured_count = 0

            print("Continuous capture started.")
            print("Press Ctrl+C to stop.")

            while rclpy.ok():
                if args.count > 0 and captured_count >= args.count:
                    break

                node.capture_once()
                captured_count += 1

                print(
                    f"Completed capture {captured_count}"
                )

                if args.interval > 0:
                    time.sleep(args.interval)

        else:
            print("Press Enter to capture.")
            print("Type q and press Enter to quit.")

            while rclpy.ok():
                command = input("> ").strip().lower()

                if command == "q":
                    break

                node.capture_once()

    except KeyboardInterrupt:
        print("\nCapture stopped.")

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()