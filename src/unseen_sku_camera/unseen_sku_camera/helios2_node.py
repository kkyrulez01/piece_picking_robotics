#!/usr/bin/env python3

import numpy as np
import cv2
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node

from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import (
    PointCloud2,
    PointField,
)
from std_srvs.srv import Trigger

from .helios2_camera import Helios2Camera
from .camera_utils import (
    save_debug_npz,
    load_camera_config
)

PACKAGE_SHARE_DIRECTORY = Path(get_package_share_directory("unseen_sku_camera"))
DEFAULT_CONFIG_PATH = PACKAGE_SHARE_DIRECTORY / "config" / "helios2.yaml"

class Helios2Node(Node):
    """
    ROS 2 wrapper around Helios2Camera.

    ROS interfaces:
        Service:
            /helios2/capture
            std_srvs/srv/Trigger
        Publisher:
            /helios2/points
            sensor_msgs/msg/PointCloud2

    PointCloud2 fields:
        x         float32, metres
        y         float32, metres
        z         float32, metres
        intensity float32
    """
    def __init__(self):
        super().__init__("helios2_node")

        # ROS parameters
        self.declare_parameter("config_path", str(DEFAULT_CONFIG_PATH))
        self.declare_parameter("frame_id", "helios2_optical_frame")
        self.declare_parameter("pointcloud_topic", "/helios2/points")
        self.declare_parameter("capture_service", "/helios2/capture")
        self.declare_parameter("warmup_seconds", 0.5)

        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        helios_config = load_camera_config(config_path, "helios2_capture")
        serial = helios_config.get("serial")
        exposure = helios_config.get("exposure", "Exp250Us")
        output_root = helios_config.get("output_root", "/home/support/data/helios2")
        self.output_root = Path(output_root).expanduser().resolve()
        self.save_debug = bool(helios_config.get("save_debug", True))
        self.scene_output_directory = self.output_root / "scene"
        self.scene_output_directory.mkdir(parents=True, exist_ok=True)

        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        pointcloud_topic = self.get_parameter( "pointcloud_topic").get_parameter_value().string_value
        capture_service = self.get_parameter("capture_service").get_parameter_value().string_value
        warmup_seconds = self.get_parameter("warmup_seconds").get_parameter_value().double_value

        if serial == "":
            serial = None

        # Point cloud publisher
        pointcloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.pointcloud_publisher = (
            self.create_publisher(
                PointCloud2,
                pointcloud_topic,
                pointcloud_qos,
            )
        )

        # Capture service
        self.capture_service = (
            self.create_service(
                Trigger,
                capture_service,
                self.capture_callback,
            )
        )

        # Camera
        self.get_logger().info("Connecting to Helios2...")
        self.camera = Helios2Camera(
            serial=serial,
            exposure=exposure,
            warmup_seconds=warmup_seconds,
            log_callback=(self.get_logger().info),
            auto_connect=True,
        )

        # Start up information
        self.get_logger().info("Helios2 ROS node ready.")
        self.get_logger().info(f"Point cloud topic: {pointcloud_topic}")
        self.get_logger().info(f"Capture service: {capture_service}")
        self.get_logger().info(f"Frame ID: {self.frame_id}")

    # Capture service
    def capture_callback(self, request, response):
        del request

        try:
            self.get_logger().info("Capturing Helios2 frame...")

            # Capture from camera
            frame = self.camera.capture()
            xyz_m = frame.xyz_m
            intensity = frame.intensity

            # Save debug NPZ
            if self.save_debug:
                scene_index, debug_path = save_debug_npz(
                output_directory=self.scene_output_directory,
                prefix="scene",
                data={
                    "xyz": frame.xyz_m,
                    "intensity": frame.intensity,
                    "xyz_unit": np.array("m"),
                    "width": np.array(frame.width),
                    "height": np.array(frame.height),
                    "exposure": np.array(frame.exposure),
                },
            )
            self.get_logger().info(f"Debug capture saved: {debug_path}")

            # Statistics
            xyz_flat = xyz_m.reshape(-1, 3,)
            valid = np.isfinite(xyz_flat).all(axis=1)

            valid_count = int(np.count_nonzero(valid))
            total_count = int(len(xyz_flat))

            # Convert to ROS PointCloud2
            stamp = self.get_clock().now().to_msg()

            cloud_message = (
                self.create_pointcloud_message(
                    xyz_m=xyz_m,
                    intensity=intensity,
                    stamp=stamp,
                )
            )

            # Publish
            self.pointcloud_publisher.publish(cloud_message)

            response.success = True
            response.message = (
                "Helios2 frame published: "
                f"{frame.width}x{frame.height}, "
                f"{valid_count}/{total_count} "
                "valid points."
            )

            self.get_logger().info(response.message)

        except Exception as error:
            response.success = False
            response.message = str(error)

            self.get_logger().error("Helios2 capture failed: {error}")

        return response

    # PointCloud2 conversion
    def create_pointcloud_message(self, xyz_m, intensity, stamp):
        """
        Convert organized Helios XYZ + intensity into
        sensor_msgs/msg/PointCloud2.

        The cloud remains organized:
            height = Helios image height
            width  = Helios image width

        Invalid XYZ coordinates remain NaN.

        Point layout:

            offset 0  : x         float32
            offset 4  : y         float32
            offset 8  : z         float32
            offset 12 : intensity float32

        point_step = 16 bytes
        """
        xyz_m = np.asarray(xyz_m, dtype=np.float32,)
        intensity = np.asarray(intensity, dtype=np.float32,)

        if (xyz_m.ndim != 3 or xyz_m.shape[2] != 3):
            raise ValueError(
                "Expected XYZ shape (H, W, 3), "
                f"received {xyz_m.shape}."
            )

        height, width = xyz_m.shape[:2]

        if intensity.shape != (height, width,):
            raise ValueError(
                "Intensity dimensions do not match XYZ. "
                f"XYZ={xyz_m.shape}, "
                f"intensity={intensity.shape}"
            )

        # Pack x, y, z, intensity as four float32 values per point.
        packed = np.empty((height, width, 4), dtype=np.float32,)

        packed[:, :, 0:3] = xyz_m
        packed[:, :, 3] = intensity

        # Construct PointCloud2 message
        message = PointCloud2()

        message.header.stamp = stamp
        message.header.frame_id = (self.frame_id)

        message.height = height
        message.width = width

        message.fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="intensity",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]

        message.is_bigendian = False

        # 4 float32 values:
        # 4 bytes * 4 = 16 bytes
        message.point_step = 16
        message.row_step = message.point_step * width

        # Helios invalid coordinates are represented
        # by NaN, therefore this is generally False.
        message.is_dense = bool(np.isfinite(xyz_m).all())
        message.data = packed.tobytes(order="C")

        return message

    # Cleanup
    def destroy_node(self):
        self.get_logger().info("Shutting down Helios2 node...")

        if hasattr(self, "camera"):
            self.camera.close()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = Helios2Node()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()