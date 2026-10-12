#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge

from realsense2_camera_msgs.msg import RGBD

class RealSenseCapture(Node):
    def __init__(self):
        super().__init__("realsense_capture")

        # Parameters
        self.declare_parameter(
            "rgbd_topic",
            "/camera/camera/rgbd",
        )
        self.declare_parameter(
            "output_root",
            "/home/support/unseen_sku_ws/data/realsense/scene",
        )
        # Default D455 depth scale: 1 unit = 1 mm
        # Hence, multiply values by 0.001 to get in metres
        self.declare_parameter(
            "depth_scale",
            0.001,
        )

        # Read parameters
        self.rgbd_topic = self.get_parameter("rgbd_topic").get_parameter_value().string_value
        self.output_root = Path(self.get_parameter("output_root").get_parameter_value().string_value)
        self.depth_scale = self.get_parameter("depth_scale").get_parameter_value().double_value

        # Output directories
        self.image_directory = self.output_root / "images"
        self.depth_directory = self.output_root / "depth_images"

        self.image_directory.mkdir(parents=True, exist_ok=True)
        self.depth_directory.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()

        self.latest_rgb = None
        self.latest_depth = None

        self.latest_rgb_encoding = None
        self.latest_depth_encoding = None

        # To determine next file index
        self.next_index = self.find_next_index()

        # RGBD Subscriber
        rgbd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            RGBD,
            self.rgbd_topic,
            self.rgbd_callback,
            rgbd_qos,
        )

        self.get_logger().info("RealSense RGB-D capture node started.")
        self.get_logger().info(f"RGBD topic: {self.rgbd_topic}")
        self.get_logger().info(f"Output directory: {self.output_root}")

    def find_next_index(self):
        directories_and_patterns = [
            (Path(self.image_directory), "image_*.png"),
            (Path(self.depth_directory), "depth_*.tiff"),
        ]

        highest_index = 0
    
        for directory, pattern in directories_and_patterns:
            for path in directory.glob(pattern):
                try:
                    index = int(path.stem.rsplit("_", 1)[1])
                    highest_index = max(highest_index, index)
    
                except (IndexError, ValueError):
                    continue
    
        return highest_index + 1

    def rgbd_callback(self, message):
        try:
            rgb = self.bridge.imgmsg_to_cv2(message.rgb, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(message.depth, desired_encoding="passthrough")

        except Exception as error:
            self.get_logger().error(f"RGBD conversion failed: {error}")
            return

        self.latest_rgb = np.array(rgb, copy=True)
        self.latest_depth = np.array(depth, copy=True)

        self.latest_rgb_encoding = message.rgb.encoding
        self.latest_depth_encoding = message.depth.encoding

    def convert_depth_to_metres(self, depth):
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * self.depth_scale

        if np.issubdtype(depth.dtype, np.floating,):
            return depth.astype(np.float32)

    def capture_pair(self):
        if (self.latest_rgb is None or self.latest_depth is None):
            self.get_logger().warning("no RGB-D frame received yet.")
            return False

        index = self.next_index

        image_path = self.image_directory / f"image_{index:04d}.png"
        depth_path = self.depth_directory / f"depth_{index:04d}.tiff"

        try:
            depth_metres = self.convert_depth_to_metres(self.latest_depth)

            image_success = cv2.imwrite(str(image_path), self.latest_rgb)
            depth_success = cv2.imwrite(str(depth_path), depth_metres)

            if not image_success:
                raise RuntimeError("Failed to save RGB image.")

            if not depth_success:
                raise RuntimeError("Failed to save depth TIFF.")

        except Exception as error:
            self.get_logger().error(f"Capture failed: {error}")
            return False

        valid = (np.isfinite(depth_metres) & (depth_metres > 0.0))
        if np.any(valid):
            median_depth = np.median(depth_metres[valid])
            self.get_logger().info(f"Median depth: {median_depth:.3f} m")

        self.get_logger().info(f"Saved {image_path}")
        self.get_logger().info(f"Saved {depth_path}")

        self.next_index += 1

        return True

def main(args=None):
    rclpy.init(args=args)

    node = RealSenseCapture()

    # Run ROS callbacks in background.
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )

    spin_thread.start()

    print()
    print("RealSense RGB-D capture ready.")
    print("Press ENTER to capture RGB + depth.")
    print("Type q then ENTER to quit.")
    print()

    try:
        while rclpy.ok():
            command = input("[ENTER = capture, q = quit] > ").strip().lower()
            if command == "q":
                break

            node.capture_pair()

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()

        spin_thread.join(timeout=1.0,)

if __name__ == "__main__":
    main()