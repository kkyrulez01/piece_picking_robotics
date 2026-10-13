#!/usr/bin/env python3
import threading
from pathlib import Path
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from ament_index_python.packages import get_package_share_directory

from cv_bridge import CvBridge
from realsense2_camera_msgs.msg import RGBD
from std_srvs.srv import Trigger

from unseen_sku_interfaces.msg import ObjectMask, ObjectMaskArray
from .sam2_RGB_processor import Sam2Processor
from .output_utils import find_next_index

PACKAGE_SHARE_DIRECTORY = Path(get_package_share_directory("unseen_sku_perception"))
DEFAULT_CONFIG_FILE = PACKAGE_SHARE_DIRECTORY / "config" / "sam2.yaml"

class Sam2Node(Node):
    """
    ROS 2 wrapper for the SAM2 perception pipeline.

    The node:
        1. Subscribes to the RealSense RGBD topic.
        2. Caches the latest synchronized RGB/depth pair.
        3. Waits for /sam2/process_scene.
        4. Runs Sam2Processor once.
        5. Publishes ObjectMaskArray.

    SAM2 processing logic remains in main_RGB.py.
    """
    def __init__(self):
        super().__init__("sam2_node")

        # Parameters
        self.declare_parameter("config_file", str(DEFAULT_CONFIG_FILE),)
        self.declare_parameter("rgbd_topic", "/camera/camera/rgbd",)
        self.declare_parameter("process_service", "/sam2/process_scene",)
        self.declare_parameter("masks_topic", "/sam2/object_masks",)
        self.declare_parameter("log_images", False)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("initial_scene_index", 1)

        self.config_file = str(self.get_parameter("config_file").value)
        self.rgbd_topic = str(self.get_parameter("rgbd_topic").value)
        self.process_service_name = str(self.get_parameter("process_service").value)
        self.masks_topic = str(self.get_parameter("masks_topic").value)
        self.log_images = bool(self.get_parameter("log_images").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        
        self.realsense_rgb_directory = Path("/home/support/unseen_sku_ws/data/realsense/scene/images")
        self.realsense_depth_directory = Path("/home/support/unseen_sku_ws/data/realsense/scene/depth_images")
        self.realsense_rgb_directory.mkdir(parents=True, exist_ok=True)
        self.realsense_depth_directory.mkdir(parents=True, exist_ok=True)
        self.next_scene_index = find_next_index([
            (self.realsense_rgb_directory, "image_*.png",),
            (self.realsense_depth_directory, "depth_*.tiff",),
        ])

        rgbd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()

        self.processor = Sam2Processor(
            config_file=self.config_file,
            log_images=None
        )

        self.data_lock = threading.Lock()

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_stamp = None
        self.latest_frame_id = ""

        # Subscribers
        self.rgbd_subscription = (
            self.create_subscription(
                RGBD,
                self.rgbd_topic,
                self.rgbd_callback,
                rgbd_qos,
            )
        )

        # Publishers
        self.mask_publisher = (
            self.create_publisher(
                ObjectMaskArray,
                self.masks_topic,
                10,
            )
        )

        # Services
        self.process_service = (
            self.create_service(
                Trigger,
                self.process_service_name,
                self.process_scene_callback,
            )
        )

        # Startup logging
        self.get_logger().info("SAM2 node started.")
        self.get_logger().info(f"RGBD topic: {self.rgbd_topic}")
        self.get_logger().info(f"Process service: {self.process_service_name}")
        self.get_logger().info(f"Masks topic: {self.masks_topic}")
        self.get_logger().info(f"Configuration: {self.config_file}")
        self.get_logger().info(f"Log images: {self.log_images}")
        self.get_logger().info(f"Depth scale: {self.depth_scale}")

    # RealSense RGBD callback
    def rgbd_callback(self, message):
        """
        Cache the latest synchronized RGB-D frame.

        SAM2 inference is not performed here.
        """
        try:
            # RGB
            bgr_image = (
                self.bridge.imgmsg_to_cv2(
                    message.rgb,
                    desired_encoding="bgr8",
                )
            )

            bgr_image = np.asarray(bgr_image, dtype=np.uint8,).copy()

            # Depth
            raw_depth = (
                self.bridge.imgmsg_to_cv2(
                    message.depth,
                    desired_encoding="passthrough",
                )
            )

            raw_depth = np.asarray(raw_depth)

            scene_depth = (
                self._convert_depth_to_metres(
                    raw_depth=raw_depth,
                    encoding=message.depth.encoding,
                )
            )

            # Cache one synchronized pair
            with self.data_lock:
                self.latest_rgb = bgr_image
                self.latest_depth = scene_depth
                self.latest_stamp = message.rgb.header.stamp
                self.latest_frame_id = message.rgb.header.frame_id

        except Exception as error:
            self.get_logger().error("Failed to process RGBD message: {error}")

    # Depth  conversion
    def _convert_depth_to_metres(
        self,
        raw_depth,
        encoding,
    ):
        """
        Convert ROS depth data into float32 metres.

        Common RealSense case:

            16UC1
                integer depth values * depth_scale

        32FC1:
            assumed to already be metres.
        """

        encoding = str(encoding).upper()
        if encoding in {"16UC1", "MONO16"}:
            depth_m = raw_depth.astype(np.float32) * self.depth_scale

        elif encoding == "32FC1":
            depth_m = raw_depth.astype(np.float32)

        elif np.issubdtype(raw_depth.dtype, np.integer,):
            self.get_logger().warning(
                "Unknown integer depth encoding "
                f"'{encoding}'. Applying "
                f"depth_scale={self.depth_scale}."
            )

            depth_m = raw_depth.astype(np.float32) * self.depth_scale

        else:
            self.get_logger().warning(
                "Unknown floating-point depth "
                f"encoding '{encoding}'. "
                "Assuming values are metres."
            )

            depth_m = raw_depth.astype(np.float32)

        return depth_m

    # Save image for debugging
    def _save_realsense_scene(
        self,
        scene_index,
        bgr_image,
        scene_depth,
    ):
        rgb_path = self.realsense_rgb_directory / f"image_{scene_index:04d}.png"
        depth_path = self.realsense_depth_directory / f"depth_{scene_index:04d}.tiff"

        cv2.imwrite(str(rgb_path), bgr_image,)

        # scene_depth is currently in metres inside sam2_node.
        # Save it back as uint16 millimetres.
        depth_mm = np.round(scene_depth * 1000.0).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm,)

        self.get_logger().info(f"Saved RealSense RGB: {rgb_path}")
        self.get_logger().info(f"Saved RealSense depth: {depth_path}")

    # Process service
    def process_scene_callback(self, request, response):
        """
        Process exactly one cached RealSense scene.
        """
        del request

        # Copy cached frame
        with self.data_lock:
            if (self.latest_rgb is None or self.latest_depth is None):
                response.success = False
                response.message = "No realsense RGBD frame received yet."

                return response

            bgr_image = self.latest_rgb.copy()
            scene_depth = self.latest_depth.copy()
            stamp = self.latest_stamp
            frame_id = self.latest_frame_id

        scene_index = int(self.next_scene_index)
        self._save_realsense_scene(
            scene_index=scene_index,
            bgr_image=bgr_image,
            scene_depth=scene_depth,
        )
        self.get_logger().info(f"Processing scene_{scene_index:04d}...")

        # Run SAM2 pipeline
        try:
            accepted_masks = self.processor.process(
                    scene_depth=scene_depth,
                    bgr_image=bgr_image,
                    scene_index=scene_index,
                )
            
            # Publish masks
            self._publish_masks(
                accepted_masks=accepted_masks,
                scene_index=scene_index,
                stamp=stamp,
                frame_id=frame_id,
            )

            # Only increment after successful processing.
            self.next_scene_index += 1

            response.success = True
            response.message = (
                f"scene_{scene_index:04d}: "
                f"{len(accepted_masks)} "
                "object masks."
            )

            self.get_logger().info(response.message)

        except Exception as error:
            response.success = False

            response.message = "SAM2 processing failed: {error}"
            self.get_logger().error(response.message)

        return response

    # Publish SAM2 masks
    def _publish_masks(
        self,
        accepted_masks,
        scene_index,
        stamp,
        frame_id
    ):
        """
        Publish all final masks for one scene in a single
        ObjectMaskArray message.
        """

        output = ObjectMaskArray()

        output.header.stamp = stamp
        output.header.frame_id = frame_id
        output.scene_index = int(scene_index)

        # Convert each boolean NumPy mask to mono8 ROS
        for (object_id, mask_result,) in enumerate(accepted_masks, start=1):
            mask = np.asarray(mask_result.mask, dtype=bool,)
            mask_uint8 = mask.astype(np.uint8) * 255

            mask_image = self.bridge.cv2_to_imgmsg(mask_uint8,
                                                   encoding="mono8")
        
            mask_image.header.stamp = stamp
            mask_image.header.frame_id = frame_id

            object_mask = ObjectMask()

            object_mask.object_id = int(object_id)
            object_mask.mask = mask_image

            output.objects.append(object_mask)

        self.mask_publisher.publish(output)

        self.get_logger().info(
            "Published SAM2 masks: "
            f"scene={scene_index:04d}, "
            f"objects={len(output.objects)}, "
            f"frame={frame_id}"
        )

def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = Sam2Node()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error("SAM2 node failed: {error}")

        else:
            print(f"SAM2 node failed during startup: {error}")

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()