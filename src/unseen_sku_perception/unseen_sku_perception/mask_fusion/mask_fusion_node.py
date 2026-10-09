#!/usr/bin/env python3
"""ROS 2 adapter for MaskFusionPipeline (algorithm code lives elsewhere).

Subscribes to Helios PointCloud2 and SAM2 ObjectMaskArray.
Publishes per-object fused clouds in the Helios optical frame.
"""
from threading import Lock

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

from unseen_sku_interfaces.msg import (
    FusedObject,
    FusedObjectArray,
    ObjectMaskArray,
)
from .mask_fusion_processor import MaskFusionProcessor

class MaskFusionNode(Node):
    def __init__(self):
        super().__init__("mask_fusion_node")

        # Parameters
        self.declare_parameter("helios_topic", "/helios2/points")
        self.declare_parameter("masks_topic", "/sam2/object_masks")
        self.declare_parameter("output_topic", "/mask_fusion/objects")
        self.declare_parameter("helios_to_realsense_file", "")
        self.declare_parameter("realsense_calibration_file", "")
        self.declare_parameter("erosion_size", 1)
        self.declare_parameter("z_buffer", False)
        self.declare_parameter("save_debug", True)
        self.declare_parameter("output_root", "~/unseen_sku_ws/outputs")

        transform_path = str(
            self.get_parameter("helios_to_realsense_file").value
        ).strip()
        intrinsics_path = str(
            self.get_parameter("realsense_calibration_file").value
        ).strip()
        if not transform_path or not intrinsics_path:
            raise ValueError(
                "Set ROS parameters helios_to_realsense_file and "
                "realsense_calibration_file to your NPZ calibration paths."
            )

        self.fusion = MaskFusionProcessor(
            helios_to_realsense_file=transform_path,
            realsense_calibration_file=intrinsics_path,
            erosion_size=int(self.get_parameter("erosion_size").value),
            z_buffer=bool(self.get_parameter("z_buffer").value),
            save_debug=bool(self.get_parameter("save_debug").value),
            output_root=str(self.get_parameter("output_root").value),
        )

        self.bridge = CvBridge()
        self._lock = Lock()
        self._pending_cloud = None
        self._pending_masks = None
        self._last_scene_index = None

        # Subscribers
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("helios_topic").value),
            self._helios_callback,
            reliable_qos
        )
        self.masks_sub = self.create_subscription(
            ObjectMaskArray,
            str(self.get_parameter("masks_topic").value),
            self._masks_callback,
            reliable_qos
        )

        # Publishers
        self.objects_pub = self.create_publisher(
            FusedObjectArray,
            str(self.get_parameter("output_topic").value),
            10
        )

        self.get_logger().info(
            "Mask fusion ready. Waiting for /helios2/points and /sam2/object_masks..."
        )

    def _helios_callback(self, message):
        with self._lock:
            self._pending_cloud = message
        self._process_if_ready()

    def _masks_callback(self, message):
        with self._lock:
            self._pending_masks = message
        self._process_if_ready()

    def _process_if_ready(self):
        with self._lock:
            if self._pending_cloud is None or self._pending_masks is None:
                return
            cloud = self._pending_cloud
            masks = self._pending_masks
            self._pending_cloud = None
            self._pending_masks = None

        scene_index = int(masks.scene_index)
        if scene_index == self._last_scene_index:
            self.get_logger().warning(
                f"Skipping duplicate SAM2 scene_{scene_index:04d}"
            )
            return

        try:
            self._publish_fused_objects(cloud, masks)
            self._last_scene_index = scene_index
        except Exception as error:
            self.get_logger().error(
                f"Mask fusion failed for scene_{scene_index:04d}: {error}"
            )

    def _publish_fused_objects(self, cloud_msg, mask_msg):
        """
        Main mask point cloud fusion process.
        """

        # ROS -> NumPy
        helios_xyz = self._cloud_to_xyz(cloud_msg)

        object_masks = {}
        for incoming in mask_msg.objects:
            object_id = int(incoming.object_id)
            if object_id in object_masks:
                raise ValueError(f"Duplicate SAM2 object_id: {object_id}")
            mask = self.bridge.imgmsg_to_cv2(incoming.mask, desired_encoding="mono8")
            object_masks[object_id] = np.asarray(mask) > 127

        # Call process() from MaskFusionProcessor
        points_by_object = self.fusion.process(
            helios_xyz=helios_xyz,
            object_masks=object_masks,
            scene_index=int(mask_msg.scene_index),
        )

        # NumPy -> ROS. Output XYZ is still in the original Helios frame.
        output = FusedObjectArray()
        output.header = cloud_msg.header
        output.scene_index = int(mask_msg.scene_index)

        for incoming in mask_msg.objects:
            object_id = int(incoming.object_id)
            data = points_by_object.get(object_id)
            if data is None or len(data["helios_xyz"]) == 0:
                self.get_logger().warning(
                    f"Object {object_id}: no projected Helios points inside mask"
                )
                continue

            fused_object = FusedObject()
            fused_object.object_id = object_id
            fused_object.mask = incoming.mask
            fused_object.cloud = self._make_object_cloud(cloud_msg.header, data)
            output.objects.append(fused_object)
            self.get_logger().info(
                f"Object {object_id}: {len(data['helios_xyz'])} fused points"
            )

        self.objects_pub.publish(output)
        self.get_logger().info(
            f"scene_{output.scene_index:04d}: published "
            f"{len(output.objects)} fused objects"
        )

    @staticmethod
    def _cloud_to_xyz(message):
        """
        Read organized PointCloud2 XYZ while preserving flattened indices.
        """
        fields = {field.name: field for field in message.fields}
        for name in ("x", "y", "z"):
            if name not in fields:
                raise ValueError(f"Helios point cloud lacks '{name}'")
            field = fields[name]
            if field.datatype != PointField.FLOAT32 or field.count != 1:
                raise ValueError(f"Helios field '{name}' must be FLOAT32")

        endian = ">" if message.is_bigendian else "<"
        dtype = np.dtype({
            "names": ("x", "y", "z"),
            "formats": (endian + "f4",) * 3,
            "offsets": tuple(fields[name].offset for name in ("x", "y", "z")),
            "itemsize": message.point_step,
        })
        view = np.ndarray(
            shape=(message.height, message.width),
            dtype=dtype,
            buffer=message.data,
            strides=(message.row_step, message.point_step),
        )
        return np.column_stack((
            view["x"].reshape(-1),
            view["y"].reshape(-1),
            view["z"].reshape(-1),
        )).astype(np.float32)

    @staticmethod
    def _make_object_cloud(header, object_data):
        """
        Write XYZ, RealSense XYZ, UV, and source index into PointCloud2 message.
        """
        xyz = np.asarray(object_data["helios_xyz"], dtype=np.float32)
        rs_xyz = np.asarray(object_data["realsense_xyz"], dtype=np.float32)
        uv = np.asarray(object_data["uv"], dtype=np.int32)
        indices = np.asarray(object_data["source_indices"], dtype=np.uint32)

        field_names = (
            "x", "y", "z", "rs_x", "rs_y", "rs_z", "u", "v", "source_index"
        )
        field_types = (
            PointField.FLOAT32, PointField.FLOAT32, PointField.FLOAT32,
            PointField.FLOAT32, PointField.FLOAT32, PointField.FLOAT32,
            PointField.INT32, PointField.INT32, PointField.UINT32,
        )
        records = np.empty(len(xyz), dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("rs_x", "<f4"), ("rs_y", "<f4"), ("rs_z", "<f4"),
            ("u", "<i4"), ("v", "<i4"), ("source_index", "<u4"),
        ]))
        for col, field in enumerate(("x", "y", "z")):
            records[field] = xyz[:, col]
        for col, field in enumerate(("rs_x", "rs_y", "rs_z")):
            records[field] = rs_xyz[:, col]
        records["u"] = uv[:, 0]
        records["v"] = uv[:, 1]
        records["source_index"] = indices

        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = len(xyz)
        cloud.fields = [
            PointField(name=name, offset=4 * idx, datatype=data_type, count=1)
            for idx, (name, data_type) in enumerate(zip(field_names, field_types))
        ]
        cloud.is_bigendian = False
        cloud.point_step = records.dtype.itemsize
        cloud.row_step = cloud.width * cloud.point_step
        cloud.is_dense = bool(np.isfinite(xyz).all())
        cloud.data = records.tobytes()
        return cloud

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MaskFusionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"mask_fusion_node failed: {error}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()