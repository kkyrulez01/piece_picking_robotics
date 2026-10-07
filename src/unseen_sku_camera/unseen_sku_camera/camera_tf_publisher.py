#!/usr/bin/env python3

from pathlib import Path
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .camera_utils import (
    load_camera_config,
    load_camera_to_world_calibration,
    rotation_matrix_to_quaternion,
)

PACKAGE_SHARE_DIRECTORY = Path(get_package_share_directory("unseen_sku_camera"))
DEFAULT_CONFIG_PATH = PACKAGE_SHARE_DIRECTORY / "config" / "camera_calibration.yaml"
CALIBRATION_DIRECTORY = PACKAGE_SHARE_DIRECTORY / "calibration_data"

class CameraTfPublisher(Node):
    """
    Publish static TF transforms for the calibrated cameras.

    TF tree:
        base_link
        ├── helios2_optical_frame
        └── camera_color_optical_frame

    Calibration convention stored in the NPZ files:
        p_camera = R_world2cam @ p_world + t_world2cam

    camera_utils.load_camera_to_world_calibration()
    converts this into:
        p_world = R_cam2world @ p_camera + t_cam2world
    """
    def __init__(self):
        super().__init__("camera_tf_publisher")

        # Parameters
        self.declare_parameter("config_path", str(DEFAULT_CONFIG_PATH),)
        config_path = self.get_parameter("config_path").get_parameter_value().string_value

        # Load camera calibration configuration
        config = load_camera_config(config_path=config_path, section_name="camera_calibration")

        self.base_frame = str(config.get("base_frame", "base_link"))

        helios_config = config["helios2"]
        realsense_config = config["realsense"]

        self.helios_frame = str(helios_config.get("frame_id", "helios2_optical_frame"))
        self.realsense_frame = str(realsense_config.get("frame_id", "camera_color_optical_frame"))

        # Calibration file paths
        helios_calibration_path = CALIBRATION_DIRECTORY / helios_config["calibration_file"]
        realsense_calibration_path = CALIBRATION_DIRECTORY / realsense_config["calibration_file"]

        # Load camera -> world/base transforms
        (self.R_helios2base, 
         self.t_helios2base)= load_camera_to_world_calibration(helios_calibration_path)

        (self.R_realsense2base,
         self.t_realsense2base) = load_camera_to_world_calibration(realsense_calibration_path)

        # Static TF broadcaster
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Create transforms
        helios_transform = self.create_transform(
            parent_frame=self.base_frame,
            child_frame=self.helios_frame,
            rotation=self.R_helios2base,
            translation=self.t_helios2base
        )

        realsense_transform = self.create_transform(
            parent_frame=self.base_frame,
            child_frame=self.realsense_frame,
            rotation=self.R_realsense2base,
            translation=self.t_realsense2base
        )

        # Publish both static transforms
        self.static_tf_broadcaster.sendTransform([helios_transform, realsense_transform])

        # Logging
        self.get_logger().info("Published static camera TFs.")

        self.log_transform(
            name="Helios2",
            parent_frame=self.base_frame,
            child_frame=self.helios_frame,
            rotation=self.R_helios2base,
            translation=self.t_helios2base,
        )
        self.log_transform(
            name="RealSense",
            parent_frame=self.base_frame,
            child_frame=self.realsense_frame,
            rotation=self.R_realsense2base,
            translation=self.t_realsense2base,
        )

    def create_transform(self, parent_frame, child_frame, rotation, translation):
        """
        Create a geometry_msgs/TransformStamped.

        rotation:
            3x3 child -> parent rotation.
        translation:
            Child origin expressed in parent coordinates.
        """
        quaternion = rotation_matrix_to_quaternion(rotation)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        # Translation
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])

        # Rotation
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])

        return transform

    # Logging helper
    def log_transform(self, name, parent_frame, child_frame, rotation, translation):
        quaternion = rotation_matrix_to_quaternion(rotation)

        self.get_logger().info(
            f"{name} static TF:\n"
            f"  parent: {parent_frame}\n"
            f"  child:  {child_frame}\n"
            f"  translation [m]: "
            f"[{translation[0]:.6f}, "
            f"{translation[1]:.6f}, "
            f"{translation[2]:.6f}]\n"
            f"  quaternion [x y z w]: "
            f"[{quaternion[0]:.6f}, "
            f"{quaternion[1]:.6f}, "
            f"{quaternion[2]:.6f}, "
            f"{quaternion[3]:.6f}]"
        )

def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = CameraTfPublisher()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(
                f"Camera TF publisher failed: {error}"
            )
        else:
            print(
                f"Camera TF publisher failed during startup: {error}"
            )

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()