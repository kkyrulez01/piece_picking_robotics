"""ROS 2 communication interfaces for the perception pipeline."""

from dataclasses import dataclass
from threading import Lock
from time import monotonic

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2
from realsense2_camera_msgs.msg import RGBD

from unseen_sku_interfaces.msg import (
    ObjectMaskArray,
    FusedObjectArray,
)

from .scene_context import ObservationStamp


@dataclass(frozen=True)
class ReceivedObservation:
    """Metadata of a received ROS 2 message."""

    stamp: ObservationStamp
    received_monotonic: float
    scene_index: int | None = None
    object_count: int | None = None


class PerceptionManager:
    """Manage ROS 2 interfaces for camera, SAM2, and fusion."""

    def __init__(self, node: Node, settings: dict):

        self.node = node
        self.settings = settings

        # Latest message metadata, not full camera data.
        self._lock = Lock()
        self._latest_observations = {}

        # ------------------------------------------
        # QoS
        # ------------------------------------------

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ------------------------------------------
        # Service Clients
        # ------------------------------------------

        self.helios_capture_client = (
            self.node.create_client(
                Trigger,
                settings["helios_capture_service"],
            )
        )

        self.sam2_process_client = (
            self.node.create_client(
                Trigger,
                settings["sam2_process_service"],
            )
        )

        # ------------------------------------------
        # Subscriptions
        # ------------------------------------------

        self.rgbd_sub = self.node.create_subscription(
            RGBD,
            settings["realsense_rgbd_topic"],
            self._rgbd_callback,
            reliable_qos,
        )

        self.helios_sub = self.node.create_subscription(
            PointCloud2,
            settings["helios_cloud_topic"],
            self._helios_callback,
            reliable_qos,
        )

        self.masks_sub = self.node.create_subscription(
            ObjectMaskArray,
            settings["sam2_masks_topic"],
            self._masks_callback,
            reliable_qos,
        )

        self.fusion_sub = self.node.create_subscription(
            FusedObjectArray,
            settings["fused_objects_topic"],
            self._fusion_callback,
            reliable_qos,
        )

        self.node.get_logger().info(
            "PerceptionManager ROS 2 interfaces registered."
        )

    # ==============================================
    # Observation Tracking
    # ==============================================

    def _record_observation(
        self,
        source,
        header,
        scene_index=None,
        object_count=None,
    ):
        """Store timestamp and metadata without copying image data."""

        observation = ReceivedObservation(
            stamp=ObservationStamp(
                sec=int(header.stamp.sec),
                nanosec=int(header.stamp.nanosec),
                frame_id=str(header.frame_id),
            ),
            received_monotonic=monotonic(),
            scene_index=scene_index,
            object_count=object_count,
        )

        with self._lock:
            self._latest_observations[source] = observation

    def get_latest_observations(self):
        """Return a snapshot of the latest observations."""

        with self._lock:
            return dict(self._latest_observations)

    # ==============================================
    # Service Availability
    # ==============================================

    def get_service_availability(self) -> dict:
        """Check which perception services are available."""

        return {
            "helios2": (
                self.helios_capture_client.service_is_ready()
            ),
            "sam2": (
                self.sam2_process_client.service_is_ready()
            ),
        }

    # ==============================================
    # Asynchronous Service Requests
    # ==============================================

    def request_helios_capture(self):
        """Request a Helios2 capture without blocking."""

        if not self.helios_capture_client.service_is_ready():
            raise RuntimeError(
                "Helios2 capture service is unavailable"
            )

        self.node.get_logger().info(
            "Requesting Helios2 capture..."
        )

        return self.helios_capture_client.call_async(
            Trigger.Request()
        )

    def request_sam2_processing(self):
        """Request SAM2 processing without blocking."""

        if not self.sam2_process_client.service_is_ready():
            raise RuntimeError(
                "SAM2 processing service is unavailable"
            )

        self.node.get_logger().info(
            "Requesting SAM2 processing..."
        )

        return self.sam2_process_client.call_async(
            Trigger.Request()
        )

    # ==============================================
    # ROS 2 Callbacks
    # ==============================================

    def _rgbd_callback(self, message):

        self._record_observation(
            source="realsense",
            header=message.rgb.header,
        )

    def _helios_callback(self, message):

        self._record_observation(
            source="helios2",
            header=message.header,
        )

    def _masks_callback(self, message):

        self._record_observation(
            source="sam2",
            header=message.header,
            scene_index=int(message.scene_index),
            object_count=len(message.objects),
        )

    def _fusion_callback(self, message):

        self._record_observation(
            source="mask_fusion",
            header=message.header,
            scene_index=int(message.scene_index),
            object_count=len(message.objects),
        )