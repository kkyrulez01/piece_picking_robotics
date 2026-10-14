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
    FusionInput,
    FusionResult,
    ObjectMaskArray,
    FusedObjectArray,
    SuctionInput,
    SuctionResult,
)

from .observation_buffer import (
    ObservationBuffer,
    ReceivedObservation,
)

from .scene_context import ObservationStamp


class PerceptionManager:
    """Manage ROS 2 interfaces for camera, SAM2, and fusion."""

    def __init__(self, node: Node, settings: dict):

        self.node = node
        self.settings = settings

        self.observations = ObservationBuffer(
            rgbd_history_size=settings["rgbd_history_size"],
        )

        # Storage of the Fusion Result
        self._fusion_result_lock = Lock()

        self._expected_fusion_scene_id = None
        self._latest_fusion_result = None

        # Storage of the Suction Result
        self._suction_result_lock = Lock()

        self._expected_suction_scene_id = None
        self._latest_suction_result = None

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

        # Subscribes to the Mask Fusion Result
        self.fusion_result_sub = self.node.create_subscription(
            FusionResult,
            settings["fusion_result_topic"],
            self._fusion_result_callback,
            reliable_qos,
        )

        # Subscribes to the Sim Suction Result
        self.suction_result_sub = self.node.create_subscription(
            SuctionResult,
            settings["suction_result_topic"],
            self._suction_result_callback,
            reliable_qos,
        )

        # Publishes the Fusion Input to the Coordinator
        self.fusion_input_pub = self.node.create_publisher(
            FusionInput,
            settings.get(
                "fusion_input_topic",
                "/coordinator/fusion_input",
            ),
            reliable_qos,
        )

        # Publishes the Suction Input to the Coordinator
        self.suction_input_pub = self.node.create_publisher(
            SuctionInput,
            settings["suction_input_topic"],
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
        payload=None,
    ):
        """Store observation metadata and an optional ROS message reference."""

        observation = ReceivedObservation(
            stamp=ObservationStamp(
                sec=int(header.stamp.sec),
                nanosec=int(header.stamp.nanosec),
                frame_id=str(header.frame_id),
            ),
            received_monotonic=monotonic(),
            scene_index=scene_index,
            object_count=object_count,
            payload=payload,
        )

        self.observations.record(
            source=source,
            observation=observation,
        )

    # ==============================================
    # FusionInput Publisher
    # ==============================================

    def publish_fusion_input(
        self,
        scene_id: str,
        cloud: PointCloud2,
        masks: ObjectMaskArray,
    ) -> None:
        """
        Publish one coordinator-approved observation pair.

        This method handles ROS message transport only.

        SceneTransaction is responsible for ensuring
        that the observations have passed validation
        before calling this method.
        """

        # Ensure the transaction has an identifier.
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError(
                "Cannot publish FusionInput without a valid scene_id"
            )

        # Verify that the expected ROS message types are provided.
        if not isinstance(cloud, PointCloud2):
            raise TypeError(
                "FusionInput cloud must be a PointCloud2 message"
            )

        if not isinstance(masks, ObjectMaskArray):
            raise TypeError(
                "FusionInput masks must be an ObjectMaskArray message"
            )

        # Construct one atomic observation-pair message.
        message = FusionInput()

        message.scene_id = scene_id.strip()
        message.cloud = cloud
        message.masks = masks

        # Publish to the coordinator-controlled Mask Fusion topic.
        self.fusion_input_pub.publish(message)

        self.node.get_logger().info(
            f"Published FusionInput: "
            f"scene_id={message.scene_id}; "
            f"sam2_scene_index={message.masks.scene_index}; "
            f"objects={len(message.masks.objects)}"
        )

    # ==============================================
    # SuctionInput Publisher
    # ==============================================

    def publish_suction_input(
        self,
        scene_id: str,
        fused_objects: FusedObjectArray,
    ) -> None:
        """
        Publish one coordinator-approved fused scene
        to Sim-Suction.

        SceneTransaction is responsible for validating
        FusionResult before calling this method.
        """

        # Validate coordinator transaction identifier.
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError(
                "Cannot publish SuctionInput without a valid scene_id"
            )

        # Validate the expected ROS message type.
        if not isinstance(fused_objects, FusedObjectArray):
            raise TypeError(
                "SuctionInput fused_objects must be a FusedObjectArray"
            )

        # Construct the coordinator-approved scoring request.
        message = SuctionInput()

        message.scene_id = scene_id.strip()
        message.fused_objects = fused_objects

        # Publish once when requested by SceneTransaction.
        self.suction_input_pub.publish(message)

        self.node.get_logger().info(
            f"Published SuctionInput: "
            f"scene_id={message.scene_id}; "
            f"scene_index={message.fused_objects.scene_index}; "
            f"objects={len(message.fused_objects.objects)}"
        )

    def get_latest_observations(self):
        """Return a snapshot of the latest observations."""

        return self.observations.get_latest()

    def find_realsense_frame_for_sam2(
        self,
        mask_stamp: ObservationStamp,
        scene_started_monotonic: float,
    ):
        """Find the RealSense frame associated with a SAM2 result."""

        return self.observations.find_realsense_by_stamp(
            stamp=mask_stamp,
            not_before_monotonic=scene_started_monotonic,
        )

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
    # M2.3C.3 — Fusion Result Tracking
    # ==============================================

    def expect_fusion_result(self, scene_id: str) -> None:
        """
        Register the transaction whose fusion result
        we are expecting.

        Must be called BEFORE publishing FusionInput.
        """

        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("Expected fusion scene_id must be nonempty")

        with self._fusion_result_lock:
            self._expected_fusion_scene_id = scene_id.strip()

            # Clear any result from the previous transaction.
            self._latest_fusion_result = None


    def get_fusion_result(self, scene_id: str):
        """
        Return the result belonging to the expected scene.

        Return None if no matching result has arrived.
        """

        with self._fusion_result_lock:

            if scene_id != self._expected_fusion_scene_id:
                return None

            return self._latest_fusion_result


    def clear_expected_fusion_result(self, scene_id: str) -> None:
        """
        Release the retained result after the transaction
        has finished or been cancelled.
        """

        with self._fusion_result_lock:

            if scene_id == self._expected_fusion_scene_id:
                self._expected_fusion_scene_id = None
                self._latest_fusion_result = None

    # ==============================================
    # Suction Result Tracking
    # ==============================================

    def expect_suction_result(self, scene_id: str) -> None:
        """
        Register the transaction whose suction result
        we are expecting.

        Must be called BEFORE publishing SuctionInput.
        """

        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError(
                "Expected suction scene_id must be nonempty"
            )

        with self._suction_result_lock:

            self._expected_suction_scene_id = scene_id.strip()

            # Remove the result from any previous transaction.
            self._latest_suction_result = None

    def get_suction_result(self, scene_id: str):
        """
        Return the retained result for the expected transaction.

        Return None if no matching result has arrived.
        """

        with self._suction_result_lock:

            if scene_id != self._expected_suction_scene_id:
                return None

            return self._latest_suction_result

    def clear_expected_suction_result(self, scene_id: str) -> None:
        """
        Release the retained suction result after
        transaction completion, failure, or cancellation.
        """

        with self._suction_result_lock:

            if scene_id == self._expected_suction_scene_id:

                self._expected_suction_scene_id = None
                self._latest_suction_result = None

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
            payload=message,
        )

    def _masks_callback(self, message):

        self._record_observation(
            source="sam2",
            header=message.header,
            scene_index=int(message.scene_index),
            object_count=len(message.objects),
            payload=message,
        )

    def _fusion_callback(self, message):

        self._record_observation(
            source="mask_fusion",
            header=message.header,
            scene_index=int(message.scene_index),
            object_count=len(message.objects),
        )

    def _fusion_result_callback(self, message: FusionResult):
        """
        Receive a transaction-correlated Mask Fusion result.

        Ignore results from unrelated scenes and prevent
        duplicate results from overwriting the first result.
        """

        with self._fusion_result_lock:

            # Reject results that bel   ong to another transaction.
            if message.scene_id != self._expected_fusion_scene_id:
                return

            # Do not replace a result already accepted.
            if self._latest_fusion_result is not None:
                return

            # Retain the exact FusionResult message.
            self._latest_fusion_result = message

        self.node.get_logger().info(
            f"Received FusionResult: "
            f"scene_id={message.scene_id}; "
            f"scene_index={message.scene_index}; "
            f"success={message.success}"
        )

    # ==============================================
    # SuctionResult Callback
    # ==============================================

    def _suction_result_callback(
        self,
        message: SuctionResult,
    ):
        """
        Receive a transaction-correlated Sim-Suction result.

        Ignore unrelated transactions and prevent
        duplicate results from overwriting the first.
        """

        with self._suction_result_lock:

            # Reject results from unrelated transactions.
            if message.scene_id != self._expected_suction_scene_id:
                return

            # First matching result wins.
            if self._latest_suction_result is not None:
                return

            # Retain the full SuctionResult message.
            self._latest_suction_result = message

        self.node.get_logger().info(
            f"Received SuctionResult: "
            f"scene_id={message.scene_id}; "
            f"scene_index={message.candidates.scene_index}; "
            f"status={message.candidates.status}; "
            f"candidates={len(message.candidates.candidates)}"
        )