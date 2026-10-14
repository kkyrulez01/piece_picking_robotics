import copy
import math

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, PointCloud2

from mecheye_ros_interface.srv import CaptureColorImage, CapturePointCloud

class PerceptionManager:
    def __init__(
        self,
        node,
        callback_group=None,
        obstacle_cloud_topic="/moma/obstacle_cloud",
        timeout_sec=15.0,
    ):
        self.node = node
        self.callback_group = callback_group

        self.timeout_timer = None
        self.timeout_sec = timeout_sec

        self.pointcloud_request_time_ns = None

        # Outputs waiting to be paired by capture timestamp.
        self.pending_pick_points = {}
        self.pending_obstacle_clouds = {}

        self.max_pending_entries = 8

        # Topic names
        self.mask_topic = "/moma/selected_box_mask"
        self.pick_point_topic = "/moma/pick_point_base"
        self.obstacle_cloud_topic = obstacle_cloud_topic

        # ROS Clients
        self.capture_image_client = self.node.create_client(
            CaptureColorImage,
            "/capture_color_image",
            callback_group=self.callback_group,
        )

        self.capture_pointcloud_client = self.node.create_client(
            CapturePointCloud,
            "/capture_point_cloud",
            callback_group=self.callback_group,
        )

        # ROS Subscriptions
        self.mask_subscription = self.node.create_subscription(
            Image,
            self.mask_topic,
            self._mask_callback,
            10,
            callback_group=self.callback_group,
        )

        self.pick_point_subscription = self.node.create_subscription(
            PointStamped,
            self.pick_point_topic,
            self._pick_point_callback,
            10,
            callback_group=self.callback_group,
        )

        self.obstacle_cloud_subscription = self.node.create_subscription(
            PointCloud2,
            self.obstacle_cloud_topic,
            self._obstacle_cloud_callback,
            10,
            callback_group=self.callback_group,
        )

        # Simple control flags
        # True while perception cycle is running
        self.active = False

        # Wait for YOLO to publish a mask from new RGB image
        self.waiting_for_mask = False

        # Waiting for outputs produced from new point cloud
        self.waiting_for_pointcloud_results = False

        # Point cloud service completed
        self.pointcloud_capture_complete = False

        # Fresh downstream results
        self.pick_point_received = False
        self.obstacle_cloud_received = False

        # Whole next-cycle perception pipeline is ready
        self.next_cycle_ready = False

        # Fresh perception completed successfully, but no boxes remaining
        self.no_box_detected = False

        # Freshness counters
        self.mask_generation = 0
        self.pick_point_generation = 0
        self.obstacle_cloud_generation = 0

        self.mask_baseline = 0
        self.pick_point_baseline = 0
        self.obstacle_cloud_baseline = 0

        # Cached next-cycle data
        self.next_pick_point = None
        self.next_obstacle_cloud = None

        # Completion callback
        self.result_ready_callback = None

        self.node.get_logger().info("PerceptionManager initialized.")

    # ============================================================
    # Capture Image
    # ============================================================
    def start_capture(self, result_ready_callback=None):
        """
        Starts one perception cycle. 
        Returns:
            True -> Perception successfully started
            False -> Perception could not be started
        """
        if self.active:
            self.node.get_logger().warn(
                "Cannot start perception: "
                "another perception cycle is already active."
            )
            return False

        # Check if required services exist
        if not self._services_ready():
            return False

        # Reset one-cycle state
        self.active = True

        self.waiting_for_mask = True
        self.waiting_for_pointcloud_results = False

        self.pointcloud_capture_complete = False

        self.pick_point_received = False
        self.obstacle_cloud_received = False

        self.next_cycle_ready = False
        self.next_pick_point = None
        self.next_obstacle_cloud = None
        self.no_box_detected = False

        self.pending_pick_points.clear()
        self.pending_obstacle_clouds.clear()

        self.pointcloud_request_time_ns = None

        self.result_ready_callback = result_ready_callback

        # Record before requesting new image
        self.mask_baseline = self.mask_generation

        self._start_timeout()

        self.node.get_logger().info("Starting next-cycle perception.")

        self._call_service(
            self.capture_image_client,
            CaptureColorImage.Request(),
            "RGB capture"
        )

        return True

    def obstacle_cloud_is_ready(self):
        return self.obstacle_cloud_received

    def is_active(self):
        return self.active

    def is_ready(self):
        return (self.next_cycle_ready
                and self.next_pick_point is not None
                and self.next_obstacle_cloud is not None)

    def is_no_box_detected(self):
        return self.no_box_detected
    
    def get_pick_point(self):
        if not self.is_ready():
            return None

        return copy.deepcopy(self.next_pick_point)

    def get_obstacle_cloud(self):
        if not self.is_ready():
            return None

        return self.next_obstacle_cloud
    
    def mark_cache_consumed(self):
        self.next_pick_point = None
        self.next_obstacle_cloud = None

        self.pick_point_received = False
        self.obstacle_cloud_received = False

        self.next_cycle_ready = False
        self.no_box_detected = False

    # ============================================================
    # MASK
    # ============================================================
    def _mask_callback(self, msg):
        self.mask_generation += 1

        if not self.active:
            return

        if not self.waiting_for_mask:
            return

        if self.mask_generation <= self.mask_baseline:
            return

        self.waiting_for_mask = False

        # Check whether YOLO actually detected a box
        if self._mask_is_empty(msg):
            self.node.get_logger().info("Fresh perception result: no boxes detected.")

            self._finish_no_box()
            return

        # If boxes left
        self.node.get_logger().info("Fresh YOLO mask received.")

        # Mask is ready, now capture corresponding point cloud
        self._capture_pointcloud()

    # ============================================================
    # POINT CLOUD
    # ============================================================
    def _capture_pointcloud(self):
        # Record baselines BEFORE requesting capture.
        self.pick_point_baseline = self.pick_point_generation
        self.obstacle_cloud_baseline = self.obstacle_cloud_generation

        self.waiting_for_pointcloud_results = True

        # Request for pointcloud
        future = self.capture_pointcloud_client.call_async(CapturePointCloud.Request())

        future.add_done_callback(self._pointcloud_capture_response)

        self.node.get_logger().info("Point-cloud capture requested.")

    def _pointcloud_capture_response(self, future):

        if not self.active:
            return
        
        try:
            response = future.result()

        except Exception as exception:
            self._fail(
                f"Point-cloud capture failed: {exception}"
            )
            return

        if response is None:
            self._fail("Point-cloud capture returned no response.")
            return

        if hasattr(response, "success") and not response.success:
            self._fail("Point-cloud capture returned success=False.")
            return

        self.pointcloud_capture_complete = True

        self.node.get_logger().info("Point-cloud capture completed.")

        self._try_finish()

    # ============================================================
    # PERCEPTION RESULTS
    # ============================================================
    def _pick_point_callback(self, msg):
        self.pick_point_generation += 1

        if not self.active or not self.waiting_for_pointcloud_results:
            return

        if self.pick_point_generation <= self.pick_point_baseline:
            return

        if self.pick_point_received:
            return

        self.next_pick_point = copy.deepcopy(msg)
        self.pick_point_received = True

        self.node.get_logger().info(
            "Fresh next-cycle pick point received."
        )

        self._try_finish()

    def _obstacle_cloud_callback(self, msg):
        self.obstacle_cloud_generation += 1

        if not self.active or not self.waiting_for_pointcloud_results:
            return

        # if self.obstacle_cloud_generation <= self.obstacle_cloud_baseline:
        #     return

        # if self.obstacle_cloud_received:
        #     return

        if not msg.data:
            self.node.get_logger().warn("Fresh obstacle cloud is empty.")
            return
        
        # Cache actual PointCloud2. Do not send it to MoveIt.
        self.next_obstacle_cloud = msg

        self.obstacle_cloud_received = True

        self.node.get_logger().info("Fresh next-cycle obstacle cloud received.")

        self._try_finish()

    # ============================================================
    # COMPLETION
    # ============================================================
    def _try_finish(self):

        if not self.pointcloud_capture_complete:
            return
        if not self.pick_point_received:
            return
        if not self.obstacle_cloud_received:
            return

        self._cancel_timeout()

        # Reset
        self.active = False
        self.waiting_for_pointcloud_results = False
        self.next_cycle_ready = True

        self.node.get_logger().info("Next-cycle perception ready.")

        callback = self.result_ready_callback
        self.result_ready_callback = None

        if callback is not None:
            callback(
                True,
                "Next-cycle perception ready.",
            )

    def _finish_no_box(self):
        if not self.active:
            return

        self._cancel_timeout()
        # Perception cycle successfully completed
        self.active = False

        self.waiting_for_mask = False
        self.waiting_for_pointcloud_results = False

        # No pick pointcloud for this cycle since no more boxes
        self.pointcloud_capture_complete = False
        self.pick_point_received = False
        self.obstacle_cloud_received = False

        self.next_cycle_ready = False
        self.next_pick_point = None
        self.next_obstacle_cloud = None

        # Important
        self.no_box_detected = True
        self.node.get_logger().info("Perception cycle completed: NO BOXES detected.")

        callback = self.result_ready_callback
        self.result_ready_callback = None

        if callback is not None:
            callback(True, "No boxes detected.")
    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    def _services_ready(self):

        services = [
            (self.capture_image_client,"/capture_color_image"),
            (self.capture_pointcloud_client,"/capture_point_cloud"),
        ]

        for client, name in services:
            if not client.service_is_ready():
                self.node.get_logger().error(f"Required service unavailable: {name}")
                return False

        return True

    def _call_service(
        self,
        client,
        request,
        description,
    ):
        try:
            future = client.call_async(request)

        except Exception as exception:
            self._fail(
                f"{description} request failed: {exception}"
            )
            return

        future.add_done_callback(
            lambda future: self._simple_service_response(
                future,
                description,
            )
        )

    def _simple_service_response(
        self,
        future,
        description,
    ):

        try:
            response = future.result()

        except Exception as exception:
            self._fail(f"{description} failed: {exception}")
            return

        if response is None:
            self._fail(f"{description} returned no response.")
            return

        if hasattr(response, "success") and not response.success:
            self._fail(f"{description} returned success=False.")

    def _fail(self, message):
        if not self.active:
            return

        self._cancel_timeout()

        self.node.get_logger().error(f"Perception failed: {message}")

        self.active = False
        self.waiting_for_mask = False
        self.waiting_for_pointcloud_results = False

        self.next_cycle_ready = False

        callback = self.result_ready_callback
        self.result_ready_callback = None

        if callback is not None:
            callback(False, message)

    def _mask_is_empty(self, msg):
        if not msg.data:
            return True

        return not any(msg.data)
    # ============================================================
    # TIMEOUT
    # ============================================================
    def _start_timeout(self):

        self._cancel_timeout()

        if self.timeout_sec <= 0.0:
            return

        self.timeout_timer = self.node.create_timer(
            self.timeout_sec,
            self._timeout_callback,
            callback_group=self.callback_group,
        )

    def _timeout_callback(self):

        self._cancel_timeout()

        if not self.active:
            return

        if self.waiting_for_mask:
            detail = "waiting for fresh YOLO mask"

        elif self.waiting_for_pointcloud_results:
            missing = []

            if not self.pointcloud_capture_complete:
                missing.append("point-cloud capture response")

            if not self.pick_point_received:
                missing.append("pick point")

            if not self.obstacle_cloud_received:
                missing.append("obstacle cloud" )

            detail = (
                "waiting for "
                + ", ".join(missing)
            )

        else:
            detail = ("waiting for perception response")

        self._fail(
            f"Perception timed out after "
            f"{self.timeout_sec:.1f} s while "
            f"{detail}."
        )

    def _cancel_timeout(self):
        if self.timeout_timer is None:
            return

        self.timeout_timer.cancel()

        self.node.destroy_timer(self.timeout_timer)

        self.timeout_timer = None

    @staticmethod
    def _stamp_to_ns(stamp):
        return (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )


    def _trim_pending(self, cache):
        while len(cache) > self.max_pending_entries:
            oldest_stamp = next(iter(cache))
            cache.pop(oldest_stamp, None)