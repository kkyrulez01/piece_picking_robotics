import math
import copy

from geometry_msgs.msg import PointStamped, PoseStamped
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene

from std_srvs.srv import Empty

from moma_depalletizing.depalletizing_state import (
    DepalletizingPhase,
    PickState,
)

class PickPhase:

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.node = coordinator

        # PICK State
        self.state = PickState.WAITING_FOR_PICK_POINT

        # Stored PICK data
        self.pending_pick_point = None

        self.pre_pick_pose = None
        self.pick_pose = None

        # OctoMap
        self.octomap_retry_count = 0
        self.octomap_retry_timer = None
        self.planning_scene_request_in_flight = False

        # Vacuum build delay
        self.vacuum_build_timer = None

        self.node.get_logger().info("PickPhase initialized.")

    # ==============================================================
    # Reset
    # ==============================================================

    def reset(self):
        self._cancel_octomap_retry_timer()
        self._cancel_vacuum_build_timer()

        self.state = PickState.WAITING_FOR_PICK_POINT

        self.pending_pick_point = None

        self.pick_pose = None
        self.pre_pick_pose = None

        self.octomap_retry_count = 0
        self.planning_scene_request_in_flight = False

        self.node.get_logger().info("PickPhase reset. Waiting for pick point.")

    # ==============================================================
    # PICK POINT
    # ==============================================================

    def pick_callback(self,message: PointStamped):

        if self.state != PickState.WAITING_FOR_PICK_POINT:
            self.node.get_logger().warn(
                "PICK phase is already active. "
                "Ignoring new pick point."
            )
            return

        if message.header.frame_id != self.coordinator.base_frame:
            self.node.get_logger().error(
                f"Pick point frame '{message.header.frame_id}' "
                f"does not match expected frame "
                f"'{self.coordinator.base_frame}'."
            )
            return

        # Verify pick point coordinates
        if not (
            math.isfinite(message.point.x)
            and math.isfinite(message.point.y)
            and math.isfinite(message.point.z)
        ):
            self.node.get_logger().error("Received non-finite pick point.")

            return

        self.pending_pick_point = message

        self.octomap_retry_count = 0

        self.node.get_logger().info(
            f"Received pick point: "
            f"[{message.point.x:.4f}, "
            f"{message.point.y:.4f}, "
            f"{message.point.z:.4f}]"
        )

        # Pick point done, now wait for obstacle cloud
        self.state = PickState.WAITING_FOR_OBSTACLE_CLOUD

        self.node.get_logger().info(
            "Waiting for obstacle cloud..."
        )

    # ==============================================================
    # OBSTACLE CLOUD
    # ==============================================================
    
    def obstacle_cloud_callback(self, message):

        if self.state != PickState.WAITING_FOR_OBSTACLE_CLOUD:
            return

        if not message.data:
            self.node.get_logger().warn(
                "Received empty obstacle cloud. "
                "Waiting for another cloud."
            )
            return

        self.node.get_logger().info(
            f"Obstacle cloud received: "
            f"{message.width} x {message.height}, "
            f"frame='{message.header.frame_id}'"
        )

        if not self.coordinator.require_octomap:
            self.node.get_logger().warn(
                "OctoMap requirement disabled."
            )

            self.state = PickState.READY_FOR_PRE_PICK

            self.pick_ready()
            return

        # Obstacle cloud received, now wait for MoveIt OctoMap
        self.state = PickState.WAITING_FOR_OCTOMAP

        self.octomap_retry_count = 0

        self.request_octomap_check()

    # ==============================================================
    # OCTOMAP CHECK
    # ==============================================================

    def request_octomap_check(self):

        if (self.coordinator.phase != DepalletizingPhase.PICK):
            return

        if (self.state != PickState.WAITING_FOR_OCTOMAP):
            return

        # Prevent overlapping requests
        if self.planning_scene_request_in_flight:
            return

        # Check if /get_planning_scene service is ready
        if not self.coordinator.planning_scene_client.service_is_ready():
            self.node.get_logger().warn("/get_planning_scene is not ready yet.")

            self.schedule_octomap_retry()
            return

        request = GetPlanningScene.Request()

        # Only extract OctoMap component
        request.components.components = (PlanningSceneComponents.OCTOMAP)

        self.planning_scene_request_in_flight = True

        future =  self.coordinator.planning_scene_client.call_async(request)

        future.add_done_callback(self._octomap_response_callback)

    def _octomap_response_callback(self, future):

        self.planning_scene_request_in_flight = False

        if (self.coordinator.phase != DepalletizingPhase.PICK):
            return

        if (self.state != PickState.WAITING_FOR_OCTOMAP):
            return

        try:
            response = future.result()

        except Exception as exception:
            self.node.get_logger().error(
                f"Failed to obtain MoveIt planning scene: "
                f"{exception}"
            )

            self.schedule_octomap_retry()
            return

        octomap = response.scene.world.octomap.octomap

        if not octomap.data:
            self.node.get_logger().warn(
                "MoveIt OctoMap is still empty."
            )

            self.schedule_octomap_retry()
            return

        self.node.get_logger().info(
            f"MoveIt OctoMap is ready. "
        )

        self.octomap_retry_count = 0
        self._cancel_octomap_retry_timer()

        self.state = PickState.READY_FOR_PRE_PICK

        self.node.get_logger().info("Pick state -> READY_FOR_PRE_PICK")

        self.pick_ready()

    # ==============================================================
    # OCTOMAP RETRY
    # ==============================================================

    def schedule_octomap_retry(self):

        self.octomap_retry_count += 1

        if (self.octomap_retry_count > self.coordinator.max_octomap_retries):
            self.coordinator.fail(
                "MoveIt OctoMap did not become ready "
                "within the allowed retry period."
            )
            return

        self.node.get_logger().warn(
            f"Retrying OctoMap check in "
            f"{self.coordinator.octomap_retry_interval_ms} ms "
            f"(attempt {self.octomap_retry_count}/"
            f"{self.coordinator.max_octomap_retries})."
        )

        self._cancel_octomap_retry_timer()

        retry_seconds = self.coordinator.octomap_retry_interval_ms / 1000.0

        self.octomap_retry_timer = self.node.create_timer(
            retry_seconds,
            self._octomap_retry_timer_callback,
            callback_group=self.coordinator.callback_group,
        )

    def _octomap_retry_timer_callback(self):
        self._cancel_octomap_retry_timer()
        self.request_octomap_check()

    def _cancel_octomap_retry_timer(self):
        if self.octomap_retry_timer is None:
            return

        self.octomap_retry_timer.cancel()

        self.node.destroy_timer(self.octomap_retry_timer)

        self.octomap_retry_timer = None

    # Clear OctoMap before pick
    def clear_octomap_before_pick(self):

        if not self.coordinator.clear_octomap_client.service_is_ready():
            self.coordinator.fail(
                "/clear_octomap service is not available."
            )
            return

        self.node.get_logger().info(
            "Clearing OctoMap before LIN pick approach."
        )

        request = Empty.Request()

        future = self.coordinator.clear_octomap_client.call_async(request)

        future.add_done_callback(self._clear_octomap_before_pick_response)

    def _clear_octomap_before_pick_response(self, future):

        if self.state != PickState.EXECUTING_PRE_PICK:
            return

        try:
            future.result()

        except Exception as exception:
            self.coordinator.fail(
                f"Failed to clear OctoMap: {exception}"
            )
            return

        self.node.get_logger().info(
            "OctoMap cleared successfully."
        )

        self.node.get_logger().info(
            "Planning LIN approach from pre-pick to pick."
        )

        self.plan_pick()

    # ==============================================================
    # PICK READY
    #
    # AUTO / MANUAL execution is handled by ArcTrajectoryExecutor.
    # ==============================================================

    def pick_ready(self):

        if (self.state != PickState.READY_FOR_PRE_PICK):
            self.node.get_logger().warn("pick_ready() called before PICK environment was ready.")

            return
    
        self.node.get_logger().info(
            "PICK enviroment ready. Planning pre-pick trajectory..."
        )

        self.plan_pre_pick()

    # ==============================================================
    # GENERATE PICK / PRE-PICK POSES
    # ==============================================================

    def generate_pick_poses(self):

        if self.pending_pick_point is None:
            self.coordinator.fail("No pending pick point available.")

            return False

        orientation = self.coordinator.tcp_orientation

        qx = orientation["x"]
        qy = orientation["y"]
        qz = orientation["z"]
        qw = orientation["w"]

        quaternion_norm = math.sqrt(
            qx * qx
            + qy * qy
            + qz * qz
            + qw * qw
        )

        if quaternion_norm < 1e-6:
            self.coordinator.fail(
                "Configured TCP quaternion is invalid."
            )
            return False

        # Pick pose
        pick_pose = PoseStamped()

        pick_pose.header.frame_id = self.coordinator.base_frame
        pick_pose.header.stamp = self.node.get_clock().now().to_msg()

        pick_pose.pose.position.x = self.pending_pick_point.point.x
        pick_pose.pose.position.y = self.pending_pick_point.point.y
        pick_pose.pose.position.z = self.pending_pick_point.point.z

        pick_pose.pose.orientation.x = qx / quaternion_norm
        pick_pose.pose.orientation.y = qy / quaternion_norm
        pick_pose.pose.orientation.z = qz / quaternion_norm
        pick_pose.pose.orientation.w = qw / quaternion_norm

        # Pre-pick pose
        pre_pick_pose = PoseStamped()

        pre_pick_pose.header = pick_pose.header

        pre_pick_pose.pose.position.x = pick_pose.pose.position.x
        pre_pick_pose.pose.position.y = pick_pose.pose.position.y
        pre_pick_pose.pose.position.z = pick_pose.pose.position.z + self.coordinator.pre_pick_offset

        pre_pick_pose.pose.orientation = pick_pose.pose.orientation

        self.pick_pose = pick_pose
        self.pre_pick_pose = pre_pick_pose

        self.node.get_logger().info(
            f"Pick pose: "
            f"[{self.pick_pose.pose.position.x:.4f}, "
            f"{self.pick_pose.pose.position.y:.4f}, "
            f"{self.pick_pose.pose.position.z:.4f}]"
        )

        self.node.get_logger().info(
            f"Pre-pick pose: "
            f"[{self.pre_pick_pose.pose.position.x:.4f}, "
            f"{self.pre_pick_pose.pose.position.y:.4f}, "
            f"{self.pre_pick_pose.pose.position.z:.4f}]"
        )

        return True

    # ==============================================================
    # PLAN PRE-PICK
    # ==============================================================

    def plan_pre_pick(self):

        if self.state != PickState.READY_FOR_PRE_PICK:
            self.node.get_logger().warn("plan_pre_pick() called before PICK was ready.")

            return

        if not self.generate_pick_poses():
            return

        self.state = PickState.PLANNING_PRE_PICK

        self.node.get_logger().info("Planning PTP motion to pre-pick.")

        self.coordinator.request_motion(
            pose=self.pre_pick_pose,
            motion_type="PTP",
            callback=self._pre_pick_plan_response,
        )

    def _pre_pick_plan_response(
        self,
        success,
        message
    ):
        if self.state != PickState.PLANNING_PRE_PICK:
            return

        if not success:
            self.coordinator.fail(
                f"Failed to plan pre-pick motion: {message}")
            
            return

        self.node.get_logger().info(
            "Pre-pick trajectory planned and published. "
        )

        self.state = PickState.EXECUTING_PRE_PICK

        self.node.get_logger().info("Pre-pick trajectory published.")

        self.node.get_logger().info("Waiting for physical execution completion.")

    # ==============================================================
    # PLAN PICK
    # ==============================================================

    def plan_pick(self):

        if self.state != PickState.EXECUTING_PRE_PICK:
            self.node.get_logger().warn(
                "plan_pick() called from unexpected state."
            )
            return

        self.state = PickState.PLANNING_PICK

        self.node.get_logger().info(
            "Planning LIN motion from pre-pick to pick."
        )

        self.coordinator.request_motion(
            pose=self.pick_pose,
            motion_type="LIN",
            callback=self._pick_plan_response,
        )

    def _pick_plan_response(
        self,
        success,
        message,
    ):
        if self.state != PickState.PLANNING_PICK:
            return

        if not success:
            self.coordinator.fail(
                f"Failed to plan LIN pick motion: "
                f"{message}"
            )
            return

        self.state = PickState.EXECUTING_PICK

        self.node.get_logger().info(
            "Pick trajectory published. "
            "Waiting for robot execution completion."
        )

    # ==============================================================
    # VACUUM ON
    # ==============================================================

    def activate_suction(self):

        if self.state != PickState.ACTIVATING_SUCTION:
            self.coordinator.fail("activate_suction() called from unexpected PICK state.")

            return

        if self.coordinator.vacuum.is_enabled():
            self.coordinator.fail("Cannot activate suction: vacuum is already ON.")

            return

        self.node.get_logger().warn("PICK: VACUUM ON")

        self.coordinator.vacuum.activate(callback=self._vacuum_on_response)

    def _vacuum_on_response(
        self,
        success,
        message
    ):
        if self.state != PickState.ACTIVATING_SUCTION:
            return

        if not success:
            self.coordinator.fail(f"Failed to activate vacuum: {message}")

            return

        self.node.get_logger().info("Vacuum ON command accepted.")

        if self.coordinator.vacuum_build_wait_sec <= 0.0:
            self.plan_retract()
            return

        self.node.get_logger().info(
            f"Waiting "
            f"{self.coordinator.vacuum_build_wait_sec:.2f} s "
            f"for vacuum to build."
        )

        self._cancel_vacuum_build_timer()

        self.vacuum_build_timer = self.node.create_timer(
            self.coordinator.vacuum_build_wait_sec,
            self._vacuum_build_complete,
            callback_group=self.coordinator.callback_group,
        )

    def _vacuum_build_complete(self):
        self._cancel_vacuum_build_timer()

        if self.state != PickState.ACTIVATING_SUCTION:
            return

        if not self.coordinator.vacuum.is_enabled():
            self.coordinator.fail("Vacuum build timer completed, but vacuum is not marked ON.")
            return

        self.node.get_logger().info("Vacuum build wait complete.")

        self.plan_retract()

    def _cancel_vacuum_build_timer(self):

        if self.vacuum_build_timer is None:
            return

        self.vacuum_build_timer.cancel()

        self.node.destroy_timer(self.vacuum_build_timer)

        self.vacuum_build_timer = None

    # ==============================================================
    # PLAN RETRACT
    # ==============================================================

    def plan_retract(self):

        if self.state != PickState.ACTIVATING_SUCTION:
            self.node.get_logger().warn("plan_retract() called from unexpected state.")

            return

        self.state = PickState.PLANNING_RETRACT

        self.node.get_logger().info("Planning LIN retract from pick to pre-pick.")

        self.coordinator.request_motion(
            pose=self.pre_pick_pose,
            motion_type="LIN",
            callback=self._retract_plan_response,
        )

    def _retract_plan_response(
        self,
        success,
        message,
    ):
        if self.state != PickState.PLANNING_RETRACT:
            return

        if not success:
            self.coordinator.fail(f"Failed to plan retract motion: {message}")
            return

        self.state = PickState.EXECUTING_RETRACT

        self.node.get_logger().info(
            "Retract trajectory published. "
            "Waiting for robot execution completion."
        )

    # ==============================================================
    # Start with cached perception
    # ==============================================================
    def start_with_cached_perception(
        self,
        pick_point,
    ):

        if (self.coordinator.phase!= DepalletizingPhase.PICK):
            self.node.get_logger().error(
                "Cannot start PICK: coordinator is not in PICK phase."
            )

            return False

        self.reset()

        if pick_point is None:
            self.node.get_logger().error("Cached pick point is None.")

            return False

        self.pending_pick_point = copy.deepcopy(pick_point)

        self.octomap_retry_count = 0

        self.node.get_logger().info(
            "Starting PICK from cached perception: "
            f"x={pick_point.point.x:.4f}, "
            f"y={pick_point.point.y:.4f}, "
            f"z={pick_point.point.z:.4f}"
        )

        if not self.coordinator.require_octomap:
            self.state = PickState.READY_FOR_PRE_PICK

            self.plan_pre_pick()

            return True

        self.state = PickState.WAITING_FOR_OCTOMAP

        self.node.get_logger().info("Pick state -> WAITING_FOR_OCTOMAP")

        return True
    # ==============================================================
    # PHYSICAL EXECUTION SUCCESS
    # ==============================================================

    def handle_execution_success(self):

        if (self.coordinator.phase!= DepalletizingPhase.PICK):
            self.node.get_logger().warn(
                "PICK execution success received "
                "while coordinator is not in PICK phase."
            )
            return

        # Pre-pick (PTP) finished
        if self.state == PickState.EXECUTING_PRE_PICK:

            self.node.get_logger().info("Pre-pick execution completed successfully." )

            self.clear_octomap_before_pick()

            return

        # Pick descent (LIN) finished
        if self.state == PickState.EXECUTING_PICK:

            self.node.get_logger().info(
                "Pick execution completed successfully."
            )

            self.state = PickState.ACTIVATING_SUCTION

            self.node.get_logger().info("Pick state -> ACTIVATING_SUCTION")

            self.activate_suction()

            return

        # Retract finished
        if self.state == PickState.EXECUTING_RETRACT:

            self.node.get_logger().info(
                "Retract execution completed successfully."
            )

            self.state = PickState.PICK_FINISHED

            self.node.get_logger().info(
                "Pick state -> PICK_FINISHED"
            )

            self.node.get_logger().info(
                "PICK phase completed successfully."
            )

            self.coordinator.pick_phase_finished()
            return

        # Unexpected result
        self.node.get_logger().warn(
            "Physical trajectory execution succeeded, "
            f"but PICK is in unexpected state: "
            f"{self.state.name}"
        )