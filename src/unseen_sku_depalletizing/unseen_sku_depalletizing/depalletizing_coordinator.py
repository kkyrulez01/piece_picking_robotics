#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Trigger, Empty

from moveit_msgs.srv import GetPlanningScene

from moma_motion_planning.srv import PlanPose

from moma_depalletizing.depalletizing_state import (
    ExecutionMode,
    DepalletizingPhase,
)

from moma_depalletizing.pick_phase import PickPhase
from moma_depalletizing.place_phase import PlacePhase
from moma_depalletizing.complete_phase import CompletePhase
from moma_depalletizing.perception_manager import PerceptionManager
from moma_depalletizing.vacuum_controller import VacuumController

class DepalletizingCoordinator(Node):

    def __init__(self):
        super().__init__("depalletizing_coordinator")

        self.get_logger().info("Starting MOMA Depalletizing Coordinator...")

        # Callback group
        self.callback_group = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter("execution_mode", "manual")

        self.declare_parameter("base_frame", "base_link")

        # ============================================================
        # PICK parameters
        # ============================================================
        self.declare_parameter("pre_pick_offset", 0.05)

        # OctoMap
        self.declare_parameter("require_octomap", True,)
        self.declare_parameter("octomap_retry_interval_ms", 200)
        self.declare_parameter("max_octomap_retries", 10)

        # Perception manager related
        self.declare_parameter("perception_timeout_sec",15.0,)
        self.perception_timeout_sec = float(self.get_parameter("perception_timeout_sec").value)

        # Vacuum related
        self.declare_parameter("vacuum_do_pin", 0)
        self.declare_parameter("vacuum_build_wait_sec", 1.0)

        # Fixed top-down TCP orientation
        self.declare_parameter("tcp_qx", 1.0)
        self.declare_parameter("tcp_qy", 0.0)
        self.declare_parameter("tcp_qz",0.0)
        self.declare_parameter("tcp_qw", 0.0)

        # Read parameters
        execution_mode_string = self.get_parameter("execution_mode").get_parameter_value().string_value
        self.execution_mode = self._parse_execution_mode(execution_mode_string)

        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value

        self.pre_pick_offset = self.get_parameter("pre_pick_offset").get_parameter_value().double_value

        self.require_octomap = self.get_parameter("require_octomap").get_parameter_value().bool_value
        self.octomap_retry_interval_ms = self.get_parameter("octomap_retry_interval_ms").get_parameter_value().integer_value
        self.max_octomap_retries = self.get_parameter("max_octomap_retries").get_parameter_value().integer_value

        self.vacuum_build_wait_sec = self.get_parameter("vacuum_build_wait_sec").get_parameter_value().double_value
        self.vacuum_do_pin = self.get_parameter("vacuum_do_pin").get_parameter_value().integer_value

        self.tcp_orientation = {
            "x": self.get_parameter("tcp_qx").value,
            "y": self.get_parameter("tcp_qy").value,
            "z": self.get_parameter("tcp_qz").value,
            "w": self.get_parameter("tcp_qw").value,
        }

        # ============================================================
        # PLACE parameters
        # ============================================================

        self.declare_parameter("place_x", -0.135116)
        self.declare_parameter("place_y", 0.291513)
        self.declare_parameter("place_z", 0.027340)

        self.declare_parameter("pre_place_offset", 0.15)

        self.declare_parameter("place_qx", 0.983550)
        self.declare_parameter("place_qy", -0.179957)
        self.declare_parameter("place_qz", 0.008523)
        self.declare_parameter("place_qw", -0.013122)

        self.declare_parameter("vacuum_release_wait_sec", 0.6)

        # Read parameters
        self.place_x = (
            self.get_parameter("place_x").get_parameter_value().double_value
        )

        self.place_y = (
            self.get_parameter("place_y").get_parameter_value().double_value
        )

        self.place_z = (
            self.get_parameter("place_z").get_parameter_value().double_value
        )

        self.pre_place_offset = (
            self.get_parameter("pre_place_offset").get_parameter_value().double_value
        )

        self.vacuum_release_wait_sec = (
            self.get_parameter("vacuum_release_wait_sec").get_parameter_value().double_value
        )

        self.place_orientation = {
            "x": self.get_parameter("place_qx").value,
            "y": self.get_parameter("place_qy").value,
            "z": self.get_parameter("place_qz").value,
            "w": self.get_parameter("place_qw").value,
        }

        # ============================================================
        # COMPLETE / HOME parameters
        # ============================================================

        self.declare_parameter("home_x", 0.0)
        self.declare_parameter("home_y", 0.0)
        self.declare_parameter("home_z", 0.0)

        self.declare_parameter("home_orientation.x", 0.0)
        self.declare_parameter("home_orientation.y", 0.0)
        self.declare_parameter("home_orientation.z", 0.0)
        self.declare_parameter("home_orientation.w", 1.0)

        self.home_x = self.get_parameter("home_x").value
        self.home_y = self.get_parameter("home_y").value
        self.home_z = self.get_parameter("home_z").value

        self.home_orientation = {
            "x": self.get_parameter("home_orientation.x").value,
            "y": self.get_parameter("home_orientation.y").value,
            "z": self.get_parameter("home_orientation.z").value,
            "w": self.get_parameter("home_orientation.w").value,
        }
        # State, start from IDLE
        self.phase = DepalletizingPhase.PICK

        # Perception manager
        self.perception = PerceptionManager(
            node=self,
            callback_group=self.callback_group,
            obstacle_cloud_topic="/moma/obstacle_cloud",
            timeout_sec=self.perception_timeout_sec,
        )

        self.waiting_for_next_perception = False
        self.next_perception_error = None
        self.preparing_pick_from_cache = False

        self.pending_cached_pick_point = None
        self.pending_cached_obstacle_cloud = None

        # Vacuum Controller
        self.vacuum = VacuumController(
            node=self,
            do_pin=self.vacuum_do_pin
        )

        # Phase objects
        self.pick_phase = PickPhase(coordinator=self)
        self.place_phase = PlacePhase(coordinator=self)
        self.complete_phase = CompletePhase(coordinator=self)

        # Publishers
        # Publish cached obstacle-cloud gate for MoveIt
        self.moveit_obstacle_cloud_publisher = (
            self.create_publisher(
                PointCloud2,
                "/moma/obstacle_cloud_for_moveit",
                10,
            )
        )

        # For debugging
        self.active_pick_point_publisher = self.create_publisher(
            PointStamped,
            "/moma/active_pick_point",
            10,
        )
        
        # Subscribers
        self.execution_result_subscriber = self.create_subscription(
            Bool,
            "/moma/arc_trajectory_execution_result",
            self.trajectory_execution_result_callback,
            10,
            callback_group=self.callback_group,
        )

        # MoveIt planning-scene service
        self.planning_scene_client = self.create_client(
            GetPlanningScene,
            "/get_planning_scene",
            callback_group=self.callback_group,
        )

        # Trajectory trigger service
        self.execute_trajectory_client = self.create_client(
            Trigger,
            "/moma/execute_latest_arc_trajectory",
            callback_group=self.callback_group
        )

        # Clear OctoMap service
        self.clear_octomap_client = self.create_client(
            Empty,
            "/clear_octomap",
            callback_group=self.callback_group,
        )

        # C++ MotionPlanner service
        #
        # Python Coordinator
        #       ↓
        # /moma/plan_pose
        #       ↓
        # C++ MotionPlanner
        #       ↓
        # Pilz PTP / LIN
        self.motion_plan_client = self.create_client(
            PlanPose,
            "/moma/plan_pose",
            callback_group=self.callback_group
        )

        # Manual phase-start services
        self.start_pick_service = self.create_service(
            Trigger,
            "/moma/start_pick_phase",
            self.start_pick_callback,
            callback_group=self.callback_group,
        )

        self.start_place_service = self.create_service(
            Trigger,
            "/moma/start_place_phase",
            self.start_place_callback,
            callback_group=self.callback_group,
        )

        # For automatic startup perception
        self.initial_perception_started = False

        self.initial_perception_timer = self.create_timer(
            0.25,
            self._try_start_initial_perception,
            callback_group=self.callback_group,
        )

        # Startup information for debugging
        self.get_logger().warn(f"Execution mode: {self.execution_mode.name}")
        self.get_logger().info(f"Base frame: {self.base_frame}")
        self.get_logger().info("Coordinator ready. Waiting for pick point.")

    # ============================================================
    # INITIAL PERCEPTION
    # ============================================================

    def _try_start_initial_perception(self):

        if self.initial_perception_started:
            return

        if not self.perception._services_ready():
            return

        self.get_logger().info(
            "Perception services ready. "
            "Starting initial capture."
        )

        started = self.perception.start_capture(
            result_ready_callback=(self._initial_perception_result))

        if not started:
            return

        self.initial_perception_started = True

        if self.initial_perception_timer is not None:
            self.initial_perception_timer.cancel()

            self.destroy_timer(self.initial_perception_timer)

            self.initial_perception_timer = None

    def _initial_perception_result(
        self,
        success,
        message,
    ):
        if not success:
            self.fail(f"Initial perception failed: {message}")

            return

        # Pallet already empty
        if self.perception.is_no_box_detected():
            self.get_logger().info("Initial perception found no boxes.")
            self.enter_complete("No boxes detected during initial perception.")

            return

        # Box detected
        if not self.perception.is_ready():
            self.fail("Initial perception succeeded, but cached PICK data not ready.")
            return
        
        self.get_logger().info("Initial perception cached successfully.")
        self.prepare_pick_from_cache()

    # ============================================================
    # Execution mode
    # ============================================================

    def _parse_execution_mode(self, mode_string):
        mode = mode_string.strip().lower()

        if mode == "manual":
            return ExecutionMode.MANUAL

        if mode == "auto":
            return ExecutionMode.AUTO

        raise ValueError(
            f"Invalid execution_mode '{mode_string}'. "
            f"Expected 'manual' or 'auto'."
        )
    # ============================================================
    # Common failure handler
    # ============================================================

    def fail(self, message):
        self.get_logger().error(message)

        self.phase = DepalletizingPhase.ERROR

        self.get_logger().error("Coordinator -> ERROR")

    # ============================================================
    # MOTION REQUEST
    #
    # Common Python -> C++ motion-planner bridge.
    # ============================================================

    def request_motion(
        self,
        pose,
        motion_type,
        callback
    ):
        motion_type = motion_type.strip().upper()

        if motion_type not in ("PTP", "LIN"):
            message = (
                f"Invalid motion type '{motion_type}'. "
                f"Expected 'PTP' or 'LIN'."
            )

            self.get_logger().error(message)

            callback(False,message,)

            return

        # Check planner service
        if not self.motion_plan_client.service_is_ready():
            message = (
                "MotionPlanner service "
                "'/moma/plan_pose' is unavailable."
            )

            self.get_logger().error(message)

            callback(False, message,)

            return

        # Build service request
        request = PlanPose.Request()

        request.target_pose = pose
        request.motion_type = motion_type

        self.get_logger().info(
            f"Requesting {motion_type} motion to: "
            f"x={pose.pose.position.x:.4f}, "
            f"y={pose.pose.position.y:.4f}, "
            f"z={pose.pose.position.z:.4f}"
        )

        # Send asynchronously
        future = self.motion_plan_client.call_async(
            request
        )

        future.add_done_callback(
            lambda future:
                self._motion_plan_response(
                    future,
                    motion_type,
                    callback,
                )
        )

    # ============================================================
    # MotionPlanner service response
    # ============================================================
    def _motion_plan_response(
        self,
        future,
        motion_type,
        callback,
    ):
        try:
            response = future.result()

        except Exception as exception:
            message = (
                f"{motion_type} planning service call failed: {exception}"
            )

            self.get_logger().error(message)
            callback(False, message)

            return

        if response is None:
            message = f"{motion_type} planning returned no response."

            self.get_logger().error(message)
            callback(False, message)

            return

        if not response.success:
            message = (
                response.message
                if response.message
                else f"{motion_type} planning failed."
            )

            self.get_logger().error(f"{motion_type} planning failed: {message}")
            callback(False, message)

            return

        message = (
            response.message
            if response.message
            else f"{motion_type} trajectory planned successfully."
        )

        self.get_logger().info(f"{motion_type} planning successful.")

        callback(True, message)

    # ============================================================
    # Send a trigger to execute planned trajectory
    # ============================================================
    def execute_latest_trajectory(self, callback):
        if not self.execute_trajectory_client.service_is_ready():
            self.fail(
                "ARC trajectory execution service is not ready."
            )
            return False

        request = Trigger.Request()

        future = self.execute_trajectory_client.call_async(
            request
        )

        future.add_done_callback(
            lambda future: callback(future)
        )

        return True
    
    # ============================================================
    # Physical trajectory execution result
    #
    # This result comes from arc_trajectory_executor.py after
    # FollowJointTrajectory actually completes.
    # ============================================================

    def trajectory_execution_result_callback(
        self,
        message,
    ):
        if not message.data:
            self.fail("Physical trajectory execution FAILED.")

            return

        if self.phase == DepalletizingPhase.PICK:
            self.pick_phase.handle_execution_success()

            return

        if self.phase == DepalletizingPhase.PLACE:
            self.place_phase.handle_execution_success()

            return

        if self.phase == DepalletizingPhase.COMPLETE:
            self.complete_phase.handle_execution_success()

            return

        self.get_logger().warn(
            "Physical trajectory success received "
            f"in unexpected phase: {self.phase.name}"
        )

    # ============================================================
    # Manual PICK trigger
    # ============================================================

    def start_pick_callback(
        self,
        request,
        response,
    ):
        del request

        if self.execution_mode != ExecutionMode.MANUAL:
            response.success = False
            response.message = "Coordinator is not in MANUAL mode."

            return response

        if self.phase != DepalletizingPhase.PICK:
            response.success = False
            response.message = "Coordinator is not in PICK phase."

            return response

        success, message = self.pick_phase.start_manual()

        response.success = success
        response.message = message

        return response

    # ============================================================
    # Manual PLACE trigger
    # ============================================================

    def start_place_callback(
        self,
        request,
        response,
    ):
        del request

        # PLACE can only be started when the coordinator
        # has already transitioned into the PLACE phase.
        if self.phase != DepalletizingPhase.PLACE:
            response.success = False
            response.message = (
                "Cannot start PLACE: "
                f"coordinator is in {self.phase.name} phase."
            )

            return response

        # Prevent accidentally starting PLACE twice.
        if self.place_phase.state is not None:
            response.success = False
            response.message = (
                "Cannot start PLACE: "
                "PlacePhase is already active."
            )

            return response

        # The carton should still be held after PICK.
        if not self.vacuum.is_enabled():
            response.success = False
            response.message = (
                "Cannot start PLACE: vacuum is not ON."
            )

            return response

        self.get_logger().info(
            "Manual PLACE phase start requested."
        )

        self.place_phase.start_place()

        response.success = True
        response.message = (
            "PLACE phase started."
        )

        return response
    
    # ============================================================
    # PICK phase finished
    # ============================================================

    def pick_phase_finished(self):

        if self.phase != DepalletizingPhase.PICK:
            self.get_logger().warn(
                "pick_phase_finished() called while "
                "coordinator is not in PICK phase."
            )

            return

        self.get_logger().info("Coordinator received PICK phase completion.")

        # Transition to PLACE phase
        self.phase = DepalletizingPhase.PLACE
        self.get_logger().info("Coordinator phase -> PLACE")

        # # Start PLACE phase
        self.place_phase.start_place()

    # ============================================================
    # NEXT-CYCLE PERCEPTION
    # ============================================================
    def start_next_cycle_perception(self):

        if self.phase != DepalletizingPhase.PLACE:
            self.get_logger().warn(
                "Cannot start next-cycle perception: "
                f"coordinator is in {self.phase.name} phase."
            )
            return False

        # Clear any error from a previous perception attempt.
        self.next_perception_error = None

        self.get_logger().info(
            "Starting next-cycle perception during PLACE."
        )

        started = self.perception.start_capture(
            result_ready_callback=
                self._next_cycle_perception_result
        )

        if not started:
            message = (
                "Could not start next-cycle perception."
            )

            self.next_perception_error = message

            self.get_logger().error(message)

            return False

        return True

    def _next_cycle_perception_result(
        self,
        success,
        message,
    ):
        # Perception failed
        if not success:
            self.next_perception_error = message

            self.get_logger().error(
                "Next-cycle perception failed: "
                f"{message}"
            )

            # If PLACE has already completed, COMPLETE is currently
            # waiting for this perception result.
            if self.phase == DepalletizingPhase.COMPLETE:
                self.get_logger().info("COMPLETE was waiting for perception result.")

                self.complete_phase.check_for_more_objects()

                return
            
            # Otherwise allow the current PLACE to finish.
            # place_phase_finished() will handle the error later.
            self.get_logger().warn(
                "Current PLACE will continue. "
                "Perception failure will be handled in COMPLETE."
            )
            return
        
        # Perception succeeded
        self.next_perception_error = None

        # No more objects
        if self.perception.is_no_box_detected():
            self.get_logger().info("Next-cycle perception found no objects.")

        # Another object detected
        elif self.perception.is_ready():
            self.get_logger().info("Next-cycle perception found another object.")

        # Invalid perception result
        else:
            self.next_perception_error = ("Perception reported success, but neither"
            "a valid object nor an empty pallet was reported.")

            self.get_logger().error(self.next_perception_error)

            if self.phase == DepalletizingPhase.COMPLETE:
                self.complete_phase.check_for_more_objects()

            return

        # PLACE finished, COMPLETE waiting for result
        if self.phase == DepalletizingPhase.COMPLETE:
            self.complete_phase.check_for_more_objects()
            return

        # PLACE still executing
        if self.phase == DepalletizingPhase.PLACE:
            self.get_logger().info(
                "Next-cycle perception ready. "
                "Waiting for current PLACE to finish."
            )
            return

    # ============================================================
    # PREPARE PICK FROM CACHED PERCEPTION
    # ============================================================
    def prepare_pick_from_cache(self):

        if self.preparing_pick_from_cache:
            self.get_logger().warn("Cached PICK preparation already active.")
            return

        if not self.perception.is_ready():
            self.fail(
                "Cannot prepare PICK: perception cache is not ready.")

            return

        pick_point = self.perception.get_pick_point()

        obstacle_cloud = self.perception.get_obstacle_cloud()

        if pick_point is None:
            self.fail("Cannot prepare PICK: cached pick point is unavailable.")

            return

        if obstacle_cloud is None:
            self.fail("Cannot prepare PICK: cached obstacle cloud is unavailable.")

            return

        self.preparing_pick_from_cache = True

        self.pending_cached_pick_point = pick_point

        self.pending_cached_obstacle_cloud = obstacle_cloud

        # ========================================================
        # OctoMap disabled
        # ========================================================

        if not self.require_octomap:
            self._start_cached_pick_without_octomap()

            return

        # ========================================================
        # Normal path
        # Clear the OLD OctoMap before injecting the new cloud.
        # ========================================================

        if not self.clear_octomap_client.service_is_ready():
            self.preparing_pick_from_cache = False

            self.fail(
                "/clear_octomap service unavailable "
                "while preparing PICK."
            )

            return

        self.get_logger().info(
            "Clearing old MoveIt OctoMap "
            "before loading cached perception."
        )

        try:
            future = (
                self.clear_octomap_client.call_async(
                    Empty.Request()
                )
            )

        except Exception as exception:
            self.preparing_pick_from_cache = False
            self.fail(f"Failed to request OctoMap clear: {exception}")

            return

        future.add_done_callback(self._cached_pick_octomap_cleared)

    def _cached_pick_octomap_cleared(
        self,
        future,
    ):
        try:
            future.result()

        except Exception as exception:
            self.preparing_pick_from_cache = False

            self.fail(f"Failed to clear OctoMap before PICK: {exception}")

            return

        if self.pending_cached_pick_point is None:
            self.preparing_pick_from_cache = False

            self.fail("Cached pick point disappeared while preparing PICK.")

            return

        if self.pending_cached_obstacle_cloud is None:
            self.preparing_pick_from_cache = False

            self.fail("Cached obstacle cloud disappeared while preparing PICK.")

            return

        self.get_logger().info("Old OctoMap cleared.")

        # Enter PICK
        self.phase = DepalletizingPhase.PICK

        self.get_logger().info("Coordinator phase -> PICK")

        # PickPhase now receives the already-cached pick point.
        self.active_pick_point_publisher.publish(
            self.pending_cached_pick_point
        )
        
        started = (
            self.pick_phase.start_with_cached_perception(
                self.pending_cached_pick_point
            )
        )

        if not started:
            self.preparing_pick_from_cache = False

            self.fail("PickPhase could not start from cached perception.")

            return

        # Release cached cloud into MoveIt.

        self.moveit_obstacle_cloud_publisher.publish(self.pending_cached_obstacle_cloud)

        self.get_logger().info("Cached obstacle cloud published to MoveIt.")

        # The data now belongs to this PICK.
        self.perception.mark_cache_consumed()

        self.pending_cached_pick_point = None
        self.pending_cached_obstacle_cloud = None

        self.preparing_pick_from_cache = False

        # PickPhase verifies that MoveIt actually built the OctoMap.

        self.pick_phase.request_octomap_check()

    # ============================================================
    # PLACE phase finished
    # ============================================================
    def place_phase_finished(self):
        if self.phase != DepalletizingPhase.PLACE:
            self.get_logger().warn(
                "place_phase_finished() called while "
                "coordinator is not in PLACE phase."
            )
            return

        self.get_logger().info("Coordinator received PLACE phase completion.")

        self.enter_complete("PLACE finished. Checking for more objects.")

    # ============================================================
    # Enter COMPLETE phase
    # ============================================================

    def enter_complete(self, reason):
        self.complete_phase.reset()
        self.phase = DepalletizingPhase.COMPLETE

        self.get_logger().info(f"Coordinator phase -> COMPLETE: {reason}")
        self.complete_phase.start()

    def start_next_pick_from_complete(self):
        if self.phase != DepalletizingPhase.COMPLETE:
            self.get_logger().warn(
                "Cannot start next PICK: coordinator is not in COMPLETE phase."
            )
            return
        
        if not self.perception.is_ready():
            self.fail(
                "Cannot start next PICK: perception cache is not ready."
            )
            return

        self.get_logger().info("More objects remain. Preparing next PICK.")
        self.complete_phase.reset()

        self.prepare_pick_from_cache()

def main(args=None):
    rclpy.init(args=args)

    node = DepalletizingCoordinator()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()