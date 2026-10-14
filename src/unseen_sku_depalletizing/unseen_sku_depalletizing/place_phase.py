import math

from geometry_msgs.msg import PoseStamped

from moma_depalletizing.depalletizing_state import (
    DepalletizingPhase,
    PlaceState,
)

class PlacePhase:

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.node = coordinator

        # PLACE is inactive until start_place is called
        self.state = None

        # Stored PLACE poses
        self.pre_place_pose = None
        self.place_pose = None

        # Vacuum release delay
        self.vacuum_release_timer = None

        self.node.get_logger().info("PlacePhase initialized.")

    # ==============================================================
    # RESET
    # ==============================================================
    def reset(self):
        self._cancel_vacuum_release_timer()

        self.state = None

        self.pre_place_pose = None
        self.place_pose = None

        self.node.get_logger().info("PlacePhase reset")

    # ==============================================================
    # START PLACE
    # ==============================================================
    def start_place(self):
        if self.coordinator.phase != DepalletizingPhase.PLACE:
            self.node.get_logger().warn(
                "PlacePhase.start() called while coordinator "
                "is not in PLACE phase."
            )
            return

        # At this point PICK should have completed successfully,
        # therefore the carton should still be held by vacuum.
        if not self.coordinator.vacuum.is_enabled():
            self.coordinator.fail("cannot start PLACE phase: vacuum is not ON.")

            return

        self.node.get_logger().info("========================================")
        self.node.get_logger().info("Starting PLACE phase.")
        self.node.get_logger().info("========================================")

        if not self.generate_place_poses():
            return

        self.plan_pre_place()

    # ==============================================================
    # GENERATE PLACE / PRE-PLACE POSES
    # ==============================================================
    def generate_place_poses(self):

        place_x = self.coordinator.place_x
        place_y = self.coordinator.place_y
        place_z = self.coordinator.place_z

        if not (
            math.isfinite(place_x)
            and math.isfinite(place_y)
            and math.isfinite(place_z)
        ):
            self.coordinator.fail("Configured PLACE position contains non-finite values.")
            return False

        if (
            not math.isfinite(self.coordinator.pre_place_offset)
            or self.coordinator.pre_place_offset <= 0.0
        ):
            self.coordinator.fail("pre_place_offset must be finite and > 0.")
            return False

        # Normalize PLACE quaternion
        
        orientation = self.coordinator.place_orientation

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
            self.coordinator.fail("Configured PLACE quaternion is invalid.")
            return False

        # PLACE pose
        place_pose = PoseStamped()

        place_pose.header.frame_id = self.coordinator.base_frame
        place_pose.header.stamp = (self.node.get_clock().now().to_msg())

        place_pose.pose.position.x = place_x
        place_pose.pose.position.y = place_y
        place_pose.pose.position.z = place_z

        place_pose.pose.orientation.x = qx / quaternion_norm
        place_pose.pose.orientation.y = qy / quaternion_norm
        place_pose.pose.orientation.z = qz / quaternion_norm
        place_pose.pose.orientation.w = qw / quaternion_norm

        # PRE-PLACE pose
        pre_place_pose = PoseStamped()

        pre_place_pose.header = place_pose.header

        pre_place_pose.pose.position.x = place_pose.pose.position.x
        pre_place_pose.pose.position.y = place_pose.pose.position.y
        pre_place_pose.pose.position.z = place_pose.pose.position.z + self.coordinator.pre_place_offset

        pre_place_pose.pose.orientation = place_pose.pose.orientation

        self.place_pose = place_pose
        self.pre_place_pose = pre_place_pose

        self.node.get_logger().info("PLACE poses generated.")

        self.node.get_logger().info(
            f"Place pose: "
            f"[{self.place_pose.pose.position.x:.4f}, "
            f"{self.place_pose.pose.position.y:.4f}, "
            f"{self.place_pose.pose.position.z:.4f}]"
        )

        self.node.get_logger().info(
            f"Pre-place pose: "
            f"[{self.pre_place_pose.pose.position.x:.4f}, "
            f"{self.pre_place_pose.pose.position.y:.4f}, "
            f"{self.pre_place_pose.pose.position.z:.4f}]"
        )

        self.node.get_logger().info(
            f"Pre-place clearance: "
            f"{self.coordinator.pre_place_offset:.4f} m"
        )

        return True

    # ==============================================================
    # PLAN PRE-PLACE (PTP)
    # ==============================================================
    def plan_pre_place(self):
        if self.coordinator.phase != DepalletizingPhase.PLACE:
            self.node.get_logger().warn("plan_pre_place() called outside PLACE phase.")
            return

        if self.pre_place_pose is None:
            self.coordinator.fail("Cannot plan pre-place: pre_place_pose is unavailable.")
            return

        self.state = PlaceState.PLANNING_PRE_PLACE

        self.node.get_logger().info("Place state -> PLANNING_PRE_PLACE")

        self.coordinator.request_motion(
            pose=self.pre_place_pose,
            motion_type="PTP",
            callback=self._pre_place_plan_response,
        )

    def _pre_place_plan_response(
        self,
        success,
        message
    ):
        if self.state != PlaceState.PLANNING_PRE_PLACE:
            return

        if not success:
            self.coordinator.fail(
                f"Failed to plan pre-place motion: {message}"
            )
            return

        self.state = PlaceState.EXECUTING_PRE_PLACE

        self.node.get_logger().info("Place state -> EXECUTING_PRE_PLACE")

        self.node.get_logger().info(
            "Pre-place trajectory published. "
            "Waiting for physical execution completion."
        )

    # ==============================================================
    # PLAN PLACE (LIN)
    # ==============================================================
    def plan_place(self):
        if self.state != PlaceState.EXECUTING_PRE_PLACE:
            self.node.get_logger().warn("plan_place() called from unexpected PLACE state.")
            return

        if self.place_pose is None:
            self.coordinator.fail("Cannot plan PLACE motion: place_pose is unavailable.")
            return

        self.state = PlaceState.PLANNING_PLACE

        self.node.get_logger().info("Place state -> PLANNING_PLACE")

        self.coordinator.request_motion(
            pose=self.place_pose,
            motion_type="LIN",
            callback=self._place_plan_response,
        )

    def _place_plan_response(
        self,
        success,
        message
    ):
        if self.state != PlaceState.PLANNING_PLACE:
            return

        if not success:
            self.coordinator.fail(f"Failed to plan LIN PLACE motion: {message}")
            return

        self.state = PlaceState.EXECUTING_PLACE

        self.node.get_logger().info("Place state -> EXECUTING_PLACE")

        self.node.get_logger().info(
            "Place trajectory published. "
            "Waiting for physical execution completion."
        )

    # ==============================================================
    # RELEASE SUCTION
    # ==============================================================
    def release_suction(self):
        if self.state != PlaceState.RELEASING_SUCTION:
            self.coordinator.fail(
                "release_suction() called from unexpected PLACE state."
            )
            return

        # The vacuum should still be holding the carton.
        if not self.coordinator.vacuum.is_enabled():
            self.coordinator.fail(
                "Cannot release suction: vacuum is already OFF."
            )
            return

        self.node.get_logger().warn("========================================")
        self.node.get_logger().warn("PLACE: VACUUM OFF")
        self.node.get_logger().warn("========================================")

        self.coordinator.vacuum.release(callback=self._vacuum_off_response)

    def _vacuum_off_response(
        self,
        success,
        message
    ):
        if self.state != PlaceState.RELEASING_SUCTION:
            return

        if not success:
            self.coordinator.fail(f"Failed to deactivate vacuum: {message}")
            return

        self.node.get_logger().info("Vacuum OFF command accepted.")

        # Give the object time to physically seperate from suction
        if self.coordinator.vacuum_release_wait_sec <= 0.0:
            self.plan_retract()
            return

        self.node.get_logger().info(
            f"Waiting "
            f"{self.coordinator.vacuum_release_wait_sec:.2f} s "
            f"for carton release."
        )

        self._cancel_vacuum_release_timer()

        # Start vacuum release timer
        self.vacuum_release_timer = self.node.create_timer(
            self.coordinator.vacuum_release_wait_sec,
            self._vacuum_release_complete,
            callback_group=self.coordinator.callback_group,
        )

    def _vacuum_release_complete(self):
        self._cancel_vacuum_release_timer()

        if self.state != PlaceState.RELEASING_SUCTION:
            return

        # Vacuum controller should now agree that output is OFF.
        if self.coordinator.vacuum.is_enabled():
            self.coordinator.fail(
                "Vacuum release timer completed, "
                "but vacuum is still marked ON."
            )
            return

        self.node.get_logger().info("Vacuum release wait complete.")

        self.plan_retract()

    def _cancel_vacuum_release_timer(self):
        if self.vacuum_release_timer is None:
            return

        self.vacuum_release_timer.cancel()

        self.node.destroy_timer(
            self.vacuum_release_timer
        )

        self.vacuum_release_timer = None

    # ==============================================================
    # PLAN RETRACT (LIN)
    # ==============================================================
    def plan_retract(self):
        if self.state != PlaceState.RELEASING_SUCTION:
            self.node.get_logger().warn("plan_retract() called from unexpected PLACE state.")
            return

        if self.pre_place_pose is None:
            self.coordinator.fail(
                "Cannot plan PLACE retract: "
                "pre_place_pose is unavailable."
            )
            return

        # Do not leave place position if vacuum is still on
        if self.coordinator.vacuum.is_enabled():
            self.coordinator.fail(
                "Refusing PLACE retract because vacuum is still ON."
            )
            return

        self.state = PlaceState.PLANNING_RETRACT

        self.node.get_logger().info("Place state -> PLANNING_RETRACT")

        self.coordinator.request_motion(
            pose=self.pre_place_pose,
            motion_type="LIN",
            callback=self._retract_plan_response,
        )

    def _retract_plan_response(
        self,
        success,
        message
    ):
        if self.state != PlaceState.PLANNING_RETRACT:
            return

        if not success:
            self.coordinator.fail(f"Failed to plan PLACE retract motion: {message}")
            return

        self.state = PlaceState.EXECUTING_RETRACT

        self.node.get_logger().info("Place state -> EXECUTING_RETRACT")

        self.node.get_logger().info(
            "Place retract trajectory published. "
            "Waiting for physical execution completion."
        )

    # ==============================================================
    # PHYSICAL EXECUTION SUCCESS
    # ==============================================================
    def handle_execution_success(self):

        if self.coordinator.phase != DepalletizingPhase.PLACE:
            self.node.get_logger().warn(
                "PLACE execution success received while "
                "coordinator is not in PLACE phase."
            )
            return
        
        # PRE-PLACE (PTP) completed
        if self.state == PlaceState.EXECUTING_PRE_PLACE:

            self.node.get_logger().info("Pre-place execution completed successfully.")

            # Tell coordinator to start next-cycle perception asynchronously
            self.coordinator.start_next_cycle_perception()

            self.plan_place()

            return

        # PLACE (LIN) completed
        if self.state == PlaceState.EXECUTING_PLACE:

            self.node.get_logger().info("PLACE execution completed successfully.")

            self.state = PlaceState.RELEASING_SUCTION

            self.node.get_logger().info("Place state -> RELEASING_SUCTION")

            self.release_suction()

            return

        # PRE-PLACE retract (LIN) completed
        if self.state == PlaceState.EXECUTING_RETRACT:

            self.node.get_logger().info("PLACE retract execution completed successfully.")

            self.state = PlaceState.PLACE_FINISHED

            self.node.get_logger().info("Place state -> PLACE_FINISHED")

            self.node.get_logger().info("========================================")
            self.node.get_logger().info("PLACE phase completed successfully.")
            self.node.get_logger().info("========================================")

            self.coordinator.place_phase_finished()

            return

        # ----------------------------------------------------------
        # Unexpected result
        # ----------------------------------------------------------

        self.node.get_logger().warn(
            "Physical trajectory execution succeeded, "
            f"but PLACE is in unexpected state: "
            f"{self.state.name if self.state is not None else 'None'}"
        )