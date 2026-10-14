from geometry_msgs.msg import PoseStamped

from moma_depalletizing.depalletizing_state import DepalletizingPhase, CompleteState

class CompletePhase:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.node = coordinator

        self.state = None
        self.home_pose = None

        self.node.get_logger().info("CompletePhase initialized.")

    # ============================================================
    # RESET
    # ============================================================
    def reset(self):
        self.state = None
        self.home_pose = None

    # ============================================================
    # START COMPLETE
    # ============================================================
    def start(self):
        if self.coordinator.phase != DepalletizingPhase.COMPLETE:
            self.node.get_logger().warn(
                "CompletePhase.start() called while "
                "coordinator is not in COMPLETE phase."
            )
            return False

        self.state = CompleteState.CHECKING_FOR_MORE_OBJECTS
        self.node.get_logger().info("Complete state -> CHECKING FOR MORE OBJECTS")
        self.check_for_more_objects()

        return True
    
    # ============================================================
    # CHECK FOR MORE OBJECTS
    # ============================================================
    def check_for_more_objects(self):
        if self.state != CompleteState.CHECKING_FOR_MORE_OBJECTS:
            return

        # Perception failed
        if self.coordinator.next_perception_error is not None:

            self.coordinator.fail(
                "Cannot determine whether depalletizing is complete: "
                f"{self.coordinator.next_perception_error}"
            )
            return

        # No objects remain
        if self.coordinator.perception.is_no_box_detected():
            self.node.get_logger().info("No more objects detected.")

            self.plan_home()

            return

        # Another object exists
        if self.coordinator.perception.is_ready():
            self.node.get_logger().info(
                "More objects detected. Returning to PICK."
            )
            self.coordinator.start_next_pick_from_complete()

            return

        # Perception is still running
        if self.coordinator.perception.is_active():

            self.node.get_logger().info(
                "Waiting for perception to determine whether more objects remain."
            )

            return

    # ============================================================
    # GENERATE HOME POSE
    # ============================================================
    def generate_home_pose(self):

        home_pose = PoseStamped()

        home_pose.header.frame_id = self.coordinator.base_frame
        home_pose.header.stamp = (
            self.node.get_clock().now().to_msg()
        )

        home_pose.pose.position.x = self.coordinator.home_x
        home_pose.pose.position.y = self.coordinator.home_y
        home_pose.pose.position.z = self.coordinator.home_z

        home_pose.pose.orientation.x = self.coordinator.home_qx
        home_pose.pose.orientation.y = self.coordinator.home_qy
        home_pose.pose.orientation.z = self.coordinator.home_qz
        home_pose.pose.orientation.w = self.coordinator.home_qw

        self.home_pose = home_pose

    # ============================================================
    # PLAN HOME
    # ============================================================
    def plan_home(self):
        if self.state != CompleteState.CHECKING_FOR_MORE_OBJECTS:
            self.node.get_logger().warn(
                "plan_home() called from unexpected COMPLETE state."
            )
            return

        self.generate_home_pose()
        self.state = CompleteState.PLANNING_HOME
        self.node.get_logger().info("Complete state -> PLANNING_HOME")

        self.node.get_logger().info("Planning PTP motion to HOME.")
        self.coordinator.request_motion(
            pose=self.home_pose,
            motion_type="PTP",
            callback=self._home_plan_response,
        )

    def _home_plan_response(
        self,
        success,
        message,
    ):

        if self.state != CompleteState.PLANNING_HOME:
            return

        if not success:
            self.coordinator.fail(f"Failed to plan HOME motion: {message}")
            return

        self.state = CompleteState.EXECUTING_HOME
        self.node.get_logger().info("Complete state -> EXECUTING_HOME")

        self.node.get_logger().info(
            "HOME trajectory published. Waiting for physical execution completion."
        )

    # ============================================================
    # PHYSICAL EXECUTION SUCCESS
    # ============================================================
    def handle_execution_success(self):
        if self.coordinator.phase != DepalletizingPhase.COMPLETE:
            return

        if self.state != CompleteState.EXECUTING_HOME:

            self.node.get_logger().warn(
                "Trajectory execution succeeded while COMPLETE "
                f"is in unexpected state: {self.state}"
            )
            return

        self.state = CompleteState.COMPLETE_FINISHED
        self.node.get_logger().info("Complete state -> COMPLETE_FINISHED")

        self.node.get_logger().info("========================================")
        self.node.get_logger().info("DEPALLETIZING COMPLETED SUCCESSFULLY")
        self.node.get_logger().info("Robot returned to HOME position.")
        self.node.get_logger().info("========================================")