#!/usr/bin/env python3

"""
ROS 2 node for transforming suction candidates.

Input:
    /sim_suction/candidates
    unseen_sku_interfaces/msg/SuctionCandidateArray

Output:
    /sim_suction/candidates_base
    unseen_sku_interfaces/msg/SuctionCandidateArray

The input header.frame_id identifies the source frame.
"""

import copy
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener, TransformException
from unseen_sku_interfaces.msg import SuctionCandidateArray
from candidate_transform.coordinate_transform import transform_candidate

class CandidateTransformNode(Node):
    def __init__(self):
        super().__init__("candidate_transform_node")

        # Parameters
        self.declare_parameter("input_topic","/sim_suction/candidates")
        self.declare_parameter("output_topic", "/sim_suction/candidates_base")

        self.declare_parameter("source_frame", "helios2_optical_frame")
        self.declare_parameter("target_frame", "base_link")

        self.declare_parameter("max_tf_wait_sec", 3.0)
        self.declare_parameter("retry_period_sec", 0.1)
        self.declare_parameter("require_nonzero_stamp", False)

        # Read parameters
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.source_frame = self.get_parameter("source_frame").value
        self.target_frame = self.get_parameter("target_frame").value

        self.max_tf_wait_sec = float(self.get_parameter("max_tf_wait_sec").value)
        self.retry_period_sec = float(self.get_parameter("retry_period_sec").value)
        self.require_nonzero_stamp = bool(self.get_parameter("require_nonzero_stamp").value)

        if self.input_topic == self.output_topic:
            raise ValueError(
                "Input and output topics must be different."
            )

        if not self.target_frame:
            raise ValueError("target_frame cannot be empty")

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # Publishers
        self.candidate_publisher = self.create_publisher(
            SuctionCandidateArray,
            self.output_topic,
            10
        )

        # Subscribers
        self.candidate_subscriber = self.create_subscription(
            SuctionCandidateArray,
            self.input_topic,
            self.candidate_callback,
            10
        )

        # Pending scene
        self.pending_message = None
        self.pending_since = 0.0
        self.last_tf_error = ""

        self.warned_zero_stamp = False

        self.retry_timer = self.create_timer(
            self.retry_period_sec,
            self.try_transform_pending
        )

        self.get_logger().info("Candidate transform node started.")

        self.get_logger().info(f"Input: {self.input_topic}")
        self.get_logger().info(f"Output: {self.output_topic}")

        self.get_logger().info(f"Source frame: {self.source_frame}")
        self.get_logger().info(f"Target frame: {self.target_frame}")

    # After receiving suction candidates
    def candidate_callback(self, msg: SuctionCandidateArray):
        source_frame = msg.header.frame_id.strip()
        if not source_frame:
            self.get_logger().error("Candidate array rejected: header.frame_id is empty.")
            return

        # Verify the input is in the expected coordinate frame.
        if (self.source_frame and source_frame != self.source_frame):
            self.get_logger().error(
                "Candidate array rejected: "
                f"received frame '{source_frame}', "
                f"expected '{self.source_frame}'."
            )
            return

        stamp = msg.header.stamp
        stamp_is_zero = (stamp.sec == 0 and stamp.nanosec == 0)

        if stamp_is_zero and self.require_nonzero_stamp:
            self.get_logger().error(
                "Candidate array rejected: measurement timestamp is zero."
            )
            return

        if stamp_is_zero and not self.warned_zero_stamp:
            self.get_logger().warning(
                "Input timestamp is zero. Using the latest available TF."
            )
            self.warned_zero_stamp = True

        # A new scene supersedes an older pending scene.
        if self.pending_message is not None:
            self.get_logger().warning(
                "Replacing pending scene "
                f"{self.pending_message.scene_index} "
                f"with scene {msg.scene_index}."
            )

        self.pending_message = None

        # No candidates
        if len(msg.candidates) == 0:
            output = copy.deepcopy(msg)

            # No geometric data needs transforming.
            output.header.frame_id = self.target_frame

            self.candidate_publisher.publish(output)
            self.get_logger().info(
                f"Scene {msg.scene_index}: no candidates to transform."
            )

            return

        # Store newest scene for TF processing
        self.pending_message = msg
        self.pending_since = time.monotonic()
        self.last_tf_error = ""

        # Attempt immediately.
        self.try_transform_pending()

    # Transform pending scene
    def try_transform_pending(self):
        if self.pending_message is None:
            return

        msg = self.pending_message

        # Check timeout
        elapsed = time.monotonic() - self.pending_since
        if elapsed > self.max_tf_wait_sec:
            self.get_logger().error(
                f"Scene {msg.scene_index}: "
                f"TF unavailable after {elapsed:.2f}s. "
                f"Last error: {self.last_tf_error}"
            )

            self.pending_message = None
            return

        source_frame = msg.header.frame_id

        # Look up TF
        transform = None

        if source_frame != self.target_frame:
            stamp = msg.header.stamp
            stamp_is_zero = (stamp.sec == 0 and stamp.nanosec == 0)

            # For stamped scenes use capture time
            # For zero timestamps use latest available TF
            lookup_time = (Time() if stamp_is_zero else Time.from_msg(stamp))

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    lookup_time
                )

            except TransformException as error:
                self.last_tf_error = str(error)
                return

        # Transform complete candidate array
        try:
            output = copy.deepcopy(msg)
            if transform is not None:
                output.candidates = [
                    transform_candidate(candidate, transform) for candidate in msg.candidates
                ]

            output.header.frame_id = self.target_frame

        except (ValueError, TypeError, AttributeError, AssertionError) as error:
            self.get_logger().error(
                f"Scene {msg.scene_index}: "
                f"candidate transformation failed: {error}"
            )

            self.pending_message = None
            return

        # Publish transformed candidates
        self.candidate_publisher.publish(output)

        self.get_logger().info(
            f"Scene {msg.scene_index}: transformed "
            f"{len(output.candidates)} candidates "
            f"from {source_frame} to {self.target_frame}."
        )

        self.pending_message = None

def main(args=None):
    rclpy.init(args=args)
    node = CandidateTransformNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()