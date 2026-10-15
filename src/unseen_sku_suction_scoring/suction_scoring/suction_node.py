#!/usr/bin/env python3
"""ROS 2 transport/lifecycle node for Sim-Suction. No scoring algorithms here.

All numerical processing lives in sim_suction_pipeline.py.
All ROS message conversions live in sim_suction_ros_adapter.py.
All debugging file output lives in sim_suction_debug.py.
"""
import time
from threading import Lock, Thread

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ament_index_python.packages import (
    get_package_share_directory
)
from pathlib import Path

from unseen_sku_interfaces.msg import (
    FusedObjectArray,
    SuctionCandidateArray,
    SuctionInput,
    SuctionResult,
)

from suction_scoring.suction_pipeline import SuctionPipeline, load_sim_config
from suction_scoring.suction_ros_adapter import (
    fused_array_to_scene,
    result_to_message,
    status_message,
)

from suction_scoring.suction_debug import write_suction_debug

class SuctionNode(Node):
    """
    Process each newly received fused scene and publish ranked candidates.
    """
    def __init__(self):
        super().__init__("suction_node")

        self.declare_parameter('input_topic', '/mask_fusion/objects')
        self.declare_parameter('output_topic', '/sim_suction/candidates')
        
        # Coordinator-controlled suction input.
        self.declare_parameter('input_mode', 'standalone')

        self.declare_parameter('suction_input_topic', '/coordinator/suction_input')

        # Coordinator-correlated scoring result.
        self.declare_parameter(
            'suction_result_topic',
            '/sim_suction/result',
        )

        # Locate the installed configuration through ROS 2.
        package_share = Path(
            get_package_share_directory("unseen_sku_suction_scoring")
        )

        default_config_path = (
            package_share
            / "config"
            / "sim_suction_config.yaml"
        )

        self.declare_parameter(
            'config_path',
            str(default_config_path),
        )
            
        # ------------------------------------------
        # M3.2B — Select input mode
        # ------------------------------------------

        self._input_mode = str(
            self.get_parameter('input_mode').value
        ).strip().lower()

        if self._input_mode not in ('standalone', 'controlled'):
            raise ValueError(
                f"Unsupported suction input_mode: {self._input_mode}"
            )
        
        self.bridge = CvBridge()

        # Build and load the model once, not once per scene
        config_path = str(self.get_parameter('config_path').value)
        config = load_sim_config(config_path)
        self.pipeline = SuctionPipeline(config)

        output_config = config.get('output', {})
        self.save_debug = bool(output_config.get('save_debug', False))

        # Debug outputs must use the configured directory.
        output_directory = output_config.get(
            'directory'
        )

        if self.save_debug and not output_directory:
            raise ValueError(
                "output.directory must be configured "
                "when output.save_debug is enabled"
            )

        self.output_root = (
            str(Path(output_directory).expanduser().resolve())
            if output_directory
            else None
        )

        self._lock = Lock()

        # Existing worker state.
        self._busy = False
        self._last_scene_index = -1
        self._worker = None

        # M3.3C: Identify the transaction currently being scored.
        self._active_scene_id = None
        self._active_scene_index = None

        # Publishers
        self.candidates_pub = self.create_publisher(
            SuctionCandidateArray,
            str(self.get_parameter('output_topic').value),
            10
        )

        # Only controlled mode publishes
        # transaction-correlated scoring results.
        self.suction_result_pub = None

        if self._input_mode == 'controlled':
            self.suction_result_pub = self.create_publisher(
                SuctionResult,
                str(self.get_parameter('suction_result_topic').value),
                10,
            )

        # Subscribers
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        # ------------------------------------------
        # M3.2B — Input mode subscriptions
        # ------------------------------------------

        self.fused_objects_sub = None
        self.suction_input_sub = None

        if self._input_mode == 'standalone':

            # Original Sim-Suction behavior.
            self.fused_objects_sub = self.create_subscription(
                FusedObjectArray,
                str(self.get_parameter('input_topic').value),
                self._fused_objects_callback,
                qos,
            )

        else:

            # Coordinator-controlled behavior.
            self.suction_input_sub = self.create_subscription(
                SuctionInput,
                str(self.get_parameter('suction_input_topic').value),
                self._suction_input_callback,
                qos,
            )

        self.get_logger().info(f"Suction ready. Input mode: {self._input_mode}")

    # ==========================================================
    # M3.3B — Coordinator-controlled input
    # ==========================================================

    def _suction_input_callback(self, message: SuctionInput):
        """
        Receive a coordinator-approved fused scene.

        Preserve the transaction ID throughout scoring.
        """

        scene_id = message.scene_id.strip()

        if not scene_id:
            self.get_logger().warning(
                "Rejecting SuctionInput with empty scene_id"
            )
            return

        fused_objects = message.fused_objects
        scene_index = int(fused_objects.scene_index)

        self.get_logger().info(
            f"Received coordinator-approved suction input: "
            f"scene_id={scene_id}; "
            f"scene_index={scene_index}; "
            f"objects={len(fused_objects.objects)}"
        )

        self._queue_scene(
            fused_objects,
            scene_id=scene_id,
        )

    # ==========================================================
    # Original standalone input
    # ==========================================================

    def _fused_objects_callback(self, message: FusedObjectArray):
        """
        Preserve the original standalone subscription behavior.
        """

        self._queue_scene(
            message,
            scene_id=None,
        )

    # ==========================================================
    # Shared worker scheduling
    # ==========================================================

    def _queue_scene(
        self,
        message: FusedObjectArray,
        scene_id=None,
    ):
        """
        Schedule Sim-Suction scoring.

        Controlled mode:
          - Process new transactions.
          - Ignore duplicates of the active transaction.
          - Reject different transactions while busy.
          - Reject duplicate or older scene indexes.

        Standalone mode:
          - Preserve existing busy and scene-index checks.

        No completed-result caching or automatic retries.
        """

        scene_index = int(message.scene_index)

        rejection_reason = None
        duplicate_reason = None

        # --------------------------------------------------
        # Determine whether this scene can be processed.
        # --------------------------------------------------

        with self._lock:

            if self._busy:

                # Same coordinator transaction is already active.
                if (
                    scene_id is not None
                    and scene_id == self._active_scene_id
                ):

                    if scene_index == self._active_scene_index:

                        duplicate_reason = (
                            f"Duplicate active transaction ignored: "
                            f"scene_id={scene_id}; "
                            f"scene_index={scene_index}"
                        )

                    else:

                        # Same ID but inconsistent scene index.
                        # Do not publish a contradictory response
                        # for the active transaction.
                        duplicate_reason = (
                            f"Inconsistent active transaction ignored: "
                            f"scene_id={scene_id}; "
                            f"active_index={self._active_scene_index}; "
                            f"received_index={scene_index}"
                        )

                else:

                    # Another transaction is already processing.
                    rejection_reason = (
                        f"Sim-Suction busy processing "
                        f"scene_id={self._active_scene_id}; "
                        f"cannot process "
                        f"scene_id={scene_id}, "
                        f"scene_index={scene_index}"
                    )

            elif scene_index <= self._last_scene_index:

                # Preserve existing stale-scene protection.
                rejection_reason = (
                    f"Duplicate or older scene_index={scene_index}; "
                    f"last_processed={self._last_scene_index}"
                )

            else:

                # Accept this scene for processing.
                self._busy = True

                self._active_scene_id = scene_id
                self._active_scene_index = scene_index

        # --------------------------------------------------
        # Duplicate active transaction: ignore it.
        # --------------------------------------------------

        if duplicate_reason is not None:

            self.get_logger().warning(duplicate_reason)

            return

        # --------------------------------------------------
        # Rejected transaction: report an error.
        # --------------------------------------------------

        if rejection_reason is not None:

            self.get_logger().warning(rejection_reason)

            if scene_id is not None:

                output = status_message(
                    message,
                    "ERROR",
                    rejection_reason,
                )

                self._publish_controlled_result(
                    scene_id,
                    output,
                )

            return

        # --------------------------------------------------
        # Accepted transaction: start the scoring worker.
        # --------------------------------------------------

        try:

            self._worker = Thread(
                target=self._process_scene,
                args=(message, scene_id),
                daemon=True,
            )

            self._worker.start()

        except Exception as error:

            # Worker startup failed.
            # Release the active transaction.

            with self._lock:

                self._busy = False

                self._active_scene_id = None
                self._active_scene_index = None

            error_detail = (
                f"Could not start scoring worker "
                f"for scene {scene_index}: {error}"
            )

            self.get_logger().error(error_detail)

            if scene_id is not None:

                output = status_message(
                    message,
                    "ERROR",
                    error_detail,
                )

                self._publish_controlled_result(
                    scene_id,
                    output,
                )

    # ==========================================================
    # Transaction-correlated result publishing
    # ==========================================================

    def _publish_controlled_result(
        self,
        scene_id: str,
        output: SuctionCandidateArray,
    ):
        """
        Publish the scoring outcome with its coordinator scene ID.
        """

        if self.suction_result_pub is None:

            self.get_logger().error(
                "SuctionResult publisher is unavailable"
            )
            return

        response = SuctionResult()

        response.scene_id = scene_id
        response.candidates = output

        try:

            self.suction_result_pub.publish(response)

        except Exception as error:

            self.get_logger().error(
                f"Could not publish SuctionResult "
                f"for scene_id={scene_id}: {error}"
            )

    # ==========================================================
    # Original scoring worker, extended with scene_id
    # ==========================================================

    def _process_scene(
        self,
        message: FusedObjectArray,
        scene_id=None,
    ):
        """
        Execute the existing scoring pipeline.

        scene_id is None in standalone mode.
        """

        scene_index = int(message.scene_index)

        start_time = time.perf_counter()

        result = None
        output = None
        scene = None

        try:

            # ROS message -> NumPy/Python scene.
            scene = fused_array_to_scene(
                message,
                self.bridge,
            )

            if scene is None:

                output = status_message(
                    message,
                    "NO_CANDIDATES",
                )

            else:

                # Existing Sim-Suction numerical pipeline.
                # Do not change its scoring algorithms.
                result = self.pipeline.process(scene)

                # Existing ROS adapter.
                output = result_to_message(
                    message,
                    scene,
                    result,
                )

                self.get_logger().info(
                    f"Scene {scene_index}: "
                    f"{result.valid_points} valid samples, "
                    f"{len(output.candidates)} candidates, "
                    f"inference={result.inference_seconds:.3f} s"
                )

        except Exception as error:

            self.get_logger().error(
                f"Suction scoring failed "
                f"for scene {scene_index}: {error}"
            )

            output = status_message(
                message,
                "ERROR",
                str(error),
            )

        finally:

            try:

                if output is None:

                    output = status_message(
                        message,
                        "ERROR",
                        "Scoring produced no output",
                    )

                output.processing_time_s = float(
                    time.perf_counter() - start_time
                )

                # Preserve the existing public output topic.
                try:

                    self.candidates_pub.publish(output)

                except Exception as error:

                    self.get_logger().error(
                        f"Could not publish suction candidates "
                        f"for scene {scene_index}: {error}"
                    )

                # Additional transaction-correlated output.
                # Attempt this independently of the legacy publisher.
                if scene_id is not None:

                    self._publish_controlled_result(
                        scene_id,
                        output,
                    )

            finally:

                # Always release worker state.
                with self._lock:

                    self._last_scene_index = scene_index

                    # M3.3C: Clear active transaction.
                    self._active_scene_id = None
                    self._active_scene_index = None

                    self._busy = False

        # ======================================================
        # Existing optional debug output
        # ======================================================

        if (
            result is not None
            and output is not None
            and output.status != "ERROR"
            and self.save_debug
        ):

            try:

                write_suction_debug(
                    output_root=self.output_root,
                    scene_index=scene_index,
                    output=output,
                    result=result,
                    scene=scene,
                )

                self.get_logger().info(
                    f"Scene {scene_index}: debug outputs saved to "
                    f"{self.output_root}/"
                    f"scene_{scene_index:04d}/sim_suction/"
                )

            except Exception as error:

                self.get_logger().warning(
                    f"Could not save debug output "
                    f"for scene {scene_index}: {error}"
                )

        if output is not None:

            self.get_logger().info(
                f"Scene {scene_index}: "
                f"status={output.status}, "
                f"elapsed={output.processing_time_s:.3f} s"
            )

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SuctionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
