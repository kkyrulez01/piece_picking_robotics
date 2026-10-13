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

from unseen_sku_interfaces.msg import FusedObjectArray, SuctionCandidateArray
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
        self.declare_parameter(
            'config_path',
            '/home/support/unseen_sku_ws/src/suction_scoring/config/sim_suction_config.yaml',
        )
        self.bridge = CvBridge()

        # Build and load the model once, not once per scene
        config_path = str(self.get_parameter('config_path').value)
        config = load_sim_config(config_path)
        self.pipeline = SuctionPipeline(config)

        output_config = config.get('output', {})
        self.save_debug = bool(output_config.get('save_debug', False))
        self.output_root = str(output_config.get('directory', '/home/support/unseen_sku_ws/outputs'))

        self._lock = Lock()
        self._busy = False
        self._last_scene_index = -1
        self._worker = None

        # Publishers
        self.candidates_pub = self.create_publisher(
            SuctionCandidateArray,
            str(self.get_parameter('output_topic').value),
            10
        )

        # Subscribers
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        self.fused_objects_sub = self.create_subscription(
            FusedObjectArray,
            str(self.get_parameter('input_topic').value),
            self._fused_objects_callback,
            qos
        )

        self.get_logger().info('Suction ready; waiting for fused objects...')

    def _fused_objects_callback(self, message):
        scene_index = int(message.scene_index)

        with self._lock:
            if self._busy:
                self.get_logger().warning(
                    f'Skipping scene {scene_index}: processing still active'
                )
                return

            if scene_index <= self._last_scene_index:
                self.get_logger().warning(
                    f'Skipping duplicate or older scene {scene_index}'
                )
                return

            self._busy = True

        try:
            # Keep the ROS executor available while the GPU model is running.
            self._worker = Thread(
                target=self._process_scene,
                args=(message,),
                daemon=True,
            )
            self._worker.start()

        except Exception as error:
            with self._lock:
                self._busy = False
            self.get_logger().error(
                f'Could not start worker for scene {scene_index}: {error}'
            )

    def _process_scene(self, message):
        scene_index = int(message.scene_index)
        start_time = time.perf_counter()
        result = None
        output = None

        try:
            # ROS message -> NumPy/Python scene
            scene = fused_array_to_scene(message, self.bridge)

            if scene is None:
                output = status_message(message, 'NO_CANDIDATES')
            else:
                # Run suction pipeline on scene
                result = self.pipeline.process(scene)

                # Convert ranked results -> ROS message
                output = result_to_message(message, scene, result)

                self.get_logger().info(
                    f'Scene {scene_index}: {result.valid_points} valid samples, '
                    f'{len(output.candidates)} candidates, '
                    f'inference={result.inference_seconds:.3f} s'
                )

        except Exception as error:
            self.get_logger().error(
                f'Suction scoring failed for scene {scene_index}: {error}'
            )
            output = status_message(message, 'ERROR', str(error))

        finally:
            # Publish one completion message, including empty/error results
            try:
                if output is not None:
                    output.processing_time_s = float(time.perf_counter() - start_time)
                    self.candidates_pub.publish(output)

            except Exception as error:
                self.get_logger().error(
                    f'Could not publish scene {scene_index} result: {error}'
                )

            finally:
                with self._lock:
                    self._last_scene_index = scene_index
                    self._busy = False

        # For debugging
        if result is not None and output is not None and output.status != 'ERROR':
            if self.save_debug:
                try:
                    write_suction_debug(
                        output_root=str(self.get_parameter('output_root').value),
                        scene_index=scene_index,
                        output=output,
                        result=result,
                        scene=scene
                    )
                except Exception as error:
                    self.get_logger().warning(
                        f'Could not save debug output for scene {scene_index}: {error}'
                    )

        if output is not None:
            self.get_logger().info(
                f'Scene {scene_index}: status={output.status}, '
                f'elapsed={output.processing_time_s:.3f} s'
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
