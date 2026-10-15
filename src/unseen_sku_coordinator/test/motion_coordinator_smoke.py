#!/usr/bin/env python3
"""Planning-only integration smoke test using the verified fake-joint-state MoveIt.

Run only against planning_only_smoke.launch.py in isolated ROS domain 97 with
/move_group allow_trajectory_execution == false. No cameras or real robot.
This submits one synthetic candidate near the current TCP via MotionCoordinator.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from std_msgs.msg import Header
from unseen_sku_interfaces.msg import SuctionCandidate

from unseen_sku_coordinator.motion.motion_coordinator import MotionCoordinator
from unseen_sku_coordinator.motion.candidate_geometry import _rotate


def main():
    rclpy.init()
    node = Node('motion_coordinator_smoke')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    buffer = Buffer()
    listener = TransformListener(buffer, node, spin_thread=False)
    motion = MotionCoordinator(node)
    try:
        print('Waiting for base_link <- vacuum_tcp ...', flush=True)
        tf = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            try:
                tf = buffer.lookup_transform('base_link', 'vacuum_tcp', Time())
                break
            except TransformException:
                pass
        if tf is None:
            raise RuntimeError('No robot TCP TF in planning-only domain')

        p = tf.transform.translation
        tool_z = _rotate((0.0, 0.0, 1.0), tf.transform.rotation)
        item = SuctionCandidate()
        item.object_id = 1
        item.rank = 1
        item.position.x = p.x + 0.01 * tool_z[0]
        item.position.y = p.y + 0.01 * tool_z[1]
        item.position.z = p.z + 0.01 * tool_z[2]
        item.normal.x = -tool_z[0]
        item.normal.y = -tool_z[1]
        item.normal.z = -tool_z[2]
        header = Header()
        header.frame_id = 'base_link'
        header.stamp = node.get_clock().now().to_msg()

        print('Submitting synthetic candidate (planning-only)...', flush=True)
        future = motion.plan_candidate_async(item, header, 'smoke-scene')
        deadline = time.monotonic() + 80.0
        while not future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if not future.done():
            raise RuntimeError('No MotionCoordinator result after 80 seconds')

        result = future.result()
        print(f'Status: {result.status.value}', flush=True)
        print(f'Detail: {result.detail}', flush=True)
        if result.success:
            response = result.planning_response
            for label, trajectory in (
                ('PRE_PICK_PTP', response.pre_pick_trajectory),
                ('PICK_LIN', response.pick_trajectory),
                ('RETRACT_LIN', response.retract_trajectory),
            ):
                print(f'  {label}: {len(trajectory.points)} waypoints')
        print('No robot execution commands were sent.', flush=True)
        if not result.success:
            raise RuntimeError('Planning-only MotionCoordinator smoke failed')
    finally:
        motion.close()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
