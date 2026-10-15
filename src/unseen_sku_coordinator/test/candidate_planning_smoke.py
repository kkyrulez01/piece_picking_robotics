#!/usr/bin/env python3
"""Planning-only integration smoke for ranked candidate fallback.

Run ONLY with unseen_sku_motion_planning/planning_only_smoke.launch.py on
ROS_DOMAIN_ID=97 and /move_group allow_trajectory_execution=false.

Candidate rank 1 intentionally has a zero normal to exercise fallback; such an
invalid candidate would ordinarily be rejected by the perception validator.
Rank 2 is a synthetic reachable point near the current TCP. This is NOT a
real-perception, collision-scene, physical execution, or grasp validation test.
"""

import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from unseen_sku_interfaces.msg import SuctionCandidate, SuctionResult
from unseen_sku_coordinator.candidate.candidate_planning_coordinator import (
    CandidatePlanningCoordinator,
    CandidatePlanningStatus,
)
from unseen_sku_coordinator.motion.motion_coordinator import (
    MotionCoordinator,
    _rotate,
)


SCENE_ID = "candidate-planning-smoke"
TARGET_OBJECT = 2


def main():
    rclpy.init()
    node = Node("candidate_planning_smoke")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    tf_buffer = Buffer()
    listener = TransformListener(tf_buffer, node, spin_thread=False)
    motion = MotionCoordinator(node, tf_buffer=tf_buffer)
    candidate_planner = CandidatePlanningCoordinator(motion)
    try:
        print("Waiting for base_link <- vacuum_tcp ...", flush=True)
        tf = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            try:
                tf = tf_buffer.lookup_transform("base_link", "vacuum_tcp", Time())
                break
            except TransformException:
                pass
        if tf is None:
            raise RuntimeError("No TCP transform in isolated planning-only domain")

        p = tf.transform.translation
        tool_z = _rotate((0.0, 0.0, 1.0), tf.transform.rotation)
        scene = SuctionResult()
        scene.scene_id = SCENE_ID
        scene.candidates.scene_index = 1
        scene.candidates.status = "OK"
        scene.candidates.header.frame_id = "base_link"
        scene.candidates.header.stamp = node.get_clock().now().to_msg()

        rank1 = SuctionCandidate()
        rank1.object_id = TARGET_OBJECT
        rank1.rank = 1
        rank1.ranking_score = 0.9
        # Deliberately zero normal: MotionCoordinator returns INVALID_CANDIDATE.

        rank2 = SuctionCandidate()
        rank2.object_id = TARGET_OBJECT
        rank2.rank = 2
        rank2.ranking_score = 0.8
        rank2.position.x = p.x + 0.01 * tool_z[0]
        rank2.position.y = p.y + 0.01 * tool_z[1]
        rank2.position.z = p.z + 0.01 * tool_z[2]
        rank2.normal.x = -tool_z[0]
        rank2.normal.y = -tool_z[1]
        rank2.normal.z = -tool_z[2]

        other_object = SuctionCandidate()
        other_object.object_id = 999
        other_object.rank = 1
        other_object.ranking_score = 1.0

        # Intentionally unsorted; verify object filtering and rank ordering.
        scene.candidates.candidates = [other_object, rank2, rank1]

        print("Planning rank 1 (invalid) then rank 2 (reachable) ...", flush=True)
        future = candidate_planner.plan_object_async(
            scene, TARGET_OBJECT, expected_scene_id=SCENE_ID)
        deadline = time.monotonic() + 90.0
        while not future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if not future.done():
            candidate_planner.cancel_scene(SCENE_ID)
            raise RuntimeError("No candidate-planning result after 90 seconds")

        result = future.result()
        print(f"Status: {result.status.value}")
        print(f"Detail: {result.detail}")
        print(f"Selected rank: {result.selected_rank}")
        for attempt in result.attempts:
            print(f"  rank {attempt.rank}: {attempt.motion_status.value} "
                  f"(stage={attempt.failed_stage}, code={attempt.failure_code})")
        if result.success:
            response = result.planning_response
            for label, trajectory in (
                ("PRE_PICK_PTP", response.pre_pick_trajectory),
                ("PICK_LIN", response.pick_trajectory),
                ("RETRACT_LIN", response.retract_trajectory),
            ):
                print(f"  {label}: {len(trajectory.points)} waypoints")
        print("No robot execution commands were sent.", flush=True)
        if (result.status != CandidatePlanningStatus.PICK_READY
                or result.selected_rank != 2
                or [attempt.rank for attempt in result.attempts] != [1, 2]):
            raise RuntimeError("Unexpected ranked planning result")
    finally:
        candidate_planner.close()
        motion.close()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
