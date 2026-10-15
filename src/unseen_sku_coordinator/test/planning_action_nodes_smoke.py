#!/usr/bin/env python3
"""Planning-only interprocess test: Candidate Action -> Motion Action -> C++.

DO NOT RUN with live robot drivers. Only use the isolated planning-only launch:
ROS_DOMAIN_ID=97, synthetic /joint_states, no trajectory execution, and
/move_group allow_trajectory_execution=false.

Candidate rank 1 requests unreachable IK; rank 2 is a reachable local target.
No robot execution commands are sent.
"""

import os
import time

import rclpy
from action_msgs.msg import GoalStatus
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from unseen_sku_interfaces.action import PlanSuctionObject
from unseen_sku_interfaces.msg import SuctionCandidate, SuctionResult


TEST_DOMAIN = "97"


def rotate_vector(vector, q):
    """Rotate a direction by an xyzw quaternion (test-only helper)."""
    import math
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("Invalid TF quaternion")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    vx, vy, vz = vector
    tx, ty, tz = (2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx))
    return (vx+w*tx+y*tz-z*ty,
            vy+w*ty+z*tx-x*tz,
            vz+w*tz+x*ty-y*tx)


def wait(executor, future, seconds, label):
    deadline = time.monotonic() + seconds
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
    if not future.done():
        raise TimeoutError(f"{label} timed out after {seconds} seconds")
    return future.result()


def main():
    if os.environ.get("ROS_DOMAIN_ID") != TEST_DOMAIN:
        raise RuntimeError(
            "Refusing smoke test outside isolated ROS_DOMAIN_ID=97")
    rclpy.init()
    node = Node("planning_action_nodes_smoke")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node, spin_thread=False)
    try:
        params = node.create_client(GetParameters, "/move_group/get_parameters")
        if not params.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/move_group parameters unavailable")
        query = GetParameters.Request()
        query.names = ["allow_trajectory_execution"]
        reply = wait(executor, params.call_async(query), 8.0, "execution safety")
        values = reply.values
        if len(values) != 1 or values[0].type != ParameterType.PARAMETER_BOOL:
            raise RuntimeError("Cannot verify allow_trajectory_execution boolean")
        if values[0].bool_value:
            raise RuntimeError("Refusing to plan: MoveIt execution is enabled")
        print("PASS: MoveIt allow_trajectory_execution=False")

        tf = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and tf is None:
            executor.spin_once(timeout_sec=0.1)
            try:
                tf = tf_buffer.lookup_transform("base_link", "vacuum_tcp", Time())
            except TransformException:
                pass
        if tf is None:
            raise RuntimeError("Cannot determine synthetic current vacuum_tcp pose")

        action = ActionClient(node, PlanSuctionObject, "/unseen_sku/plan_object")
        if not action.wait_for_server(timeout_sec=12.0):
            raise RuntimeError("/unseen_sku/plan_object action server unavailable")
        scene = SuctionResult()
        scene.scene_id = "planning-actions-ik-fallback-smoke"
        scene.candidates.scene_index = 123
        scene.candidates.status = "OK"
        scene.candidates.header.frame_id = "base_link"
        scene.candidates.header.stamp = node.get_clock().now().to_msg()

        unreachable = SuctionCandidate()
        unreachable.object_id = 2
        unreachable.rank = 1
        unreachable.ranking_score = 0.9
        unreachable.position.x = 5.0
        unreachable.position.z = 1.0
        unreachable.normal.z = 1.0

        reachable = SuctionCandidate()
        reachable.object_id = 2
        reachable.rank = 2
        reachable.ranking_score = 0.8
        p = tf.transform.translation
        tool_z = rotate_vector((0.0, 0.0, 1.0), tf.transform.rotation)
        reachable.position.x = p.x + 0.01 * tool_z[0]
        reachable.position.y = p.y + 0.01 * tool_z[1]
        reachable.position.z = p.z + 0.01 * tool_z[2]
        reachable.normal.x = -tool_z[0]
        reachable.normal.y = -tool_z[1]
        reachable.normal.z = -tool_z[2]
        scene.candidates.candidates = [reachable, unreachable]  # unsorted on purpose

        goal = PlanSuctionObject.Goal()
        goal.expected_scene_id = scene.scene_id
        goal.target_object_id = 2
        goal.suction_result = scene
        print("Requesting ranked planning action: unreachable rank 1 -> rank 2")

        def report_feedback(feedback_message):
            feedback = feedback_message.feedback
            print(f"  feedback: rank={feedback.current_rank} "
                  f"phase={feedback.phase}", flush=True)

        handle = wait(executor, action.send_goal_async(
            goal, feedback_callback=report_feedback), 12.0, "action goal acceptance")
        if not handle.accepted:
            raise RuntimeError("Candidate planning action rejected the goal")
        result_future = handle.get_result_async()
        try:
            terminal = wait(executor, result_future, 115.0, "planning action result")
        except TimeoutError:
            handle.cancel_goal_async()
            raise
        result = terminal.result
        print(f"Terminal ROS action status: {terminal.status}")
        print(f"Candidate status: {result.status}")
        print(f"Selected rank: {result.selected_rank}")
        for attempt in result.attempts:
            print(f"  rank {attempt.rank}: {attempt.motion_status} "
                  f"(code={attempt.failure_code}, "
                  f"stage={attempt.failed_stage}, moveit={attempt.moveit_error_code})")
        print(f"Trajectories: "
              f"{len(result.plan.pre_pick_trajectory.points)} PTP, "
              f"{len(result.plan.pick_trajectory.points)} pick LIN, "
              f"{len(result.plan.retract_trajectory.points)} retract LIN")
        print("No execution commands were sent.")
        if (terminal.status != GoalStatus.STATUS_SUCCEEDED
                or not result.success or result.status != "pick_ready"
                or result.selected_rank != 2
                or [a.rank for a in result.attempts] != [1, 2]
                or result.attempts[0].motion_status != "candidate_rejected"
                or result.attempts[0].failure_code != 3
                or result.attempts[0].moveit_error_code != -31
                or result.attempts[1].motion_status != "planned"
                or not result.plan.pre_pick_trajectory.points
                or not result.plan.pick_trajectory.points
                or not result.plan.retract_trajectory.points):
            raise RuntimeError("Unexpected interprocess IK-fallback result")
        print("PASS: two ROS planning action nodes + C++ MoveIt2 rank fallback")
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
