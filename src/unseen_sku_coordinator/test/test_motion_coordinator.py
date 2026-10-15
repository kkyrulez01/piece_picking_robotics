"""Planning-only MotionCoordinator unit tests; no ROS graph or robot needed.

Run after sourcing ROS 2 Humble + the workspace:
python3 -m unittest discover -s src/unseen_sku_coordinator/test \
    -p test_motion_coordinator.py -v
"""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from builtin_interfaces.msg import Duration as DurationMsg, Time as TimeMsg
from geometry_msgs.msg import PoseStamped, Transform
from rclpy.task import Future
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from unseen_sku_interfaces.msg import SuctionCandidate
from unseen_sku_motion_planning.srv import PlanCandidate

from unseen_sku_coordinator.motion.motion_coordinator import (
    JOINT_NAMES,
    MotionCoordinator,
    MotionResult,
    MotionStatus,
    _classify_planner_rejection,
    transform_candidate,
    validate_plan_response,
    TransformException,
)


def candidate():
    item = SuctionCandidate()
    item.object_id = 4
    item.rank = 1
    item.position.x = 1.0
    item.position.y = 0.0
    item.position.z = 0.0
    item.normal.x = 1.0
    item.normal.y = 0.0
    item.normal.z = 0.0
    return item


def header():
    h = Header()
    h.frame_id = 'helios2_optical_frame'
    h.stamp = TimeMsg(sec=1791976782, nanosec=982296653)
    return h


def transform():
    tf = Transform()
    tf.translation.x = 1.0
    tf.translation.y = 2.0
    tf.translation.z = 3.0
    tf.rotation.z = math.sqrt(0.5)
    tf.rotation.w = math.sqrt(0.5)
    return tf


def valid_response(point, normal):
    response = PlanCandidate.Response()
    response.success = True
    response.message = 'Complete candidate trajectory planned successfully.'
    response.failure_code = PlanCandidate.Response.FAILURE_NONE
    outward = (normal.vector.x, normal.vector.y, normal.vector.z)
    coords = (point.point.x, point.point.y, point.point.z)
    poses = []
    for distance in (0.06, 0.0, 0.06):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = coords[0] + distance * outward[0]
        pose.pose.position.y = coords[1] + distance * outward[1]
        pose.pose.position.z = coords[2] + distance * outward[2]
        # Quaternion taking local TCP +Z to -outward normal.
        direction = tuple(-component for component in outward)
        axis = (-direction[1], direction[0], 0.0)  # cross((0,0,1), direction)
        w = 1.0 + direction[2]
        if w < 1e-9:
            axis, w = (1.0, 0.0, 0.0), 0.0
        qnorm = math.sqrt(sum(value * value for value in axis) + w*w)
        pose.pose.orientation.x = axis[0] / qnorm
        pose.pose.orientation.y = axis[1] / qnorm
        pose.pose.orientation.z = axis[2] / qnorm
        pose.pose.orientation.w = w / qnorm
        poses.append(pose)
    response.pre_pick_pose, response.pick_pose, response.retract_pose = poses

    trajectories = []
    for starts, ends in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)):
        traj = JointTrajectory()
        traj.joint_names = list(JOINT_NAMES)
        for joint_position, seconds in ((starts, 0), (ends, 1)):
            waypoint = JointTrajectoryPoint()
            waypoint.time_from_start = DurationMsg(sec=seconds)
            waypoint.positions = [joint_position] * 6
            waypoint.velocities = [0.0] * 6
            traj.points.append(waypoint)
        trajectories.append(traj)
    (response.pre_pick_trajectory, response.pick_trajectory,
     response.retract_trajectory) = trajectories
    return response


class TestMotionGeometry(unittest.TestCase):
    def test_point_rotated_and_translated_normal_only_rotated(self):
        point, normal = transform_candidate(candidate(), header(), 'base_link', transform())
        self.assertAlmostEqual(point.point.x, 1.0)
        self.assertAlmostEqual(point.point.y, 3.0)
        self.assertAlmostEqual(point.point.z, 3.0)
        self.assertAlmostEqual(normal.vector.x, 0.0, places=6)
        self.assertAlmostEqual(normal.vector.y, 1.0, places=6)
        self.assertAlmostEqual(normal.vector.z, 0.0, places=6)
        self.assertEqual(point.header.stamp, header().stamp)

    def test_invalid_normal(self):
        c = candidate()
        c.normal.x = 0.0
        with self.assertRaises(ValueError):
            transform_candidate(c, header(), 'base_link', transform())

    def test_bad_transform_rotation(self):
        bad_tf = transform()
        bad_tf.rotation.z = 0.0
        bad_tf.rotation.w = 0.0
        with self.assertRaises(ValueError):
            transform_candidate(candidate(), header(), 'base_link', bad_tf)

    def test_valid_trajectories(self):
        point, normal = transform_candidate(candidate(), header(), 'base_link', None)
        validate_plan_response(valid_response(point, normal), point, normal)

    def test_joint_discontinuity_is_rejected(self):
        point, normal = transform_candidate(candidate(), header(), 'base_link', None)
        response = valid_response(point, normal)
        response.pick_trajectory.points[0].positions[0] += 0.01
        with self.assertRaisesRegex(ValueError, 'joint-continuous'):
            validate_plan_response(response, point, normal)

    def test_bad_joint_order_rejected(self):
        point, normal = transform_candidate(candidate(), header(), 'base_link', None)
        response = valid_response(point, normal)
        response.pick_trajectory.joint_names = list(reversed(JOINT_NAMES))
        with self.assertRaisesRegex(ValueError, 'joint order'):
            validate_plan_response(response, point, normal)

    def test_nonincreasing_timestamp_is_rejected(self):
        point, normal = transform_candidate(candidate(), header(), 'base_link')
        response = valid_response(point, normal)
        response.pick_trajectory.points[-1].time_from_start = DurationMsg(sec=0)
        with self.assertRaisesRegex(ValueError, 'timestamps not strictly increasing'):
            validate_plan_response(response, point, normal)

    def test_nonzero_terminal_velocity_is_rejected(self):
        point, normal = transform_candidate(candidate(), header(), 'base_link')
        response = valid_response(point, normal)
        response.retract_trajectory.points[-1].velocities[0] = 0.02
        with self.assertRaisesRegex(ValueError, 'nonzero terminal velocity'):
            validate_plan_response(response, point, normal)

    def test_wrong_pick_orientation_is_rejected(self):
        point, normal = transform_candidate(candidate(), header(), 'base_link')
        response = valid_response(point, normal)
        response.pick_pose.pose.orientation.x = 0.0
        response.pick_pose.pose.orientation.y = 0.0
        response.pick_pose.pose.orientation.z = 0.0
        response.pick_pose.pose.orientation.w = 1.0
        with self.assertRaisesRegex(ValueError, 'not opposite the outward normal'):
            validate_plan_response(response, point, normal)

    def test_structured_failure_classification(self):
        # Text is deliberately misleading: codes, not strings, determine status.
        response = PlanCandidate.Response()
        response.message = 'Could not obtain current robot state.'
        response.failure_code = PlanCandidate.Response.FAILURE_CANDIDATE_INFEASIBLE
        self.assertEqual(_classify_planner_rejection(response), MotionStatus.CANDIDATE_REJECTED)
        response.message = 'PICK_LIN: MoveIt planning failed.'
        response.failure_code = PlanCandidate.Response.FAILURE_ROBOT_STATE_UNAVAILABLE
        self.assertEqual(_classify_planner_rejection(response), MotionStatus.ROBOT_STATE_UNAVAILABLE)

    def test_failure_codes_fail_closed(self):
        response = PlanCandidate.Response()
        for code in (PlanCandidate.Response.FAILURE_NONE, 255):
            response.failure_code = code
            self.assertEqual(_classify_planner_rejection(response), MotionStatus.PLANNER_ERROR)

    def test_planner_failure_status_mapping(self):
        cases = (
            (PlanCandidate.Response.FAILURE_INVALID_REQUEST, MotionStatus.INVALID_REQUEST),
            (PlanCandidate.Response.FAILURE_INVALID_CANDIDATE, MotionStatus.INVALID_CANDIDATE),
            (PlanCandidate.Response.FAILURE_CANDIDATE_INFEASIBLE, MotionStatus.CANDIDATE_REJECTED),
            (PlanCandidate.Response.FAILURE_ROBOT_STATE_UNAVAILABLE, MotionStatus.ROBOT_STATE_UNAVAILABLE),
            (PlanCandidate.Response.FAILURE_PLANNING_FAILED_UNKNOWN, MotionStatus.PLANNING_FAILED),
            (PlanCandidate.Response.FAILURE_PLANNER_ERROR, MotionStatus.PLANNER_ERROR),
            (PlanCandidate.Response.FAILURE_INVALID_TRAJECTORY, MotionStatus.INVALID_RESPONSE),
        )
        for code, status in cases:
            with self.subTest(code=code):
                response = PlanCandidate.Response()
                response.failure_code = code
                self.assertEqual(_classify_planner_rejection(response), status)
                result = MotionResult(status=status, detail='',
                                      scene_id='scene-001', object_id=4, rank=1)
                self.assertEqual(
                    result.retry_next_candidate,
                    code in (PlanCandidate.Response.FAILURE_INVALID_CANDIDATE,
                             PlanCandidate.Response.FAILURE_CANDIDATE_INFEASIBLE),
                )


class FakeClock:
    def __init__(self, stamp):
        self.now_value = stamp.sec * 1_000_000_000 + stamp.nanosec

    def now(self):
        return SimpleNamespace(nanoseconds=self.now_value)


class FakeClient:
    def __init__(self):
        self.ready = True
        self.future = Future()
        self.requests = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return self.future


class FakeNode:
    def __init__(self, stamp):
        self.client = FakeClient()
        self.clock = FakeClock(stamp)
        self.callback = None

    def create_client(self, *args):
        return self.client

    def create_timer(self, seconds, callback):
        self.callback = callback
        return object()

    def destroy_timer(self, timer):
        pass

    def get_clock(self):
        return self.clock


class FakeBuffer:
    def __init__(self, tf):
        self.tf = tf
        self.lookups = []

    def lookup_transform(self, target, source, timestamp):
        self.lookups.append((target, source, timestamp))
        return SimpleNamespace(transform=self.tf)


class TestMotionAdapter(unittest.TestCase):
    def setUp(self):
        self.node = FakeNode(header().stamp)
        self.buffer = FakeBuffer(transform())
        self.motion = MotionCoordinator(self.node, tf_buffer=self.buffer)

    def tearDown(self):
        self.motion.close()

    def test_plan_async_returns_validated_response(self):
        result_future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        self.assertFalse(result_future.done())
        self.assertEqual(len(self.node.client.requests), 1)
        self.assertEqual(self.buffer.lookups[0][0:2],
                         ('base_link', 'helios2_optical_frame'))
        request = self.node.client.requests[0]
        response = valid_response(request.suction_point, request.surface_normal)
        self.node.client.future.set_result(response)
        self.motion._tick()
        self.assertTrue(result_future.done())
        self.assertEqual(result_future.result().status, MotionStatus.PLANNED)
        self.assertEqual(result_future.result().scene_id, 'scene-001')
        self.assertIs(result_future.result().planning_response, response)

    def test_invalid_candidate_skips_service(self):
        bad = candidate()
        bad.normal.x = 0.0
        result = self.motion.plan_candidate_async(bad, header(), 'scene-001').result()
        self.assertEqual(result.status, MotionStatus.INVALID_CANDIDATE)
        self.assertFalse(self.node.client.requests)

    def test_stale_header_rejected(self):
        self.node.clock.now_value += 100 * 1_000_000_000
        result = self.motion.plan_candidate_async(candidate(), header(), 'scene-001').result()
        self.assertEqual(result.status, MotionStatus.STALE_CANDIDATE)
        self.assertFalse(self.node.client.requests)

    def test_cancel_ignores_late_result(self):
        result_future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        self.motion.cancel_scene('scene-001')
        self.assertEqual(result_future.result().status, MotionStatus.CANCELLED)
        busy = self.motion.plan_candidate_async(candidate(), header(), 'scene-002').result()
        self.assertEqual(busy.status, MotionStatus.BUSY)
        self.node.client.future.set_result(None)  # A late service reply.
        self.motion._tick()
        self.assertEqual(result_future.result().status, MotionStatus.CANCELLED)

    def test_missing_tf_rejected_without_service_request(self):
        self.motion.tf_wait_timeout_sec = 0.0
        self.buffer.lookup_transform = Mock(side_effect=TransformException('not connected'))
        result = self.motion.plan_candidate_async(candidate(), header(), 'scene-001').result()
        self.assertEqual(result.status, MotionStatus.TF_UNAVAILABLE)
        self.assertFalse(self.node.client.requests)

    def test_unavailable_service_times_out(self):
        self.node.client.ready = False
        self.motion.service_wait_timeout_sec = 0.0
        result = self.motion.plan_candidate_async(candidate(), header(), 'scene-001').result()
        self.assertEqual(result.status, MotionStatus.SERVICE_UNAVAILABLE)
        self.assertFalse(self.node.client.requests)

    def test_planning_timeout_blocks_new_requests_until_reply(self):
        self.motion.planning_timeout_sec = 0.0
        timed_out = self.motion.plan_candidate_async(candidate(), header(), 'scene-001').result()
        self.assertEqual(timed_out.status, MotionStatus.SERVICE_TIMEOUT)
        busy = self.motion.plan_candidate_async(candidate(), header(), 'scene-002').result()
        self.assertEqual(busy.status, MotionStatus.BUSY)
        self.node.client.future.set_result(None)
        self.motion._tick()

    def test_invalid_response_is_rejected(self):
        result_future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        request = self.node.client.requests[0]
        response = valid_response(request.suction_point, request.surface_normal)
        response.pick_trajectory.points[0].positions[2] += 0.05
        self.node.client.future.set_result(response)
        self.motion._tick()
        self.assertEqual(result_future.result().status, MotionStatus.INVALID_RESPONSE)

    def test_invalid_scene_id_is_not_retryable(self):
        result = self.motion.plan_candidate_async(candidate(), header(), ' ').result()
        self.assertEqual(result.status, MotionStatus.INVALID_REQUEST)
        self.assertFalse(result.retry_next_candidate)
        self.assertFalse(self.node.client.requests)

    def test_invalid_candidate_identifiers_return_a_future(self):
        item = candidate()
        bad = SimpleNamespace(
            object_id=item.object_id, rank='invalid-rank',
            position=item.position, normal=item.normal,
        )
        result_future = self.motion.plan_candidate_async(bad, header(), 'scene-001')
        self.assertTrue(result_future.done())
        self.assertEqual(result_future.result().status, MotionStatus.INVALID_REQUEST)
        self.assertFalse(result_future.result().retry_next_candidate)

    def test_missing_frame_is_not_retryable(self):
        bad_header = header()
        bad_header.frame_id = ''
        result = self.motion.plan_candidate_async(candidate(), bad_header, 'scene-001').result()
        self.assertEqual(result.status, MotionStatus.INVALID_REQUEST)
        self.assertFalse(result.retry_next_candidate)

    def test_missing_capture_stamp_is_invalid_request(self):
        bad_header = header()
        bad_header.stamp = TimeMsg(sec=0)
        result = self.motion.plan_candidate_async(candidate(), bad_header, 'scene-001').result()
        self.assertEqual(result.status, MotionStatus.INVALID_REQUEST)
        self.assertFalse(result.retry_next_candidate)

    def test_candidate_becomes_stale_while_waiting_for_service(self):
        self.node.client.ready = False
        future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        self.assertFalse(future.done())
        self.node.clock.now_value += 31 * 1_000_000_000
        self.node.client.ready = True
        self.motion._tick()
        self.assertEqual(future.result().status, MotionStatus.STALE_CANDIDATE)
        self.assertFalse(self.node.client.requests)

    def test_candidate_becomes_stale_while_planning(self):
        future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        request = self.node.client.requests[0]
        self.node.clock.now_value += 31 * 1_000_000_000
        self.node.client.future.set_result(
            valid_response(request.suction_point, request.surface_normal)
        )
        self.motion._tick()
        result = future.result()
        self.assertEqual(result.status, MotionStatus.STALE_CANDIDATE)
        self.assertFalse(result.success)
        self.assertIsNone(result.planning_response)

    def test_candidate_infeasible_allows_next_rank(self):
        future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        response = PlanCandidate.Response()
        response.success = False
        response.message = 'PRE_PICK_PTP: MoveIt planning failed (error code -31).'
        response.failure_code = PlanCandidate.Response.FAILURE_CANDIDATE_INFEASIBLE
        response.failed_stage = 'PRE_PICK_PTP'
        response.moveit_error_code = -31
        self.node.client.future.set_result(response)
        self.motion._tick()
        result = future.result()
        self.assertEqual(result.status, MotionStatus.CANDIDATE_REJECTED)
        self.assertTrue(result.retry_next_candidate)
        self.assertEqual(result.failed_stage, 'PRE_PICK_PTP')
        self.assertEqual(result.moveit_error_code, -31)
        self.assertIsNone(result.planning_response)

    def test_invalid_request_from_cpp_does_not_retry(self):
        future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        response = PlanCandidate.Response()
        response.success = False
        response.message = 'Both frames must be in base_link'
        response.failure_code = PlanCandidate.Response.FAILURE_INVALID_REQUEST
        response.failed_stage = 'INPUT'
        self.node.client.future.set_result(response)
        self.motion._tick()
        result = future.result()
        self.assertEqual(result.status, MotionStatus.INVALID_REQUEST)
        self.assertFalse(result.retry_next_candidate)

    def test_unrecognized_cpp_failure_code_is_nonretryable(self):
        future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        response = PlanCandidate.Response()
        response.success = False
        response.message = 'Unrecognized backend failure'
        response.failure_code = 255
        self.node.client.future.set_result(response)
        self.motion._tick()
        result = future.result()
        self.assertEqual(result.status, MotionStatus.PLANNER_ERROR)
        self.assertFalse(result.retry_next_candidate)

    def test_success_with_failure_code_is_rejected(self):
        future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        request = self.node.client.requests[0]
        response = valid_response(request.suction_point, request.surface_normal)
        response.failure_code = PlanCandidate.Response.FAILURE_CANDIDATE_INFEASIBLE
        self.node.client.future.set_result(response)
        self.motion._tick()
        result = future.result()
        self.assertEqual(result.status, MotionStatus.INVALID_RESPONSE)
        self.assertFalse(result.retry_next_candidate)

    def test_moveit_failure_no_automatic_rank_fallback(self):
        result_future = self.motion.plan_candidate_async(candidate(), header(), 'scene-001')
        response = PlanCandidate.Response()
        response.success = False
        response.message = 'PICK_LIN: MoveIt planning failed.'
        response.failure_code = PlanCandidate.Response.FAILURE_PLANNING_FAILED_UNKNOWN
        response.failed_stage = 'PICK_LIN'
        response.moveit_error_code = -1
        self.node.client.future.set_result(response)
        self.motion._tick()
        result = result_future.result()
        self.assertEqual(result.status, MotionStatus.PLANNING_FAILED)
        self.assertFalse(result.retry_next_candidate)
        self.assertEqual(result.failure_code, PlanCandidate.Response.FAILURE_PLANNING_FAILED_UNKNOWN)
        self.assertEqual(result.failed_stage, 'PICK_LIN')
        self.assertEqual(result.moveit_error_code, -1)


if __name__ == '__main__':
    unittest.main()
