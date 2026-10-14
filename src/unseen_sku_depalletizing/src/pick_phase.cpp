#include "moma_depalletizing/depalletizing_coordinator.hpp"

#include <chrono>
#include <cmath>
#include <functional>

#include <moveit_msgs/msg/planning_scene_components.hpp>

namespace moma_depalletizing
{

void DepalletizingCoordinator::pick_callback(
  const geometry_msgs::msg::PointStamped::SharedPtr message
) 
{
  if (phase_ != DepalletizingPhase::PICK) {
    return;
  }

  if (pick_state_ != PickState::WAITING_FOR_PICK_POINT) {
    RCLCPP_WARN(
      get_logger(),
      "Pick phase is already active."
    );

    return;
  }

  if (message->header.frame_id != base_frame_) {
    RCLCPP_ERROR(
      get_logger(),
      "Pick point is in frame '%s', expected '%s'.",
      message->header.frame_id.c_str(),
      base_frame_.c_str()
    );

    return;
  }

  // Store target
  {
    std::lock_guard<std::mutex> lock(
      target_mutex_
    );

    pending_pick_point_ = message;
  }

  octomap_retry_count_ = 0;

  RCLCPP_INFO(
    get_logger(),
    "Received pick point: [%.4f, %.4f, %.4f]",
    message->point.x,
    message->point.y,
    message->point.z
  );

  pick_state_ = PickState::WAITING_FOR_OBSTACLE_CLOUD;

  RCLCPP_INFO(
    get_logger(),
    "Waiting for obstacle cloud..."
  );
}

// ============================================================
// Obstacle cloud callback
//
// The state now should be WAITING_FOR_OBSTACLE_CLOUD
// ============================================================

void DepalletizingCoordinator::obstacle_cloud_callback(
  const sensor_msgs::msg::PointCloud2::SharedPtr message
)
{
  if (phase_ != DepalletizingPhase::PICK) {
    return;
  }
  
  if (pick_state_ != PickState::WAITING_FOR_OBSTACLE_CLOUD){
    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "Obstacle cloud received: "
    "%u x %u points, frame '%s'.",
    message->width,
    message->height,
    message->header.frame_id.c_str()
  );

  // ----------------------------------------------------------
  // If OctoMap is disabled, immediately continue.
  // ----------------------------------------------------------

  if (!require_octomap_) {
    RCLCPP_WARN(
      get_logger(),
      "OctoMap requirement disabled."
    );

    if (execution_mode_ == ExecutionMode::AUTO) {
      plan_pre_pick();
    }

    else {
      pick_state_ = PickState::WAITING_FOR_PICK_APPROVAL;

      RCLCPP_WARN(
        get_logger(),
        "PICK phase ready. Waiting for manual approval."
      );
    }
    
    return;
  }

  pick_state_ = PickState::WAITING_FOR_OCTOMAP;

  octomap_retry_count_=0;
  
  request_octomap_check();
}

// ============================================================
// Request current MoveIt OctoMap
// ============================================================
void DepalletizingCoordinator::request_octomap_check()
{
  if (phase_ != DepalletizingPhase::PICK) {
    return;
  }

  if (pick_state_ != PickState::WAITING_FOR_OCTOMAP) {
    return;
  }

  // Prevent multiple simultaneous requests.
  bool expected = false;

  if (!planning_scene_request_in_flight_.compare_exchange_strong(expected, true))
  {
    return;
  }

  // Ensure MoveIt's service to get planning scene exists
  if (!planning_scene_client_->service_is_ready()) {
    planning_scene_request_in_flight_ = false;

    RCLCPP_WARN(
      get_logger(),
      "/get_planning_scene is not ready yet."
    );

    schedule_octomap_retry();

    return;
  }

  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  
  // Request only OctoMap component
  request->components.components = moveit_msgs::msg::PlanningSceneComponents::OCTOMAP;

  planning_scene_client_->async_send_request(
    request,

    [this](
      rclcpp::Client<
        moveit_msgs::srv::GetPlanningScene
      >::SharedFuture future)
    {
      planning_scene_request_in_flight_ = false;

      if (phase_ != DepalletizingPhase::PICK) {
        return;
      }

      if (pick_state_ != PickState::WAITING_FOR_OCTOMAP)
      {
        return;
      }

      const auto response = future.get();

      const auto & octomap = response->scene.world.octomap.octomap;

      // OctoMap still empty.
      if (octomap.data.empty()) {
        RCLCPP_WARN(
          get_logger(),
          "MoveIt OctoMap is still empty."
        );

        schedule_octomap_retry();

        return;
      }

      // OctoMap is now ready.
      RCLCPP_INFO(
        get_logger(),
        "MoveIt OctoMap is ready. "
        "OctoMap contains %zu bytes.",
        octomap.data.size()
      );

      octomap_retry_count_ = 0;

      // We now have:
      //
      // 1. Pick point
      // 2. Obstacle cloud
      // 3. Populated MoveIt OctoMap
      //
      // Therefore planning may begin.
      if (execution_mode_ == ExecutionMode::AUTO) {
        plan_pre_pick();
      }

      else {
        pick_state_ = PickState::WAITING_FOR_PICK_APPROVAL;

        RCLCPP_WARN(
          get_logger(),
          "PICK phase ready. Waiting for manual approval."
        );
      }
    }
  );
}

// ============================================================
// Schedule a non-blocking OctoMap retry
// ============================================================
void DepalletizingCoordinator::schedule_octomap_retry()
{
  const int retry = octomap_retry_count_.fetch_add(1) + 1;

  if (retry > max_octomap_retries_) {
    fail(
      "MoveIt OctoMap did not become ready "
      "within the allowed retry period."
    );

    return;
  }

  RCLCPP_WARN(
    get_logger(),
    "Retrying OctoMap check in %d ms "
    "(attempt %d/%d).",
    octomap_retry_interval_ms_,
    retry,
    max_octomap_retries_
  );

  if (octomap_retry_timer_) {
    octomap_retry_timer_->cancel();
  }

  octomap_retry_timer_ =create_wall_timer(
    std::chrono::milliseconds(octomap_retry_interval_ms_),
      [this]()
      {
        if (octomap_retry_timer_) {
          octomap_retry_timer_->cancel();
        }

        request_octomap_check();
      },

      callback_group_
    );
}

// ============================================================
// GENERATE PICK POSES
//
// CURRENT ASSUMPTION:
//   flat boxes only.
//
// Therefore:
//
// pick_pose:
//   XYZ = PointStamped from perception
//   orientation = fixed top-down TCP orientation
//
// pre_pick_pose:
//   same pose
//   Z += pre_pick_offset_
// ============================================================
bool DepalletizingCoordinator::generate_pick_poses(
  const geometry_msgs::msg::PointStamped & pick_point
)
{
  // Normalize configured TCP quaternion
  const double quaternion_norm =
    std::sqrt(
      tcp_orientation_.x * tcp_orientation_.x +
      tcp_orientation_.y * tcp_orientation_.y +
      tcp_orientation_.z * tcp_orientation_.z +
      tcp_orientation_.w * tcp_orientation_.w
    );


  if (quaternion_norm < 1e-6) {
    RCLCPP_ERROR(
      get_logger(),
      "Configured TCP quaternion is invalid."
    );

    return false;
  }

  geometry_msgs::msg::Quaternion orientation;

  orientation.x = tcp_orientation_.x / quaternion_norm;
  orientation.y = tcp_orientation_.y / quaternion_norm;
  orientation.z = tcp_orientation_.z / quaternion_norm;
  orientation.w = tcp_orientation_.w / quaternion_norm;

  // Pick pose
  pick_pose_.header.frame_id = base_frame_;
  pick_pose_.header.stamp = now();
  
  pick_pose_.pose.position.x = pick_point.point.x;
  pick_pose_.pose.position.y = pick_point.point.y;
  pick_pose_.pose.position.z = pick_point.point.z;
  pick_pose_.pose.orientation = orientation;

  // Pre-pick pose
  pre_pick_pose_ = pick_pose_;

  pre_pick_pose_.pose.position.z += pre_pick_offset_;

  RCLCPP_INFO(
    get_logger(),
    "Pick pose: "
    "[%.4f, %.4f, %.4f]",
    pick_pose_.pose.position.x,
    pick_pose_.pose.position.y,
    pick_pose_.pose.position.z
  );

  RCLCPP_INFO(
    get_logger(),
    "Pre-pick pose: "
    "[%.4f, %.4f, %.4f]",
    pre_pick_pose_.pose.position.x,
    pre_pick_pose_.pose.position.y,
    pre_pick_pose_.pose.position.z
  );

  return true;
}

// ============================================================
// EXECUTION-RESULT LOGIC
// ============================================================
void DepalletizingCoordinator::handle_pick_execution_success()
{
  // Determine which PICK motion finished
  switch (pick_state_)
  {
    // ========================================================
    // PRE-PICK FINISHED
    //
    // Robot has reached pre_pick_pose_.
    // Next: LIN descent to pick_pose_.
    // ========================================================

    case PickState::EXECUTING_PRE_PICK:
    {
      RCLCPP_INFO(
        get_logger(),
        "Pre-pick execution completed successfully."
      );

      plan_pick();

      break;
    }
    
    // ========================================================
    // PICK FINISHED
    //
    // Robot is now touching the carton.
    // Next: activate vacuum.
    // ========================================================

    case PickState::EXECUTING_PICK:
    {
      RCLCPP_INFO(
        get_logger(),
        "Pick execution completed successfully."
      );

      pick_state_ =
        PickState::ACTIVATING_SUCTION;

      RCLCPP_INFO(
        get_logger(),
        "Pick state -> ACTIVATING_SUCTION"
      );

      activate_suction();

      break;
    }

    // ========================================================
    // RETRACT FINISHED
    //
    // Robot has returned from pick_pose_ to pre_pick_pose_
    // while holding the carton.
    // ========================================================

    case PickState::EXECUTING_RETRACT:
    {
      RCLCPP_INFO(
        get_logger(),
        "Retract execution completed successfully."
      );

      pick_state_ = PickState::PICK_FINISHED;

      RCLCPP_INFO(
        get_logger(),
        "Pick state -> PICK_FINISHED"
      );

      RCLCPP_INFO(
        get_logger(),
        "PICK phase completed successfully."
      );

      break;
    }

    // ========================================================
    // Unexpected result
    // ========================================================

    default:
    {
      RCLCPP_WARN(
        get_logger(),
        "Trajectory execution result received "
        "in unexpected PICK state."
      );

      break;
    }
  }
}

// ============================================================
// PLAN PRE-PICK
// ============================================================

void DepalletizingCoordinator::plan_pre_pick() 
{
  geometry_msgs::msg::PointStamped::SharedPtr pick_point;

  // Retrieve stored target
  {
    std::lock_guard<std::mutex> lock(target_mutex_);

    pick_point =
      pending_pick_point_;
  }

  if (!pick_point) {
    fail("No pending pick point available");

    return;
  }

  // Generate pick/pre-pick poses
  if (!generate_pick_poses(*pick_point)) {
    fail("Failed to generate pick poses.");

    return;
  }

  pick_state_ = PickState::PLANNING_PRE_PICK;

  RCLCPP_INFO(
    get_logger(),
    "Planning PTP motion to pre-pick."
  );

  // Request Pilz PTP
  if (!plan_and_publish_pose(
        pre_pick_pose_,
        moma_motion_planning::MotionType::PTP))
  {
    fail("Failed to plan pre-pick motion.");

    return;
  }

  // While the trajectory has been planned/published,
  // this does not mean the real robot has reached pre-pick point yet.
  pick_state_ = PickState::EXECUTING_PRE_PICK;

  RCLCPP_INFO(
    get_logger(),
    "Pre-pick trajectory published. "
    "Waiting for robot execution completion."
  );
}

// ============================================================
// PLAN PICK
//
// ONLY call this after the real robot reports that the
// pre-pick trajectory finished successfully.
// ============================================================
void DepalletizingCoordinator::plan_pick()
{
  if (phase_ != DepalletizingPhase::PICK) {
    return;
  }

  pick_state_ = PickState::PLANNING_PICK;

  RCLCPP_INFO(
    get_logger(),
    "Planning LIN motion from pre-pick to pick."
  );

  if (!plan_and_publish_pose(
        pick_pose_,
        moma_motion_planning::MotionType::LIN))
  {
    fail("Failed to plan LIN pick motion.");

    return;
  }

  pick_state_ = PickState::EXECUTING_PICK;

  RCLCPP_INFO(
    get_logger(),
    "Pick trajectory published. "
    "Waiting for robot execution completion."
  );
}

// ============================================================
// ACTIVATE SUCTION
// ============================================================
void DepalletizingCoordinator::activate_suction()
{
  // Safety: suction should only be activated during PICK.
  if (phase_ != DepalletizingPhase::PICK) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Cannot activate suction: coordinator is not in PICK phase."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  if (pick_state_ != PickState::ACTIVATING_SUCTION)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Cannot activate suction: PickState is not ACTIVATING_SUCTION."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  // Prevent unexpected second vacuum command
  if (vacuum_enabled_)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Cannot activate suction: vacuum is already marked ON."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  // Check that the Elite SetIO service is available
  if (!vacuum_io_client_->service_is_ready())
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Vacuum service is not available: "
      "/io_and_status_controller/set_io"
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  // Build Elite SetIO request
  //
  // fun = standard digital output
  // pin = DO0
  // state = ON
  auto request = std::make_shared<eli_common_interface::srv::SetIO::Request>();

  request->fun = eli_common_interface::srv::SetIO::Request::FUN_SET_DIGITAL_OUT;
  request->pin = vacuum_do_pin_;

  // Not used for a digital output
  request->analog_type = 0;

  request->state = 1.0;

  RCLCPP_WARN(
    this->get_logger(),
    "========================================"
  );

  RCLCPP_WARN(
    this->get_logger(),
    "PICK: VACUUM ON"
  );

  RCLCPP_WARN(
    this->get_logger(),
    "Elite standard DO%d -> ON",
    vacuum_do_pin_
  );

  RCLCPP_WARN(
    this->get_logger(),
    "========================================"
  );

  // ----------------------------------------------------------
  // Send asynchronously.
  //
  // Do NOT block the ROS executor while waiting for Elite.
  // ----------------------------------------------------------

  vacuum_io_client_->async_send_request(
    request,
    std::bind(
      &DepalletizingCoordinator::vacuum_on_response,
      this,
      std::placeholders::_1
    )
  );
}
// ============================================================
// VACUUM SERVICE RESPONSE
// ============================================================

void DepalletizingCoordinator::vacuum_on_response(
   rclcpp::Client<eli_common_interface::srv::SetIO>::SharedFuture future
)
{
  eli_common_interface::srv::SetIO::Response::SharedPtr response;

  try
  {
    response = future.get();
  }
  catch (const std::exception & error)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Vacuum SetIO service call failed: %s",
      error.what()
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  if (!response) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Vacuum SetIO returned a null response."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  if (!response->success)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Vacuum SetIO returned success=false."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  // Robot accepted DO0 command
  vacuum_enabled_ = true;

  RCLCPP_INFO(
    this->get_logger(),
    "Vacuum DO%d ON command accepted.",
    vacuum_do_pin_
  );

  // Digital output is now ON, but the pneumatic system
  // requires time  to build vacuum before lifting.
  if (vacuum_build_wait_sec_ <= 0.0)
  {
    vacuum_build_complete();
    return;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Waiting %.2f s for vacuum to build...",
    vacuum_build_wait_sec_
  );

  const auto delay = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(vacuum_build_wait_sec_));

  vacuum_build_timer_ =this->create_wall_timer(
      delay,
      std::bind(&DepalletizingCoordinator::vacuum_build_complete,this)
  );
}

// ============================================================
// VACUUM BUILD COMPLETE
// 
// Vacuum should be built in ACTIVATING_SUCTION pick state.
// ============================================================
void DepalletizingCoordinator::vacuum_build_complete()
{
  if (vacuum_build_timer_) {
    vacuum_build_timer_->cancel();
  }

  if (phase_ != DepalletizingPhase::PICK) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Vacuum build completed but coordinator is no longer in PICK phase."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  if (pick_state_ != PickState::ACTIVATING_SUCTION) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Vacuum build completed in unexpected PickState."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  if (!vacuum_enabled_)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Vacuum build timer completed but vacuum is not marked ON."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Vacuum build wait complete."
  );

  RCLCPP_INFO(
    this->get_logger(),
    "Proceeding to retract."
  );

  plan_retract();
}

// ============================================================
// PLAN RETRACT
// ============================================================
void DepalletizingCoordinator::plan_retract()
{
  if (
    phase_ != DepalletizingPhase::PICK ||
    pick_state_ != PickState::ACTIVATING_SUCTION)
  {
    RCLCPP_WARN(
      get_logger(),
      "plan_retract() called from unexpected state."
    );
    return;
  }

  pick_state_ = PickState::PLANNING_RETRACT;

  RCLCPP_INFO(
    get_logger(),
    "Planning LIN retract from pick to pre-pick."
  );

  const bool success = plan_and_publish_pose(
    pre_pick_pose_,
    moma_motion_planning::MotionType::LIN
  );

  if (!success) {
    RCLCPP_ERROR(
      get_logger(),
      "Failed to plan retract trajectory."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  pick_state_ = PickState::EXECUTING_RETRACT;

  RCLCPP_INFO(
    get_logger(),
    "Retract trajectory published."
  );

  RCLCPP_INFO(
    get_logger(),
    "State -> EXECUTING_RETRACT"
  );
}

} // namespace moma_depalletizing