#include "moma_depalletizing/depalletizing_coordinator.hpp"

#include <chrono>
#include <cmath>
#include <functional>

#include <moveit_msgs/msg/planning_scene_components.hpp>

namespace moma_depalletizing
{
DepalletizingCoordinator::DepalletizingCoordinator() : Node("depalletizing_coordinator")
{
}
  
// ============================================================
// Initialize
// ============================================================

bool DepalletizingCoordinator::initialize() {
  // Exeuction mode
  execution_mode_string_ = declare_parameter<std::string>("exeuction_mode", "manual");

  if (execution_mode_string_ == "manual") {
    execution_mode_ = ExecutionMode::MANUAL;

    RCLCPP_WARN(
      get_logger(),
      "Execution mode: MANUAL"
    );
  }

  else if (execution_mode_string_ == "auto") {
    execution_mode_ = ExecutionMode::AUTO;

    RCLCPP_WARN(
      get_logger(),
      "Execution mode: AUTO"
    );
  }

  else {
    RCLCPP_ERROR(
      get_logger(),
      "Invalid execution_mode '%s'. "
      "Valid options are 'manual' or 'auto'.",
      execution_mode_string_.c_str()
    );

    return false;
  }

  // Robot / MoveIt parameters
  group_name_ =declare_parameter<std::string>("group_name","ARC_S6");

  base_frame_ = declare_parameter<std::string>("base_frame","base_link");
  tcp_link_ = declare_parameter<std::string>("tcp_link","vacuum_tcp");

  planning_pipeline_ = declare_parameter<std::string>("planning_pipeline", "pilz");
  planning_time_ = declare_parameter<double>("planning_time", 5.0);
  planning_attempts_ = declare_parameter<int>("planning_attempts", 1);

  velocity_scaling_ = declare_parameter<double>("velocity_scaling",0.10);
  acceleration_scaling_ = declare_parameter<double>("acceleration_scaling", 0.10);

  // ==========================================================
  // Pick geometry
  // ==========================================================
  pre_pick_offset_ = declare_parameter<double>("pre_pick_offset", 0.05);

  tcp_orientation_.x = declare_parameter<double>("tcp_qx", 1.0);
  tcp_orientation_.y = declare_parameter<double>("tcp_qy", 0.0);
  tcp_orientation_.z = declare_parameter<double>("tcp_qz", 0.0);
  tcp_orientation_.w = declare_parameter<double>("tcp_qw", 0.0);

  // ==========================================================
  // Place geometry
  // ==========================================================

  place_x_ = declare_parameter<double>("place_x", 0.0);
  place_y_ = declare_parameter<double>("place_y", 0.0);
  place_z_ = declare_parameter<double>("place_z", 0.0);

  place_orientation_.x = declare_parameter<double>("place_qx", 1.0);
  place_orientation_.y = declare_parameter<double>("place_qy", 0.0);
  place_orientation_.z = declare_parameter<double>("place_qz", 0.0);
  place_orientation_.w = declare_parameter<double>("place_qw", 0.0);

  pre_place_offset_ = declare_parameter<double>("pre_place_offset", 0.05);

  // ==========================================================
  // HOME
  // ==========================================================
  home_target_name_ = declare_parameter<std::string>("home_target_name", "home");

  // ==========================================================
  // Vacuum gripper
  // ==========================================================
  vacuum_do_pin_ = this->declare_parameter<int>("vacuum_do_pin", 0);

  vacuum_build_wait_sec_ = this->declare_parameter<double>("vacuum_build_wait_sec", 1.0);

  vacuum_io_client_ = this->create_client<eli_common_interface::srv::SetIO>(
    "/io_and_status_controller/set_io"
  );

  // ==========================================================
  // Topics
  // ==========================================================
  pick_topic_ = declare_parameter<std::string>(
    "pick_topic", 
    "/moma/pick_point_base"
  );

  obstacle_cloud_topic_ = declare_parameter<std::string>(
    "obstacle_cloud_topic",
    "/moma/obstacle_cloud"
  );

  trajectory_topic_ = declare_parameter<std::string>(
    "trajectory_topic",
    "/moma/planned_joint_trajectory"
  );

  // ==========================================================
  // OctoMap synchronization
  // ==========================================================

  require_octomap_ = declare_parameter<bool>(
    "require_octomap",
    true
  );

  max_octomap_retries_ = declare_parameter<int>(
    "max_octomap_retries",
    10
  );

  octomap_retry_interval_ms_ = declare_parameter<int>(
    "octomap_retry_interval_ms",
    200
  );

  // ==========================================================
  // Callback group
  //
  // MutuallyExclusive is enough here because we do not block
  // while waiting for the planning-scene service.
  // ==========================================================

  callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions subscription_options;

  subscription_options.callback_group = callback_group_;

  // ==========================================================
  // Pick point subscription
  // ==========================================================

  pick_subscription_ = create_subscription<geometry_msgs::msg::PointStamped>(
    pick_topic_,
    10,
    std::bind(&DepalletizingCoordinator::pick_callback,this,std::placeholders::_1),
    subscription_options
  );

  // ==========================================================
  // Obstacle-cloud subscription
  // ==========================================================

  obstacle_cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    obstacle_cloud_topic_,
    rclcpp::QoS(5).reliable(),
    std::bind(&DepalletizingCoordinator::obstacle_cloud_callback,this,std::placeholders::_1),
    subscription_options
  );

  // ==========================================================
  // Planned trajectory publisher
  // ==========================================================
  trajectory_publisher_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
    trajectory_topic_,
    10
  );

  trajectory_execution_result_subscriber_ = create_subscription<std_msgs::msg::Bool>(
    "/moma/arc_trajectory_execution_result",
    10,
    std::bind(&DepalletizingCoordinator::trajectory_execution_result_callback,this,std::placeholders::_1)
  );

  // ==========================================================
  // MoveIt planning-scene service
  // ==========================================================

  planning_scene_client_ = create_client<moveit_msgs::srv::GetPlanningScene>(
    "/get_planning_scene",
    rmw_qos_profile_services_default,
    callback_group_
  );

  // ==========================================================
  // MotionPlanner
  //
  // MotionPlanner does the actual MoveIt/Pilz planning.
  // Coordinator only decides what motion should happen.
  // ==========================================================

  motion_planner_ =std::make_shared<moma_motion_planning::MotionPlanner>(shared_from_this());

  if (!motion_planner_->initialize(
        group_name_,
        base_frame_,
        tcp_link_,
        planning_pipeline_,
        planning_time_,
        planning_attempts_,
        velocity_scaling_,
        acceleration_scaling_))
  {
    RCLCPP_ERROR(
      get_logger(),
      "Failed to initialize MotionPlanner."
    );

    return false;
  }

  set_state(DepalletizingState::IDLE);

  // Debugging information
  RCLCPP_INFO(
    get_logger(),
    "DepalletizingCoordinator initialized."
  );

  RCLCPP_INFO(
    get_logger(),
    "Pick topic: %s",
    pick_topic_.c_str()
  );

  RCLCPP_INFO(
    get_logger(),
    "Obstacle cloud topic: %s",
    obstacle_cloud_topic_.c_str()
  );

  RCLCPP_INFO(
    get_logger(),
    "Trajectory topic: %s",
    trajectory_topic_.c_str()
  );

  RCLCPP_INFO(
    this->get_logger(),
    "Vacuum configured: Elite standard DO%d via /io_and_status_controller/set_io",
    vacuum_do_pin_
  );

  RCLCPP_INFO(
    this->get_logger(),
    "Vacuum build wait: %.2f s",
    vacuum_build_wait_sec_
  );

  return true;
}

// ============================================================
// Trajectory  result callback
// ============================================================
void DepalletizingCoordinator::trajectory_execution_result_callback(
  const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg->data)
  {
    RCLCPP_ERROR(
      get_logger(),
      "Physical trajectory execution FAILED."
    );

    phase_ = DepalletizingPhase::ERROR;
    return;
  }

  switch (phase_)
  {
    case DepalletizingPhase::PICK:
      handle_pick_execution_success();
      break;

    // To be added
    // case DepalletizingPhase::PLACE:
    //   handle_place_execution_success();
    //   break;

    // case DepalletizingPhase::COMPLETE:
    //   handle_complete_execution_success();
    //   break;

    default:
      RCLCPP_WARN(
        get_logger(),
        "Trajectory result received in unexpected phase."
      );
      break;
  }
}
// ============================================================
// Pick callback
//
// Store the target, but do not plan immediately. Wait for obstacle cloud first.
// ============================================================

void DepalletizingCoordinator::pick_callback(
  const geometry_msgs::msg::PointStamped::SharedPtr message
) 
{
  if (state_ != DepalletizingState::IDLE) {
    RCLCPP_WARN(
      get_logger(),
      "Ignoring pick point because a depalletizing cycle "
      "is already active."
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

  // Validate pick point
  if (
    !std::isfinite(message->point.x) ||
    !std::isfinite(message->point.y) ||
    !std::isfinite(message->point.z))
  {
    RCLCPP_ERROR(
      get_logger(),
      "Received invalid pick point."
    );

    return;
  }

  // ----------------------------------------------------------
  // Store target.
  // ----------------------------------------------------------

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

  // Change state to WAITING_FOR_OBSTACLE_CLOUD
  set_state(DepalletizingState::WAITING_FOR_OBSTACLE_CLOUD);

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
  if (state_ != DepalletizingState::WAITING_FOR_OBSTACLE_CLOUD){
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

    plan_pre_pick();

    return;
  }

  // Wait for MoveIt to consume the cloud and update the planning scene
  set_state(DepalletizingState::WAITING_FOR_OCTOMAP);

  octomap_retry_count_=0;
  
  request_octomap_check();
}

// ============================================================
// Request current MoveIt OctoMap
// ============================================================
void DepalletizingCoordinator::request_octomap_check()
{
  if (state_ != DepalletizingState::WAITING_FOR_OCTOMAP) {
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

      if (state_ != DepalletizingState::WAITING_FOR_OCTOMAP)
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
      //   pick point
      //   obstacle cloud
      //   populated MoveIt OctoMap
      //
      // Therefore planning may begin.
      plan_pre_pick();
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
// Generate pick and pre-pick poses
//
// CURRENT ASSUMPTION:
//   All boxes are flat.
//
// Therefore:
//   TCP orientation = fixed top-down quaternion
//   approach direction = base_link +Z
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
// Generate fixed place and pre-place poses
// ============================================================
void DepalletizingCoordinator::generate_place_poses()
{
  place_pose_.header.frame_id = base_frame_;
  place_pose_.header.stamp = now();

  place_pose_.pose.position.x = place_x_;
  place_pose_.pose.position.y = place_y_;
  place_pose_.pose.position.z = place_z_;
  place_pose_.pose.orientation = place_orientation_;

  pre_place_pose_ =place_pose_;

  pre_place_pose_.pose.position.z += pre_place_offset_;

  RCLCPP_INFO(
    get_logger(),
    "Place pose: "
    "[%.4f, %.4f, %.4f]",
    place_pose_.pose.position.x,
    place_pose_.pose.position.y,
    place_pose_.pose.position.z
  );


  RCLCPP_INFO(
    get_logger(),
    "Pre-place pose: "
    "[%.4f, %.4f, %.4f]",
    pre_place_pose_.pose.position.x,
    pre_place_pose_.pose.position.y,
    pre_place_pose_.pose.position.z
  );
}





}