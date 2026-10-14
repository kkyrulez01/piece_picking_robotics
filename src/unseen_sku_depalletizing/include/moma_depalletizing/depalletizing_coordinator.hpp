#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <moveit_msgs/srv/get_planning_scene.hpp>

#include "moma_motion_planning/motion_planner.hpp"
#include "moma_depalletizing/depalletizing_state.hpp"

#include <eli_common_interface/srv/set_io.hpp>
namespace moma_depalletizing
{

class DepalletizingCoordinator : public rclcpp::Node
{
  public:
    DepalletizingCoordinator();

    bool initialize();
  
  private:

    // ==========================================================
    // PICK PHASE
    // Implemented in pick_phase.cpp
    // ==========================================================
    // Perception / environment
    void pick_callback(const geometry_msgs::msg::PointStamped::SharedPtr message);

    void obstacle_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr message);

    // OctoMap synchronization
    void request_octomap_check();
    void schedule_octomap_retry();

    // Pose generation
    bool generate_pick_poses(const geometry_msgs::msg::PointStamped & pick_point);

    // Trajectory exeuction result
    void trajectory_execution_result_callback(const std_msgs::msg::Bool::SharedPtr msg);

    // Pick sequence
    void plan_pre_pick();
    void plan_pick();
    void activate_suction();
    void vacuum_on_response(rclcpp::Client<eli_common_interface::srv::SetIO>::SharedFuture future);
    void vacuum_build_complete();
    void plan_retract();
    
    // Pick execution success
    void handle_pick_execution_success();

    // ==========================================================
    // PLACE PHASE
    // Implemented in place_phase.cpp
    // ==========================================================

    void generate_place_poses();
    void plan_pre_place();
    void plan_place();
    void release_suction();
    void plan_place_retreat();

    // Place execution success (TO BE ADDED)
    // void handle_place_execution_success();

    // ==========================================================
    // COMPLETE PHASE
    // Implemented in complete_phase.cpp
    // ==========================================================

    void check_for_more_objects();
    void plan_home();

    // Planning helpers
    bool plan_and_publish_pose(
      const geometry_msgs::msg::PoseStamped & target_pose,
      moma_motion_planning::MotionType motion_type
    );

    bool plan_and_publish_named_target(
      const std::string & target_name
    );

    // Phase management
    void set_phase(
      DepalletizingPhase new_phase
    );

    void fail(
      const std::string & reason
    );

    // Complete execution success (TO BE ADDED)
    // void handle_complete_execution_success();

    // =========================================================
    // Execution mode
    // =========================================================
    ExecutionMode execution_mode_{ExecutionMode::MANUAL};
    
    std::string execution_mode_string_;

    // =========================================================
    // Current phase/state
    // =========================================================
    DepalletizingPhase phase_{DepalletizingPhase::IDLE};

    PickState pick_state_{PickState::WAITING_FOR_PICK_POINT};

    PlaceState place_state_{PlaceState::PLANNING_PRE_PLACE};

    CompleteState complete_state_{CompleteState::CHECKING_FOR_MORE_OBJECTS};

    // =========================================================
    // Current pick target
    // =========================================================

    std::mutex target_mutex_;

    geometry_msgs::msg::PointStamped::SharedPtr pending_pick_point_;

    // =========================================================
    // Generated poses
    // =========================================================

    geometry_msgs::msg::PoseStamped pick_pose_;
    geometry_msgs::msg::PoseStamped pre_pick_pose_;

    geometry_msgs::msg::PoseStamped place_pose_;
    geometry_msgs::msg::PoseStamped pre_place_pose_;

    // =========================================================
    // Motion planner
    // =========================================================

    std::shared_ptr<moma_motion_planning::MotionPlanner> motion_planner_;

    // =========================================================
    // Robot / MoveIt configuration
    // =========================================================

    std::string group_name_;
    std::string base_frame_;
    std::string tcp_link_;

    std::string planning_pipeline_;

    double planning_time_;
    int planning_attempts_;

    double velocity_scaling_;
    double acceleration_scaling_;

    // =========================================================
    // Pick configuration
    // =========================================================

    double pre_pick_offset_;

    geometry_msgs::msg::Quaternion
      tcp_orientation_;


    // =========================================================
    // Fixed place configuration
    // =========================================================

    double place_x_;
    double place_y_;
    double place_z_;

    geometry_msgs::msg::Quaternion
      place_orientation_;

    double pre_place_offset_;

    // HOME is an SRDF named joint state
    std::string home_target_name_;

    // =========================================================
    // Topics
    // =========================================================

    std::string pick_topic_;
    std::string obstacle_cloud_topic_;
    std::string trajectory_topic_;

    // =========================================================
    // OctoMap
    // =========================================================

    bool require_octomap_;

    std::atomic_bool planning_scene_request_in_flight_{false};

    std::atomic<int> octomap_retry_count_{0};

    int max_octomap_retries_{10};
    int octomap_retry_interval_ms_{200};

    rclcpp::TimerBase::SharedPtr octomap_retry_timer_;

    // =========================================================
    // ROS interfaces
    // =========================================================

    rclcpp::Subscription<
      geometry_msgs::msg::PointStamped
    >::SharedPtr pick_subscription_;

    rclcpp::Subscription<
      sensor_msgs::msg::PointCloud2
    >::SharedPtr obstacle_cloud_subscription_;

    rclcpp::Subscription<
      std_msgs::msg::Bool
    >::SharedPtr trajectory_execution_result_subscriber_;

    rclcpp::Publisher<
      trajectory_msgs::msg::JointTrajectory
    >::SharedPtr trajectory_publisher_;

    rclcpp::Client<
      moveit_msgs::srv::GetPlanningScene
    >::SharedPtr planning_scene_client_;

    rclcpp::CallbackGroup::SharedPtr
      callback_group_;
    
    // Vacuum related
    rclcpp::Client<
      eli_common_interface::srv::SetIO
    >::SharedPtr vacuum_io_client_;

    rclcpp::TimerBase::SharedPtr vacuum_build_timer_;

    int vacuum_do_pin_{0};
    double vacuum_build_wait_sec_{1.0};

    bool vacuum_enabled_{false};
};

} // namespace moma_depalletizing
