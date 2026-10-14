#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
namespace unseen_sku_motion_planning
{
enum class MotionType
{
  PTP,
  LIN
};

struct CandidatePlan
{
  geometry_msgs::msg::PoseStamped pre_pick_pose;
  geometry_msgs::msg::PoseStamped pick_pose;
  geometry_msgs::msg::PoseStamped retract_pose;

  trajectory_msgs::msg::JointTrajectory pre_pick_trajectory;
  trajectory_msgs::msg::JointTrajectory pick_trajectory;
  trajectory_msgs::msg::JointTrajectory retract_trajectory;

  CandidatePlan() = default;
};

// =============================================================
// MotionPlanner
//
// Responsibility:
//   - Interface with MoveIt
//   - Select Pilz PTP or LIN
//   - Plan motion to a Cartesian pose
//   - Plan PTP motion to a named joint configuration such as HOME
//   - Return the resulting JointTrajectory
//
// It does NOT:
//   - Execute trajectories
//   - Control vacuum
//   - Manage the depalletizing sequence
//   - Wait for OctoMap
// =============================================================

class MotionPlanner {
  public:
    explicit MotionPlanner(const rclcpp::Node::SharedPtr & node);

    // -----------------------------------------------------------
    // Initialize MoveGroupInterface
    // -----------------------------------------------------------
    bool initialize(
      const std::string & group_name,
      const std::string & base_frame,
      const std::string & tcp_link,
      const std::string & planning_pipeline,
      // const std::string & planner_id,
      double planning_time,
      int planning_attempts,
      double velocity_scaling,
      double acceleration_scaling
    );
    
    // -----------------------------------------------------------
    // Plan to a Cartesian TCP pose.
    //
    // Examples:
    //
    // PTP -> pre-pick
    // LIN -> pick
    // LIN -> retract
    // PTP -> pre-place
    // LIN -> place
    // -----------------------------------------------------------
    bool plan_to_pose(
      const geometry_msgs::msg::PoseStamped & target_pose,
      MotionType motion_type,
      trajectory_msgs::msg::JointTrajectory & output_trajectory
    );

    // -----------------------------------------------------------
    // Plan to a named joint configuration from the SRDF.
    //
    // Example:
    //
    // plan_to_named_target(
    //   "home",
    //   output_trajectory
    // );
    //
    // Named-target motion will use Pilz PTP.
    // -----------------------------------------------------------
    bool plan_to_named_target(
      const std::string & target_name,
      trajectory_msgs::msg::JointTrajectory & output_trajectory
    );

    // Plan the complete motion sequence for ONE suction candidate.
    //
    // Surface normal convention:
    //   Normal points outward from the object's surface.
    //
    // Motion sequence:
    //   1. PTP to pre-pick
    //   2. LIN to pick
    //   3. LIN to retract
    //
    // All stages are planned without executing.
    //
    // Returns false if any required stage fails.
    bool plan_candidate(
      const geometry_msgs::msg::PointStamped & suction_point,
      const geometry_msgs::msg::Vector3Stamped & surface_normal,
      double pre_pick_distance_m,
      double retract_distance_m,
      CandidatePlan & output,
      std::string & failure_reason
    );

  private:
    std::string motion_type_to_planner_id(MotionType motion_type);

    rclcpp::Node::SharedPtr node_;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

    std::string base_frame_;
    std::string tcp_link_;
    std::string planning_pipeline_;
};

} // namespace unseen_sku_motion_planning