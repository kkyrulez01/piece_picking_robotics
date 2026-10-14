#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "unseen_sku_motion_planning/motion_planner.hpp"
#include "unseen_sku_motion_planning/srv/plan_pose.hpp"

namespace unseen_sku_motion_planning
{

class MotionPlannerServiceNode : public rclcpp::Node
{
  public:
    MotionPlannerServiceNode() : Node("motion_planner_service_node")
    {
      // Parameters
      group_name_ = declare_parameter<std::string>("group_name","ARC_S6");

      base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
      tcp_link_ = declare_parameter<std::string>("tcp_link", "vacuum_tcp");

      planning_pipeline_ =declare_parameter<std::string>(
          "planning_pipeline",
          "pilz_industrial_motion_planner"
        );

      planning_time_ = declare_parameter<double>("planning_time", 5.0);
      planning_attempts_ = declare_parameter<int>("planning_attempts",3);

      velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.10);
      acceleration_scaling_ =declare_parameter<double>("acceleration_scaling", 0.10);

      trajectory_topic_ = declare_parameter<std::string>(
        "trajectory_topic", 
        "/unseen_sku/planned_joint_trajectory"
      );

      // Trajectory publisher
      trajectory_publisher_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
          trajectory_topic_,
          10
      );

      RCLCPP_INFO(
        get_logger(),
        "MotionPlanner service node constructed."
      );
    }

    // ============================================================
    // Initialization
    //
    // Called AFTER the node has been created with make_shared().
    // This is important because MotionPlanner requires a shared
    // pointer to this ROS node.
    // ============================================================
    bool initialize()
    {
      motion_planner_ = std::make_shared<MotionPlanner>(shared_from_this());

      const bool initialized = motion_planner_->initialize(
        group_name_,
        base_frame_,
        tcp_link_,
        planning_pipeline_,
        planning_time_,
        planning_attempts_,
        velocity_scaling_,
        acceleration_scaling_
      );

      if (!initialized) {
        RCLCPP_ERROR(
          get_logger(),
          "Failed to initialize MotionPlanner."
        );

        return false;
      }

      // PlanPose service

      plan_pose_service_ = create_service<unseen_sku_motion_planning::srv::PlanPose>(
        "/unseen_sku/plan_pose",
        std::bind(&MotionPlannerServiceNode::plan_pose_callback,this,std::placeholders::_1,std::placeholders::_2)
      );

      RCLCPP_INFO(
        get_logger(),
        "MotionPlanner initialized."
      );

      RCLCPP_INFO(
        get_logger(),
        "PlanPose service ready: /unseen_sku/plan_pose"
      );

      RCLCPP_INFO(
        get_logger(),
        "Trajectory output: %s",
        trajectory_topic_.c_str()
      );

      return true;
    }
  
  private:
    
  // PlanPose service callback
  void plan_pose_callback(
    const std::shared_ptr<unseen_sku_motion_planning::srv::PlanPose::Request> request,
    std::shared_ptr<unseen_sku_motion_planning::srv::PlanPose::Response> response
  )
  {
    if (!motion_planner_) {
      response->success = false;
      response->message =
        "MotionPlanner is not initialized.";

      return;
    }

    // Determine requested motion type
     MotionType motion_type;

    if (request->motion_type == "PTP")
    {
      motion_type = MotionType::PTP;
    }
    else if (request->motion_type == "LIN")
    {
      motion_type = MotionType::LIN;
    }
    else
    {
      response->success = false;

      response->message =
        "Invalid motion_type '" +
        request->motion_type +
        "'. Expected 'PTP' or 'LIN'.";

      RCLCPP_ERROR(
        get_logger(),
        "%s",
        response->message.c_str()
      );

      return;
    }

    // Log requested target
    const auto & target =
      request->target_pose;

    RCLCPP_INFO(
      get_logger(),
      "PlanPose request: %s",
      request->motion_type.c_str()
    );

    RCLCPP_INFO(
      get_logger(),
      "Target: frame='%s', "
      "xyz=[%.4f, %.4f, %.4f]",
      target.header.frame_id.c_str(),
      target.pose.position.x,
      target.pose.position.y,
      target.pose.position.z
    );

    // Ask existing MotionPlanner to generate trajectory
    trajectory_msgs::msg::JointTrajectory trajectory;

    const bool success =
      motion_planner_->plan_to_pose(
        target,
        motion_type,
        trajectory
      );

    if (!success)
    {
      response->success = false;

      response->message =
        request->motion_type +
        " motion planning failed.";

      RCLCPP_ERROR(
        get_logger(),
        "%s",
        response->message.c_str()
      );

      return;
    }

    // Sanity check
    if (trajectory.points.empty())
    {
      response->success = false;
      response->message = "MotionPlanner returned an empty trajectory.";

      RCLCPP_ERROR(
        get_logger(),
        "%s",
        response->message.c_str()
      );

      return;
    }

    // Publish trajectory
    //
    // From here it continues through:
    //
    // /unseen_sku/planned_joint_trajectory
    //          ↓
    // joint convention converter
    //          ↓
    // /unseen_sku/ARC_joint_trajectory
    //          ↓
    // ARC trajectory executor
    trajectory_publisher_->publish(
      trajectory
    );

    response->success = true;

    response->message =
      request->motion_type +
      " trajectory planned and published successfully.";

    RCLCPP_INFO(
      get_logger(),
      "%s Trajectory contains %zu points.",
      response->message.c_str(),
      trajectory.points.size()
    );
  }

  // ============================================================
  // MotionPlanner
  // ============================================================
  std::shared_ptr<MotionPlanner> motion_planner_;

  // ROS interfaces
  rclcpp::Service<
    unseen_sku_motion_planning::srv::PlanPose
  >::SharedPtr plan_pose_service_;

  rclcpp::Publisher<
    trajectory_msgs::msg::JointTrajectory
  >::SharedPtr trajectory_publisher_;

  // Parameters
  std::string group_name_;
  std::string base_frame_;
  std::string tcp_link_;
  std::string planning_pipeline_;
  std::string trajectory_topic_;

  double planning_time_;
  int planning_attempts_;

  double velocity_scaling_;
  double acceleration_scaling_;
};

} // namespace unseen_sku_motion_planning

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<
      unseen_sku_motion_planning::MotionPlannerServiceNode
    >();

  if (!node->initialize())
  {
    RCLCPP_FATAL(
      node->get_logger(),
      "MotionPlannerServiceNode initialization failed."
    );

    rclcpp::shutdown();

    return 1;
  }

  rclcpp::spin(node);

  rclcpp::shutdown();

  return 0;
}