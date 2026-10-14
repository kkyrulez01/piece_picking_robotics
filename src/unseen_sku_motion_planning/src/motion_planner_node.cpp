#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "unseen_sku_motion_planning/motion_planner.hpp"
#include "unseen_sku_motion_planning/srv/plan_candidate.hpp"

namespace unseen_sku_motion_planning
{

class MotionPlannerNode : public rclcpp::Node
{
  public:

    MotionPlannerNode() : Node("unseen_sku_motion_planner_core")
    {
      // Robot configuration
      group_name_ = declare_parameter<std::string>(
        "group_name", "ARC_S6"
      );
      base_frame_ = declare_parameter<std::string>(
        "base_frame", "base_link"
      );
      tcp_link_ = declare_parameter<std::string>(
        "tcp_link", "vacuum_tcp"
      );

      // MoveIt configuration
      planning_pipeline_ = declare_parameter<std::string>(
        "planning_pipeline",
        "pilz_industrial_motion_planner"
      );

      planning_time_ = declare_parameter<double>(
        "planning_time", 5.0
      );
      planning_attempts_ = declare_parameter<int>(
        "planning_attempts", 3
      );

      velocity_scaling_ = declare_parameter<double>(
        "velocity_scaling", 0.10
      );
      acceleration_scaling_ =declare_parameter<double>(
        "acceleration_scaling", 0.10
      );

      // Pick geometry
      pre_pick_distance_m_ = declare_parameter<double>(
          "pre_pick_distance_m", 0.06
      );

      retract_distance_m_ = declare_parameter<double>(
          "retract_distance_m", 0.06
      );
    }

    // Initialize MoveIt planner
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

      // Single-candidate planning service
      plan_candidate_service_ = create_service<srv::PlanCandidate>(
          "/unseen_sku/plan_candidate",
          std::bind(
              &MotionPlannerNode::plan_candidate_callback,
              this,
              std::placeholders::_1,
              std::placeholders::_2
          )
      );

      RCLCPP_INFO(
          get_logger(),
          "Unseen-SKU Motion Planner ready"
      );

      RCLCPP_INFO(
          get_logger(),
          "Service: /unseen_sku/plan_candidate"
      );

      return true;
    }
  
  private:
    
    // PlanCandidate ROS 2 callback
    void plan_candidate_callback(
        const std::shared_ptr<srv::PlanCandidate::Request> request,
        std::shared_ptr<srv::PlanCandidate::Response> response
    )
    {
      // Only one MoveIt planning request at a time.
      std::lock_guard<std::mutex> lock(planning_mutex_);

      if (!motion_planner_)
      {
          response->success = false;
          response->message = "MotionPlanner is not initialized.";
          return;
      }

      CandidatePlan result;
      std::string failure_reason;

      RCLCPP_INFO(
          get_logger(),
          "Received single-candidate planning request."
      );

      try
      {
        // All trajectory-planning delegated to C++
        response->success = motion_planner_->plan_candidate(
          request->suction_point,
          request->surface_normal,
          pre_pick_distance_m_,
          retract_distance_m_,
          result,
          failure_reason 
        );
      }

      catch (const std::exception & error)
      {
        response->success = false;
        response->message =
            std::string("Planning exception: ") + error.what();

        RCLCPP_ERROR(
            get_logger(),
            "%s",
            response->message.c_str()
        );

        return;
      }

      if (!response->success)
      {
          response->message = failure_reason.empty()
              ? "Candidate planning failed."
              : failure_reason;

          RCLCPP_WARN(
              get_logger(),
              "Candidate rejected: %s",
              response->message.c_str()
          );

          return;
      }

      // Copy successful planning results into ROS response
      response->pre_pick_pose = result.pre_pick_pose;
      response->pick_pose = result.pick_pose;
      response->retract_pose = result.retract_pose;

      response->pre_pick_trajectory =
          result.pre_pick_trajectory;

      response->pick_trajectory =
          result.pick_trajectory;

      response->retract_trajectory =
          result.retract_trajectory;

      response->message =
          "Complete candidate trajectory planned successfully.";

      RCLCPP_INFO(
          get_logger(),
          "Planning successful."
      );

      RCLCPP_INFO(
          get_logger(),
          "Pre-pick PTP: %zu points",
          response->pre_pick_trajectory.points.size()
      );

      RCLCPP_INFO(
          get_logger(),
          "Pick LIN: %zu points",
          response->pick_trajectory.points.size()
      );

      RCLCPP_INFO(
          get_logger(),
          "Retract LIN: %zu points",
          response->retract_trajectory.points.size()
      );
    }

    // Members
    std::shared_ptr<MotionPlanner> motion_planner_;

    rclcpp::Service<srv::PlanCandidate>::SharedPtr plan_candidate_service_;

    std::mutex planning_mutex_;

    // Robot config
    std::string group_name_;
    std::string base_frame_;
    std::string tcp_link_;

    // MoveIt config
    std::string planning_pipeline_;

    double planning_time_;
    int planning_attempts_;

    double velocity_scaling_;
    double acceleration_scaling_;

    // Approach configuration
    double pre_pick_distance_m_;
    double retract_distance_m_;
};

} // namespace unseen_sku_motion_planning

// Main
int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<
        unseen_sku_motion_planning::MotionPlannerNode
    >();

    int exit_code = 0;

    try
    {
        if (!node->initialize())
        {
            RCLCPP_FATAL(
                node->get_logger(),
                "Motion planner initialization failed."
            );

            exit_code = 1;
        }
        else
        {
            rclcpp::spin(node);
        }
    }
    catch (const std::exception & error)
    {
        RCLCPP_FATAL(
            node->get_logger(),
            "Motion planner error: %s",
            error.what()
        );

        exit_code = 1;
    }

    rclcpp::shutdown();

    return exit_code;
}