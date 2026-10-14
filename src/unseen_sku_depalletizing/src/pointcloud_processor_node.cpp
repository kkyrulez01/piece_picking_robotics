#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


class PointCloudProcessorNode : public rclcpp::Node
{
  public:
    PointCloudProcessorNode()
    : Node("pointcloud_processor_node")
    { 
      // All filtering will eventually happen in this frame
      target_frame_ = declare_parameter<std::string>("target_frame", "base_link");

      // Input and output topics
      input_topic_ = declare_parameter<std::string>(
        "input_topic",
        "/mechmind/point_cloud"
      );

      output_topic_ = declare_parameter<std::string>(
        "output_topic",
        "/moma/obstacle_cloud"
      );

      // Filtering parameters
      leaf_size_ = declare_parameter<double>("leaf_size", 0.01); // Small leaf size of 10mm
      
      enable_crop_ = declare_parameter<bool>("enable_crop", true);

      // Initial CropBox limits in base_link frame
      min_x_ = declare_parameter<double>("min_x", -1.0);
      max_x_ = declare_parameter<double>("max_x", -0.4);

      min_y_ = declare_parameter<double>("min_y", -0.5);
      max_y_ = declare_parameter<double>("max_y", 1.0);

      min_z_ = declare_parameter<double>("min_z", -0.8);
      max_z_ = declare_parameter<double>("max_z", 0.6);

      transform_timeout_ = declare_parameter<double>("transform_timeout", 0.2);

      // TF
      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());

      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      // ROS Publishers and subscribers
      publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        output_topic_,
        rclcpp::QoS(5).reliable()
      );

      subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_,
        rclcpp::QoS(10).reliable(),
        std::bind(&PointCloudProcessorNode::pointcloud_callback, this, std::placeholders::_1)
      );
      
      // Information for user
      RCLCPP_INFO(
        get_logger(),
        "Voxel size: %.3f m",
        leaf_size_
      );

      RCLCPP_INFO(
        get_logger(),
        "Crop limits: x[%.2f, %.2f], y[%.2f, %.2f], z[%.2f, %.2f]",
        min_x_, max_x_,
        min_y_, max_y_,
        min_z_, max_z_
      );
    }
  
  private:
    // Transform entire PointCloud2 into base_link frame
    bool transform_cloud(
      const sensor_msgs::msg::PointCloud2 & input,
      sensor_msgs::msg::PointCloud2 & output
    )
    {
      if (input.header.frame_id.empty()) {
        RCLCPP_ERROR(
          get_logger(),
          "Incoming point cloud has empty frame_id."
        );

      return false;
      }

      // No transform required if cloud is already expressed in requested target frame
      if (input.header.frame_id == target_frame_) {
        output = input;
        return true;
      }

      try
      {
        const geometry_msgs::msg::TransformStamped transform =
          tf_buffer_->lookupTransform(
            target_frame_,
            input.header.frame_id,
            rclcpp::Time(input.header.stamp),
            rclcpp::Duration::from_seconds(
              transform_timeout_
            )
          );
        
          // Apply camera -> base_link transform to every XYZ point.
          tf2::doTransform(input, output, transform);

          output.header.frame_id = target_frame_;
          output.header.stamp = input.header.stamp;

          return true;
      }

      catch (const tf2::TransformException & exception)
      {
        RCLCPP_WARN(
          get_logger(),
          "Could not transform point cloud "
          "from '%s' to '%s': %s",
          input.header.frame_id.c_str(),
          target_frame_.c_str(),
          exception.what()
        );

        return false;
      }
    }
    
    void pointcloud_callback(
      const sensor_msgs::msg::PointCloud2::SharedPtr message
    )
    { 
      // 1. Transform camera cloud -> base_link
      sensor_msgs::msg::PointCloud2 transformed_message;
      
      if (!transform_cloud(*message, transformed_message)) {
        return;
      }
      
      // 2. Convert ROS PointCloud2 into a PCL XYZ point cloud.
      auto input_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      pcl::fromROSMsg(transformed_message, *input_cloud); // Performs the ROS-to-PCL conversion

      // Get total number of 3D points
      const std::size_t input_count = input_cloud->size();

      if (input_cloud->empty()) {
        RCLCPP_WARN(get_logger(), "Received an empty point cloud");
        return;
      }

      // 3. Remove NaN and invalid XYZ points
      auto valid_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

      std::vector<int> valid_indices;

      pcl::removeNaNFromPointCloud(*input_cloud, *valid_cloud, valid_indices);

      // =========================================================
      // Debug: find real cloud bounds in base_link
      // =========================================================

      pcl::PointXYZ cloud_min;
      pcl::PointXYZ cloud_max;

      pcl::getMinMax3D(
        *valid_cloud,
        cloud_min,
        cloud_max
      );

      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Raw valid cloud bounds in '%s': "
        "x[%.3f, %.3f] "
        "y[%.3f, %.3f] "
        "z[%.3f, %.3f]",
        target_frame_.c_str(),
        cloud_min.x,
        cloud_max.x,
        cloud_min.y,
        cloud_max.y,
        cloud_min.z,
        cloud_max.z
      );

      // 4. Keep only required workspace
      auto cropped_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      
      if (enable_crop_) {
        pcl::CropBox<pcl::PointXYZ> crop_filter; // Create crop box filter object
        crop_filter.setInputCloud(valid_cloud);

        // Define 3D bounding box
        crop_filter.setMin(
          Eigen::Vector4f(
          static_cast<float>(min_x_),
          static_cast<float>(min_y_),
          static_cast<float>(min_z_),
          1.0F // Scaling factor in homogeneous transformation
          )
        );

        crop_filter.setMax(
          Eigen::Vector4f(
          static_cast<float>(max_x_),
          static_cast<float>(max_y_),
          static_cast<float>(max_z_),
          1.0F // Scaling factor in homogeneous transformation
          ) 
        );

        crop_filter.filter(*cropped_cloud); // Applies filter and return cropped cloud
      }

      else {
        *cropped_cloud = *valid_cloud;
      }

      if (cropped_cloud->empty()) {
        RCLCPP_WARN(
          get_logger(),
          "CropBox removed the entire cloud."
        );

        return;
      }
      

      // 5. Downsample the cloud using a voxel grid
      auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

      pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
      voxel_filter.setInputCloud(cropped_cloud);

      const float leaf_size = static_cast<float>(leaf_size_);

      voxel_filter.setLeafSize(
        leaf_size,
        leaf_size,
        leaf_size
      );

      voxel_filter.filter(*filtered_cloud);

      // 6. Convert filtered PCL cloud back to ROS PointCloud2
      sensor_msgs::msg::PointCloud2 output_message;
      pcl::toROSMsg(*filtered_cloud, output_message);

      
      // Preserve original timestamp. frame_id is now base_link.
      output_message.header.frame_id = target_frame_;
      output_message.header.stamp = message->header.stamp;

      // 7. Publish final MoveIt planning cloud
      publisher_->publish(output_message);

      RCLCPP_INFO(
        get_logger(),
        "Points: input=%zu, valid=%zu, cropped=%zu, filtered=%zu",
        input_count,
        valid_cloud->size(),
        cropped_cloud->size(),
        filtered_cloud->size()
      );
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Parameters
    std::string input_topic_;
    std::string output_topic_;
    std::string target_frame_;

    double leaf_size_;

    bool enable_crop_;

    double min_x_;
    double max_x_;

    double min_y_;
    double max_y_;

    double min_z_;
    double max_z_;

    double transform_timeout_;
  };

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PointCloudProcessorNode>();
  rclcpp::spin(node);
  
  rclcpp::shutdown();
  return 0;
}