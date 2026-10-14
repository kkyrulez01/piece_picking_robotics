#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <visualization_msgs/msg/marker.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class MaskPointCloudFusion : public rclcpp::Node
{
  public:
    MaskPointCloudFusion() : Node("mask_pointcloud_fusion")
    {
      mask_threshold_ = static_cast<int>(declare_parameter<std::int64_t>("mask_threshold", 127));

      pixel_stride_ = static_cast<int>(declare_parameter<std::int64_t>("pixel_stride", 1));

      min_valid_points_ = static_cast<int>(declare_parameter<std::int64_t>("min_valid_points", 100));

      marker_diameter_ = declare_parameter<double>("marker_diameter", 0.03);

      consume_mask_after_fusion_ = declare_parameter<bool>("consume_mask_after_fusion", true);

      const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

      mask_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        "/moma/selected_box_mask",
        qos,
        std::bind(&MaskPointCloudFusion::mask_callback, this, std::placeholders::_1)
      );

      cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/mechmind/point_cloud",
        qos,
        std::bind(&MaskPointCloudFusion::cloud_callback, this, std::placeholders::_1)
      );

      selected_point_publisher_ = create_publisher<geometry_msgs::msg::PointStamped>(
        "/moma/selected_pick_point",
        10
      );
      
      selected_cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/moma/selected_box_points",
        10
      );

      marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
        "/moma/selected_box_marker",
        10
      );

      RCLCPP_INFO(
      get_logger(),
      "Mask-point-cloud fusion started");

      RCLCPP_INFO(
        get_logger(),
        "Mask topic: /moma/selected_box_mask");

      RCLCPP_INFO(
        get_logger(),
        "Cloud topic: /mechmind/point_cloud");
    }
  
  private:
    static double median(std::vector<float> values) {
      if (values.empty()) {
        return 0.0;
      }

      const std::size_t middle = values.size() / 2;

      std::nth_element(values.begin(), values.begin() + middle, values.end());

      const float upper = values[middle];

      if (values.size() % 2 != 0) {
        return static_cast<double>(upper);
      }

      std::nth_element(values.begin(), values.begin() + middle - 1, values.end());

      const float lower = values[middle - 1];

      return 0.5 * (static_cast<double>(lower) + static_cast<double>(upper));
    }

    void mask_callback(const sensor_msgs::msg::Image::SharedPtr message) {
      const bool encoding_supported =
        message->encoding ==
          sensor_msgs::image_encodings::MONO8 ||
        message->encoding ==
          sensor_msgs::image_encodings::TYPE_8UC1;

      if (!encoding_supported) {
        RCLCPP_ERROR(
          get_logger(),
          "Unsupported mask encoding '%s'. "
          "Expected mono8 or 8UC1.",
          message->encoding.c_str());

        return;
      }

      if (message->width == 0 || message->height == 0) {
        RCLCPP_ERROR(
          get_logger(),
          "Received an empty mask");

        return;
      }

      if (message->step < message->width) {
        RCLCPP_ERROR(
          get_logger(),
          "Invalid mask step: step=%u width=%u",
          message->step,
          message->width);

        return;
      }

      const std::size_t required_size = 
        static_cast<std::size_t>(message->step) *
        static_cast<std::size_t>(message->height);
      
      if (message->data.size() < required_size) {
        RCLCPP_ERROR(
          get_logger(),
          "Mask data is too small: data=%zu required=%zu",
          message->data.size(),
          required_size);

        return;
      }

      latest_mask_ = message;

      RCLCPP_INFO(
        get_logger(),
        "Stored mask: %ux%u, encoding=%s",
        message->width,
        message->height,
        message->encoding.c_str());
    }

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr message) {
      if (!latest_mask_) {
        RCLCPP_WARN(
          get_logger(),
          "Point cloud received, but no YOLO mask is available");

        return;
      }

      pcl::PointCloud<pcl::PointXYZ> cloud;
      pcl::fromROSMsg(*message, cloud);

      if (cloud.empty()) {
        RCLCPP_WARN(
          get_logger(),
          "Received an empty point cloud");

        return;
      }
      
      if (cloud.height <= 1) {
        RCLCPP_ERROR(
          get_logger(),
          "Point cloud is not organized: width=%u height=%u",
          cloud.width,
          cloud.height);

        return;
      }

      const auto mask = latest_mask_;

      if (mask->width != cloud.width || mask->height != cloud.height) {
        RCLCPP_ERROR(
          get_logger(),
          "Mask/cloud dimension mismatch: "
          "mask=%ux%u cloud=%ux%u",
          mask->width,
          mask->height,
          cloud.width,
          cloud.height);

        return;
      }

      pcl::PointCloud<pcl::PointXYZ> selected_cloud;

      // Assume roughly 25% of the point cloud falls inside the mask region
      selected_cloud.points.reserve(cloud.size() / 4);

      std::vector<float> x_values;
      std::vector<float> y_values;
      std::vector<float> z_values;

      x_values.reserve(cloud.size() / 4);
      y_values.reserve(cloud.size() / 4);
      z_values.reserve(cloud.size() / 4);

      std::size_t white_mask_pixels = 0; // Track how many total 2D pixels fall in mask target area
      std::size_t invalid_cloud_points = 0; // Track how many mask pixels failed to yield 3D depth data

      for (std::uint32_t v = 0; v < cloud.height; v+= static_cast<std::uint32_t>(pixel_stride_)) {
        const std::size_t mask_row = static_cast<std::size_t>(v) * static_cast<std::size_t>(mask->step);

        for (std::uint32_t u = 0; u < cloud.width; u+= static_cast<std::uint32_t>(pixel_stride_)) {
          const std::size_t mask_index = mask_row + static_cast<std::size_t>(u);

          if (mask->data[mask_index] <= static_cast<std::uint8_t>(mask_threshold_)) {
            continue; // Skip pixels darker than mask threshold
          }

          ++white_mask_pixels; // Increase counter for each pixel brighter than threshold

          const pcl::PointXYZ& point = cloud.at(u, v); // point is a nickname for cloud.at(u, v)

          if (
            !std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            !std::isfinite(point.z) ||
            point.z <= 0.0F)
          {
            ++invalid_cloud_points;
            continue;
          }

          selected_cloud.points.push_back(point); // Append point to the end of selected cloud points

          x_values.push_back(point.x);
          y_values.push_back(point.y);
          z_values.push_back(point.z);
        }
      }

    if (selected_cloud.points.size() < static_cast<std::size_t>(min_valid_points_)) {
      RCLCPP_ERROR(
        get_logger(),
        "Not enough valid points selected: "
        "valid=%zu required=%d white_pixels=%zu invalid=%zu",
        selected_cloud.points.size(),
        min_valid_points_,
        white_mask_pixels,
        invalid_cloud_points);

      return;
    }
    
    selected_cloud.width = static_cast<std::uint32_t>(selected_cloud.points.size());
    selected_cloud.height = 1;
    selected_cloud.is_dense = true; // Faster, skips safety checks for bad data

    // Calculate median X, Y, Z values
    const double target_x = median(std::move(x_values));
    const double target_y = median(std::move(y_values));
    const double target_z = median(std::move(z_values));

    publish_selected_cloud(selected_cloud, message->header);

    publish_selected_point(
      target_x,
      target_y,
      target_z,
      message->header);
    
    publish_marker(
    target_x,
    target_y,
    target_z,
    message->header);

    RCLCPP_INFO(
      get_logger(),
      "Selected %zu valid points from %zu mask pixels",
      selected_cloud.points.size(),
      white_mask_pixels);

    RCLCPP_INFO(
      get_logger(),
      "Median target in frame '%s': "
      "x=%.4f y=%.4f z=%.4f",
      message->header.frame_id.c_str(),
      target_x,
      target_y,
      target_z);

    if (consume_mask_after_fusion_) {
      latest_mask_.reset();

      RCLCPP_INFO(
        get_logger(),
        "Mask consumed; waiting for the next YOLO mask");
      }
    }
    
    void publish_selected_cloud(
      const pcl::PointCloud<pcl::PointXYZ>& cloud,
      const std_msgs::msg::Header& header)
    {
      sensor_msgs::msg::PointCloud2 output;
      pcl::toROSMsg(cloud, output);

      output.header = header;

      selected_cloud_publisher_->publish(output);
    }
    
    void publish_selected_point(
      const double x,
      const double y,
      const double z,
      const std_msgs::msg::Header& header)
    {
      geometry_msgs::msg::PointStamped output;

      output.header = header;
      output.point.x = x;
      output.point.y = y;
      output.point.z = z;

      selected_point_publisher_ ->publish(output);
    }

    void publish_marker(
      const double x,
      const double y,
      const double z,
      const std_msgs::msg::Header & header)
    {
      visualization_msgs::msg::Marker marker;

      marker.header = header;

      marker.ns = "selected_box";
      marker.id = 0;

      marker.type =
        visualization_msgs::msg::Marker::SPHERE;

      marker.action =
        visualization_msgs::msg::Marker::ADD;

      marker.pose.position.x = x;
      marker.pose.position.y = y;
      marker.pose.position.z = z;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = marker_diameter_;
      marker.scale.y = marker_diameter_;
      marker.scale.z = marker_diameter_;

      marker.color.r = 1.0F;
      marker.color.g = 0.0F;
      marker.color.b = 0.0F;
      marker.color.a = 1.0F;

      marker_publisher_->publish(marker);
  }

  rclcpp::Subscription<
    sensor_msgs::msg::Image>::SharedPtr
    mask_subscription_;

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    cloud_subscription_;

  rclcpp::Publisher<
    geometry_msgs::msg::PointStamped>::SharedPtr
    selected_point_publisher_;

  rclcpp::Publisher<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    selected_cloud_publisher_;

  rclcpp::Publisher<
    visualization_msgs::msg::Marker>::SharedPtr
    marker_publisher_;

  sensor_msgs::msg::Image::SharedPtr latest_mask_;

  int mask_threshold_;
  int pixel_stride_;
  int min_valid_points_;

  double marker_diameter_;

  bool consume_mask_after_fusion_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  const auto node =
    std::make_shared<MaskPointCloudFusion>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}