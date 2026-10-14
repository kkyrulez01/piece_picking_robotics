#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class AlignmentChecker : public rclcpp::Node
{
  public:
    AlignmentChecker() : Node("alignment_checker"){
      depth_jump_threshold_ = declare_parameter<double>("depth_jump_threshold", 0.02);
      
      colour_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        "/mechmind/color_image",
        10,
        std::bind(&AlignmentChecker::colour_callback, this, std::placeholders::_1)
      );

      cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/mechmind/point_cloud",
        10,
        std::bind(&AlignmentChecker::cloud_callback, this, std::placeholders::_1)
      );

      depth_img_publisher_ = create_publisher<sensor_msgs::msg::Image>(
        "/moma/depth_image",
        10
      );

      overlay_publisher_ = create_publisher<sensor_msgs::msg::Image>(
        "/moma/alignment_overlay",
        10
      );

      RCLCPP_INFO(
        get_logger(),
        "Alignment checker started");

      RCLCPP_INFO(
        get_logger(),
        "Depth-edge threshold: %.3f m",
        depth_jump_threshold_);
    }
  
  private:
    void colour_callback(const sensor_msgs::msg::Image::SharedPtr message){
      try {
        const cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(message, sensor_msgs::image_encodings::BGR8);

        latest_colour_image_ = cv_image->image.clone();
        latest_colour_header_ = message->header;

        RCLCPP_INFO(
          get_logger(),
          "Received colour image: %dx%d, encoding=%s",
          latest_colour_image_.cols,
          latest_colour_image_.rows,
          message->encoding.c_str());
      } catch (const cv_bridge::Exception & error) {
        RCLCPP_ERROR(
        get_logger(),
        "Failed to convert colour image: %s",
        error.what());
      }
    }
    
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr message){
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

        RCLCPP_INFO(
          get_logger(),
          "Received organized cloud: %ux%u",
          cloud.width,
          cloud.height);
        
        const cv::Mat depth_metres = create_depth_image(cloud);

        const cv::Mat depth_visual = create_depth_visualization(depth_metres);

        publish_mono_image(depth_visual, message->header, depth_img_publisher_);

        if (latest_colour_image_.empty()) {
          RCLCPP_WARN(
            get_logger(),
            "No colour image has been received yet");
          return;
        }

        if (
          latest_colour_image_.cols != depth_metres.cols ||
          latest_colour_image_.rows != depth_metres.rows)
        {
          RCLCPP_ERROR(
            get_logger(),
            "Image/cloud dimension mismatch: "
            "colour=%dx%d cloud=%dx%d",
            latest_colour_image_.cols,
            latest_colour_image_.rows,
            depth_metres.cols,
            depth_metres.rows);
          return;
        }

        const cv::Mat overlay =create_alignment_overlay(latest_colour_image_, depth_metres);

        publish_bgr_image(overlay, message->header, overlay_publisher_);

        RCLCPP_INFO(
          get_logger(),
          "Published /moma/depth_image and "
          "/moma/alignment_overlay");
    }

    static cv::Mat create_depth_image(const pcl::PointCloud<pcl::PointXYZ> & cloud){
      cv::Mat depth(
        static_cast<int>(cloud.height),
        static_cast<int>(cloud.width),
        CV_32FC1,
        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

      for (std::uint32_t v = 0; v < cloud.height; ++v) {
        for (std::uint32_t u = 0; u < cloud.width; ++u) {
          const pcl::PointXYZ & point = cloud.at(u, v);

          if (
            std::isfinite(point.x) &&
            std::isfinite(point.y) &&
            std::isfinite(point.z) &&
            point.z > 0.0F)
          {depth.at<float>(static_cast<int>(v), static_cast<int>(u)) = point.z;}
        }
      }

      return depth;
    }

    static cv::Mat create_valid_mask(const cv::Mat & depth){
      cv::Mat valid_mask(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0));

      for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
          const float value = depth.at<float>(v, u);

          if (std::isfinite(value) && value >0.0F){
            valid_mask.at<std::uint8_t>(v, u) = 255;
          }
        }
      }

      return valid_mask;
    }

    static cv::Mat create_depth_visualization(const cv::Mat & depth){
      const cv::Mat valid_mask = create_valid_mask(depth);

      double minimum_depth = 0.0;
      double maximum_depth = 0.0;

      cv::minMaxLoc(
        depth,
        &minimum_depth,
        &maximum_depth,
        nullptr,
        nullptr,
        valid_mask);

      cv::Mat depth_visual(
        depth.rows,
        depth.cols,
        CV_8UC1,
        cv::Scalar(0));

      if (maximum_depth <= minimum_depth) {
        return depth_visual;
      }

      for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
          const float depth_value =
            depth.at<float>(v, u);

          if (!std::isfinite(depth_value) || depth_value <= 0.0F){
            continue;
          }

          const double normalized =
            (depth_value - minimum_depth) /
            (maximum_depth - minimum_depth);

          // Near points are bright; far points are dark.
          const double brightness = 255.0 * (1.0 - normalized);

          depth_visual.at<std::uint8_t>(v, u) =
            cv::saturate_cast<std::uint8_t>(brightness);
        }
      }

      return depth_visual;
    }

    cv::Mat create_depth_edges(const cv::Mat & depth) const
    {
      cv::Mat edges(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0));

      const float threshold =
        static_cast<float>(depth_jump_threshold_);

      for (int v = 0; v < depth.rows - 1; ++v) {
        for (int u = 0; u < depth.cols - 1; ++u) {
          const float centre =
            depth.at<float>(v, u);

          const float right =
            depth.at<float>(v, u + 1);

          const float below =
            depth.at<float>(v + 1, u);

          if (!std::isfinite(centre)) {
            continue;
          }

          bool is_edge = false;

          if (std::isfinite(right) && std::abs(centre - right) > threshold){
            is_edge = true;
          }

          if (std::isfinite(below) && std::abs(centre - below) > threshold){
            is_edge = true;
          }

          // Treat valid-to-invalid transitions as boundaries.
          if (!std::isfinite(right) || !std::isfinite(below)){
            is_edge = true;
          }

          if (is_edge) {
            edges.at<std::uint8_t>(v, u) = 255;
          }
        }
      }
      const cv::Mat kernel =
      cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(3, 3));

      cv::dilate(edges, edges, kernel);

      return edges;
    }

    cv::Mat create_alignment_overlay(
      const cv::Mat & colour_image,
      const cv::Mat & depth) const
    {
      cv::Mat overlay =
        colour_image.clone();

      const cv::Mat depth_edges =
        create_depth_edges(depth);

      for (int v = 0; v < overlay.rows; ++v) {
        for (int u = 0; u < overlay.cols; ++u) {
          if (
            depth_edges.at<std::uint8_t>(v, u) > 0)
          {
            // Draw cloud-derived depth edges in red.
            overlay.at<cv::Vec3b>(v, u) =
              cv::Vec3b(0, 0, 255);
          }
        }
      }

      return overlay;
    }

    static void publish_mono_image(
      const cv::Mat & image,
      const std_msgs::msg::Header & header,
      const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr & publisher){
        const sensor_msgs::msg::Image::SharedPtr message =
          cv_bridge::CvImage(
            header,
            sensor_msgs::image_encodings::MONO8,
            image).toImageMsg();

        publisher->publish(*message);
    }

    static void publish_bgr_image(
      const cv::Mat & image,
      const std_msgs::msg::Header & header,
      const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr & publisher){
        const sensor_msgs::msg::Image::SharedPtr message =
          cv_bridge::CvImage(
            header,
            sensor_msgs::image_encodings::BGR8,
            image).toImageMsg();

        publisher->publish(*message);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr colour_subscription_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscription_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_img_publisher_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_publisher_;

    cv::Mat latest_colour_image_;
    std_msgs::msg::Header latest_colour_header_;

    double depth_jump_threshold_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<AlignmentChecker>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
