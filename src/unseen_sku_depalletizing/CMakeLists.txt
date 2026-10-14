cmake_minimum_required(VERSION 3.8)

cmake_policy(SET CMP0074 NEW)
project(moma_depalletizing)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(eli_common_interface REQUIRED)
find_package(std_msgs)
find_package(mecheye_ros_interface REQUIRED)

# Own motion planning package
find_package(moma_motion_planning REQUIRED)

# pointcloud_processor
add_executable(
  pointcloud_processor_node
  src/pointcloud_processor_node.cpp
)

ament_target_dependencies(
  pointcloud_processor_node

  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
)

target_include_directories(
  pointcloud_processor_node
  PRIVATE
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(
  pointcloud_processor_node
  ${PCL_LIBRARIES}
)

# alignment_checker_node
add_executable(
  alignment_checker_node
  src/alignment_checker_node.cpp
)

ament_target_dependencies(
  alignment_checker_node

  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  cv_bridge
)

target_include_directories(
  alignment_checker_node
  PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  alignment_checker_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# mask_pointcloud_fusion_node
add_executable(
  mask_pointcloud_fusion_node
  src/mask_pointcloud_fusion_node.cpp
)

ament_target_dependencies(
  mask_pointcloud_fusion_node

  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
)

target_include_directories(
  mask_pointcloud_fusion_node
  PRIVATE
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(
  mask_pointcloud_fusion_node
  ${PCL_LIBRARIES}
)

add_definitions(${PCL_DEFINITIONS})

# Install python package
ament_python_install_package(${PROJECT_NAME})

# Install Python ROS executable
install(
  PROGRAMS
    moma_depalletizing/depalletizing_coordinator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME depalletizing_coordinator
)

# Depalletizing coordinator 
# add_executable(
#   depalletizing_coordinator
#   src/depalletizing_coordinator.cpp
#   src/pick_phase.cpp
#   src/place_phase.cpp
#   src/complete_phase.cpp
# )

# target_include_directories(
#   depalletizing_coordinator
#   PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )

# ament_target_dependencies(
#   depalletizing_coordinator

#   rclcpp
#   geometry_msgs
#   sensor_msgs
#   trajectory_msgs
#   moveit_msgs
#   moma_motion_planning
#   eli_common_interface
#   std_msgs
# )

install(
  TARGETS 
  pointcloud_processor_node
  alignment_checker_node 
  mask_pointcloud_fusion_node
  # depalletizing_coordinator
  DESTINATION lib/${PROJECT_NAME}
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
