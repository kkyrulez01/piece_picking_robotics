cmake_minimum_required(VERSION 3.8)
project(unseen_sku_motion_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "srv/PlanPose.srv"
  "srv/PlanCandidate.srv"
  DEPENDENCIES
    geometry_msgs
    trajectory_msgs
)

# Package motion planner as a C++ library
add_library(motion_planner src/motion_planner.cpp)

target_include_directories(motion_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  motion_planner
  rclcpp
  geometry_msgs
  trajectory_msgs
  moveit_ros_planning_interface
)

install(
  TARGETS motion_planner
  EXPORT export_motion_planner
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_targets(
  export_motion_planner
  HAS_LIBRARY_TARGET
)

# Motion planner service node
add_executable(
  motion_planner_service_node
  src/motion_planner_service_node.cpp
)

target_include_directories(
  motion_planner_service_node
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ament_target_dependencies(
  motion_planner_service_node
  rclcpp
  geometry_msgs
  trajectory_msgs
  moveit_ros_planning_interface
)

target_link_libraries(
  motion_planner_service_node
  motion_planner
)

rosidl_get_typesupport_target(
  cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)

target_link_libraries(
  motion_planner_service_node
  "${cpp_typesupport_target}"
)

# Install C++ service node
install(
  TARGETS
  motion_planner_service_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install single-candidate motion planner node
add_executable(
  motion_planner_node
  src/motion_planner_node.cpp
)

ament_target_dependencies(
  motion_planner_node
  rclcpp
  geometry_msgs
  trajectory_msgs
  moveit_core
  moveit_ros_planning_interface
)

target_link_libraries(
    motion_planner_node
    motion_planner
    "${cpp_typesupport_target}"
)

install(
    TARGETS motion_planner_node
    DESTINATION lib/${PROJECT_NAME}
)

# Python executables
install(
  PROGRAMS
  # Python executables
  scripts/joint_convention_converter_node.py
  scripts/arc_trajectory_executor.py

  # Python helper modules
  scripts/joint_conversion.py

  DESTINATION
  lib/${PROJECT_NAME}
)

# Launch and config files
install(
  DIRECTORY
  launch config
  DESTINATION
  share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclcpp
  geometry_msgs
  trajectory_msgs
  moveit_ros_planning_interface
  rosidl_default_runtime
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