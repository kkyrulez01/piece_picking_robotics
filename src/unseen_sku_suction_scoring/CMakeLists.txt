cmake_minimum_required(VERSION 3.8)
project(unseen_sku_suction_scoring)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python Packages
ament_python_install_package(suction_scoring)
ament_python_install_package(geometric_filters)
ament_python_install_package(candidate_transform)

# Install ROS 2 executable
install(PROGRAMS
  suction_scoring/suction_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME suction_node
)

install(PROGRAMS
  candidate_transform/candidate_transform_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME candidate_transform_node
)

# Install config
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
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

ament_export_dependencies(rosidl_default_runtime)

ament_package()
