from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, AppendEnvironmentVariable
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, TextSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix

import os
import xacro

def generate_launch_description():

    package_name = 'arc_robot_pkg'
    pkg_path = os.path.join(get_package_share_directory(package_name))

    # Check if use_sim_time
    use_sim_time = LaunchConfiguration('use_sim_time')

    # For synthetic camera 
    enable_synthetic_camera = LaunchConfiguration("enable_synthetic_camera")
    
    # World file
    world_file_name = LaunchConfiguration("world_name")
    world_file_path = PathJoinSubstitution([pkg_path, "worlds", [world_file_name, TextSubstitution(text=".sdf")]])

    # Models
    models_path = os.path.join(pkg_path, 'models')

    # Option to launch Gazebo headless
    headless = LaunchConfiguration("headless")

    headless_gz_arguments = PythonExpression([
    "'-s --headless-rendering ' "
    "if '",
    headless,
    "'.lower() == 'true' "
    "else ''",
    ])

    # Launch world in Gazebo
    launch_world = IncludeLaunchDescription(
         PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
                launch_arguments={'gz_args': 
                                  ["-r ", headless_gz_arguments,world_file_path]}
                                  .items()
    )

    # Process URDF file
    xacro_file = os.path.join(pkg_path, 'ARC_S6', 'ARC.urdf.xacro')

    # Create Robot State Publisher node
    params = {'robot_description': Command(['xacro ', xacro_file, ' enable_synthetic_camera:=', enable_synthetic_camera]), 
              'use_sim_time': use_sim_time}
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters = [params],
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', '/robot_description',
                   '-name', 'ARC_robot',
                   '-x', '1',
                   '-y', '1.4', # 1.0
                   '-z', '1.025']
    )

    # Controller nodes
    joint_state_broadcaster_node = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    joint_trajectory_controller_node = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['ARC_S6_controller', '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ros_gz_bridge node
    bridge_config_path = os.path.join(pkg_path, 'config', 'ros_gz_bridge_config.yaml')
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=['--ros-args', '-p', f'config_file:={bridge_config_path}'],
    )
    
    # For plugins
    custom_plugin_path = os.path.join(get_package_prefix('arc_robot_pkg'), 'lib')
    gz_ros2_control_plugin_path = os.path.join(get_package_prefix('gz_ros2_control'), 'lib')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            "world_name",
            default_value="bin_picking_world",
            description="Gazebo world filename without .sdf extension"
        ),
        DeclareLaunchArgument(
            'enable_synthetic_camera',
            default_value="false",
            description="Enable the synthetic instance-segmentation camera",
        ),
        # For headless Gazebo
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            choices=["true", "false"],
            description="Choose to run Gazebo headless without GUI"
        ),
        # For models
        AppendEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=models_path + ':' + os.path.dirname(pkg_path)
        ),
        # For ROS2 Control and custom plugin
        AppendEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value=os.pathsep.join([
                gz_ros2_control_plugin_path,
                custom_plugin_path,
            ])
        ),
        AppendEnvironmentVariable(
            name='LD_LIBRARY_PATH',
            value=os.path.join(get_package_prefix('gz_ros2_control'), 'lib')
        ),
        launch_world,
        robot_state_publisher_node,
        spawn_robot,
        bridge_node,
        joint_state_broadcaster_node,
        joint_trajectory_controller_node,

    ])