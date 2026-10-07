from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    helios_node = Node(
        package="unseen_sku_camera",
        executable="helios2_node",
        name="helios2_node",
        output="screen",
    )

    camera_tf_node = Node(
        package="unseen_sku_camera",
        executable="camera_tf_publisher",
        name="camera_tf_publisher",
        output="screen",
    )

    sam2_node = Node(
        package="unseen_sku_perception",
        executable="sam2_node",
        name="sam2_node",
        output="screen",
    )

    return LaunchDescription([
        helios_node,
        camera_tf_node,
        sam2_node,
    ])