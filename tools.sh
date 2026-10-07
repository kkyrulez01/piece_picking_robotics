#!/bin/bash

export WORKSPACE="${unseen_sku_ws:-$HOME/unseen_sku_ws}"
export MOMA_WS="${MOMA_WS:-$HOME/MOMA_ws}"
export ARENA_ROS_WS="${ARENA_ROS_WS:-$HOME/arena_camera_ros2/ros2_ws}"
export ELITE_ROS_WS="${ELITE_ROS_WS:-$HOME/elite_ros_ws}" # Default to elite_ros_ws

export ARENA_ROOT="$HOME/ArenaSDK_v_1.0.10.11_Linux_x64/ArenaSDK_Linux_x64"
# Source workspace
source_ws() {
    source /opt/ros/humble/setup.bash
    source "$ARENA_ROS_WS/install/setup.bash"
    source "$MOMA_WS/install/setup.bash"
    source "$ELITE_ROS_WS/install/setup.bash"
    source "$WORKSPACE/install/setup.bash"

    echo 'Arena ROS,unseen_sku_ws, MOMA_WS and ELITE_ROS_WS sourced!'
}

# Activate SAM2 venv
activate_sam2() {
    source /opt/ros/humble/setup.bash
    source venv/sam2/bin/activate
    source install/setup.bash
    echo "sam2_venv activated!"
}

# Activate Sim-Suction venv
activate_sim_suction() {
    source venv/sim_suction_venv/bin/activate
    echo "sim_suction_venv activated!"
}

# Launch RealSense camera
launch_realsense() {
    ros2 launch realsense2_camera rs_launch.py \
        config_file:=/home/support/unseen_sku_ws/src/unseen_sku_camera/config/realsense.yaml
}

# Capture RGB-D image pair using RealSense
cap_realsense() {
    python3 $WORKSPACE/tools/realsense_capture.py
}

# Service to run SAM2 on image
ros2_run_SAM2() {
    ros2 service call /sam2/process_scene std_srvs/srv/Trigger "{}"
}

# ROS2 service to trigger helios2 capture
ros2_cap_helios2() {
    ros2 service call \
    /helios2/capture \
    std_srvs/srv/Trigger "{}"
}