#!/bin/bash

export WORKSPACE="${WORKSPACE:-$HOME/unseen_sku_ws}"
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
    source "$WORKSPACE/venv/sam2_venv/bin/activate"
    source "$WORKSPACE/install/setup.bash"
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
        config_file:="$WORKSPACE/src/unseen_sku_camera/config/realsense.yaml"
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


# ============================================================
# UNSEEN SKU - PERCEPTION COORDINATOR TOOLS
# ============================================================

# ------------------------------------------------------------
# 1. Start all perception nodes
# ------------------------------------------------------------
start_all_nodes() {
    (
        cd ~/unseen_sku_ws || return 1

        source /opt/ros/humble/setup.bash
        source install/setup.bash

        ros2 launch unseen_sku_bringup \
            coordinator_up_to_suction.launch.py
    )
}


# ------------------------------------------------------------
# 2. Process one scene
# ------------------------------------------------------------
process_scene() {
    (
        cd ~/unseen_sku_ws || return 1

        source /opt/ros/humble/setup.bash
        source install/setup.bash

        ros2 service call \
            /perception_coordinator/start_scene_test \
            std_srvs/srv/Trigger \
            '{}'
    )
}


# ------------------------------------------------------------
# 3. Check scene processing status
# ------------------------------------------------------------
scene_status() {
    (
        cd ~/unseen_sku_ws || return 1

        source /opt/ros/humble/setup.bash
        source install/setup.bash

        ros2 service call \
            /perception_coordinator/scene_test_status \
            std_srvs/srv/Trigger \
            '{}'
    )
}

# Connect to ARC S6 Robot
connect_ARC() {
    source /opt/ros/humble/setup.bash
    source $ELITE_ROS_WS/install/setup.bash

    local robot_ip="$1"
    local local_ip="$2"

    if [ -z "$robot_ip" ] || [ -z "$local_ip" ]; then
        echo "Usage: connect_ARC <robot_ip> <local_ip>"
        echo "Example: connect_ARC 192.168.1.200 192.168.1.100"
        return 1
    fi

    ros2 launch eli_cs_robot_driver elite_control.launch.py \
        robot_ip:=${robot_ip}\
        local_ip:=${local_ip}\
        headless_mode:=true \
        launch_rviz:=false \
        cs_type:=cs66
}

# Adjust ARC S6 robot speed
arc_speed() {
    source /opt/ros/humble/setup.bash
    source $ELITE_ROS_WS/install/setup.bash

    local speed="$1" # Stores the first positional argument

    # -z checks if empty, output usage instructions if empty
    if [ -z "$speed" ]; then 
        echo "Usage: arc_speed <fraction>"
        echo "Example: arc_speed 0.05"
        return 1
    fi

    # Speed must be between 0.0 and 1.0
    # ! awk: If condition is true (0), invert it to 1 to skip this error block
    if ! awk -v s="$speed" 'BEGIN { exit !(s >= 0.0 && s <= 1.0) }'; then
        echo "Error: speed must be between 0.0 and 1.0"
        echo "Example: arc_speed 0.05 # 5%"
        return 1
    fi

    ros2 service call \
        /io_and_status_controller/set_speed_slider \
        eli_common_interface/srv/SetSpeedSliderFraction \
        "{speed_slider_fraction: ${speed}}"
}