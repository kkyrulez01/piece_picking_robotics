#!/usr/bin/env bash

# Usage:
#   ./show_realsense_settings.sh
#   ./show_realsense_settings.sh /camera/camera

NODE="${1:-/camera/camera}"

echo "============================================================"
echo " RealSense ROS 2 Settings"
echo " Node: ${NODE}"
echo "============================================================"
echo

# Check that the node exists
if ! ros2 node list | grep -Fxq "${NODE}"; then
    echo "ERROR: ROS 2 node '${NODE}' was not found."
    echo
    echo "Available nodes:"
    ros2 node list
    exit 1
fi


print_param()
{
    local param="$1"
    local label="$2"

    if ros2 param list "${NODE}" 2>/dev/null | sed 's/^[[:space:]]*//' | grep -Fxq "${param}"; then
        local value
        value=$(ros2 param get "${NODE}" "${param}" 2>/dev/null)

        printf "%-42s : %s\n" "${label}" "${value}"
    else
        printf "%-42s : %s\n" "${label}" "NOT AVAILABLE"
    fi
}

show_params () 
{
    echo "---------------------- DEPTH CAMERA -------------------------"

    print_param "depth_module.profile" \
                "Depth profile"

    print_param "depth_module.depth_profile" \
                "Depth profile (alternative name)"

    print_param "depth_module.enable_auto_exposure" \
                "Depth auto exposure"

    print_param "depth_module.exposure" \
                "Depth exposure"

    print_param "depth_module.gain" \
                "Depth gain"

    print_param "depth_module.visual_preset" \
                "Depth visual preset"

    print_param "depth_module.emitter_enabled" \
                "IR emitter"

    print_param "depth_module.emitter_on_off" \
                "Emitter alternate mode"

    print_param "depth_module.laser_power" \
                "Laser power"

    echo


    echo "----------------------- RGB CAMERA --------------------------"

    print_param "rgb_camera.profile" \
                "RGB profile"

    print_param "rgb_camera.color_profile" \
                "RGB profile (alternative name)"

    print_param "rgb_camera.enable_auto_exposure" \
                "RGB auto exposure"

    print_param "rgb_camera.exposure" \
                "RGB exposure"

    print_param "rgb_camera.gain" \
                "RGB gain"

    print_param "rgb_camera.enable_auto_white_balance" \
                "RGB auto white balance"

    print_param "rgb_camera.white_balance" \
                "RGB white balance"

    print_param "rgb_camera.brightness" \
                "RGB brightness"

    print_param "rgb_camera.contrast" \
                "RGB contrast"

    print_param "rgb_camera.saturation" \
                "RGB saturation"

    print_param "rgb_camera.sharpness" \
                "RGB sharpness"

    echo


    echo "---------------------- SPATIAL FILTER -----------------------"

    print_param "spatial_filter.enable" \
                "Spatial filter"

    print_param "spatial_filter.filter_magnitude" \
                "Spatial magnitude"

    print_param "spatial_filter.filter_smooth_alpha" \
                "Spatial smooth alpha"

    print_param "spatial_filter.filter_smooth_delta" \
                "Spatial smooth delta"

    print_param "spatial_filter.holes_fill" \
                "Spatial hole fill"

    echo

    echo "--------------------- TEMPORAL FILTER -----------------------"

    print_param "temporal_filter.enable" \
                "Temporal filter"

    print_param "temporal_filter.filter_smooth_alpha" \
                "Temporal smooth alpha"

    print_param "temporal_filter.filter_smooth_delta" \
                "Temporal smooth delta"

    print_param "temporal_filter.holes_fill" \
                "Temporal persistence / hole fill"

    echo

    echo "---------------------- HOLE FILLING -------------------------"

    print_param "hole_filling_filter.enable" \
                "Hole filling filter"

    print_param "hole_filling_filter.holes_fill" \
                "Hole filling mode"

    echo

    echo "------------------------ HDR --------------------------------"

    print_param "depth_module.hdr_enabled" \
                "Depth HDR"

    print_param "hdr_merge.enable" \
                "HDR merge filter"

    print_param "depth_module.sequence_id" \
                "HDR sequence ID"

    print_param "depth_module.sequence_size" \
                "HDR sequence size"

    echo

    echo "---------------------- ALIGNMENT ----------------------------"

    print_param "align_depth.enable" \
                "Align depth"

    print_param "pointcloud.enable" \
                "Point cloud"

    echo

    echo "--------------------- OTHER FILTERS -------------------------"

    print_param "decimation_filter.enable" \
                "Decimation filter"

    print_param "decimation_filter.filter_magnitude" \
                "Decimation magnitude"

    print_param "threshold_filter.enable" \
                "Threshold filter"

    print_param "threshold_filter.min_dist" \
                "Minimum depth"

    print_param "threshold_filter.max_dist" \
                "Maximum depth"

    echo

    echo "============================================================"
    echo " Matching RealSense parameters available on this installation"
    echo "============================================================"

    ros2 param list "${NODE}" 2>/dev/null \
        | sed 's/^[[:space:]]*//' \
        | grep -Ei \
        "exposure|gain|laser|emitter|preset|spatial|temporal|hole|hdr|profile|white_balance|brightness|contrast|saturation|sharpness|decimation|threshold|align"


    echo
    echo "============================================================"
    echo " Done"
    echo "============================================================"
}

show_params_ranges () 
{
    echo "============================================================"
    echo " Important parameter ranges / descriptions"
    echo "============================================================"

    for param in \
        depth_module.exposure \
        depth_module.gain \
        depth_module.laser_power \
        depth_module.visual_preset \
        spatial_filter.filter_magnitude \
        spatial_filter.filter_smooth_alpha \
        spatial_filter.filter_smooth_delta \
        temporal_filter.filter_smooth_alpha \
        temporal_filter.filter_smooth_delta
    do
        if ros2 param list "${NODE}" 2>/dev/null \
            | sed 's/^[[:space:]]*//' \
            | grep -Fxq "${param}"; then

            echo
            echo ">>> ${param}"
            ros2 param describe "${NODE}" "${param}"
        fi
    done
}