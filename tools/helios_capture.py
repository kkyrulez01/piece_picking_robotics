#!/usr/bin/env python3

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np
import yaml

from arena_api.system import system

DEFAULT_OUTPUT_ROOT = Path("/home/support/unseen_sku_ws/data/helios2")
DEFAULT_CONFIG_FILE = Path("/home/support/unseen_sku_ws/config/helios2.yaml")

def load_config(config_path):
    config_path = Path(config_path).expanduser()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Helios2 config file not found: {config_path}"
        )

    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise ValueError("Helios2 configuration must be a YAML dictionary.")

    return config

def find_device(serial=None):
    """
    Find and create a LUCID camera.

    If serial is None, use the first detected camera.
    """
    device_infos = system.device_infos

    if not device_infos:
        raise RuntimeError("No LUCID camera detected by Arena SDK.")

    print("Detected LUCID cameras:")

    for index, info in enumerate(device_infos):
        print(
            f"  [{index}] "
            f"model={info.get('model')} "
            f"serial={info.get('serial')}"
        )

    if serial is None:
        selected_info = device_infos[0]

    else:
        selected_info = None

        for info in device_infos:
            if str(info.get("serial")) == str(serial):
                selected_info = info
                break

        if selected_info is None:
            raise RuntimeError(f"Camera serial {serial} not found.")

    print()
    print(
        "Connecting to:",
        selected_info.get("model"),
        selected_info.get("serial"),
    )

    devices = system.create_device(device_infos=selected_info)

    if not devices:
        raise RuntimeError("Arena SDK failed to create camera device.")

    return devices[0]

def inspect_nodes(device, keywords=None):
    nodemap = device.nodemap

    node_names = [
        "ExposureTimeSelector",
        "Scan3dOperatingMode",
        "ConversionGain",

        # Confidence filtering
        "Scan3dConfidenceThresholdEnable",
        "Scan3dConfidenceThresholdMin",

        # Spatial filtering
        "Scan3dSpatialFilterEnable",

        # Temporal accumulation
        "Scan3dImageAccumulation",

        # Flying pixel filtering - test these names
        "Scan3dFlyingPixelsRemovalEnable",
        "Scan3dFlyingPixelsDistanceThreshold",
        "Scan3dFlyingPixelsDistanceTreshold",

        # Coordinate conversion
        "ToFXyzConversionOffsetZ",

        "Scan3dCoordinateSelector",
        "Scan3dCoordinateScale",
        "Scan3dCoordinateOffset",

        "PixelFormat",
        "AcquisitionMode",
        "AcquisitionFrameRate",
    ]

    print()
    print("==============================")
    print("Helios2 relevant nodes")
    print("==============================")

    for name in node_names:
        try:
            node = nodemap[name]
        except Exception as error:
            print(
                f"{name:35s} "
                f"NOT AVAILABLE"
            )
            continue

        try:
            value = node.value
        except Exception:
            value = "<not readable>"

        print(
            f"{name:35s} "
            f"value={value}"
        )

    print()

def configure_helios(device, exposure=None):
    """
    Configure Helios2 for XYZ + intensity output.
    """
    nodemap = device.nodemap
    stream_nodemap = device.tl_stream_nodemap

    # Network/stream settings
    stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
    stream_nodemap["StreamPacketResendEnable"].value = True

    # Important when we wait for keyboard input:
    # return the newest frame rather than old buffered frames.
    stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"

    # Helios native XYZ + intensity format
    nodemap["PixelFormat"].value = "Coord3D_ABCY16"

    # Helios exposure
    exposure_node = nodemap["ExposureTimeSelector"]

    print()
    print("Helios exposure")
    print("---------------")
    print(f"Current exposure: {exposure_node.value}")

    if exposure is not None:
        exposure_node.value = exposure
        print(f"Requested exposure: {exposure}")

    print(f"Active exposure: {exposure_node.value}")
    print()

    # XYZ conversion scale
    xyz_scale_mm = float(nodemap["Scan3dCoordinateScale"].value)

    # X offset
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateA"
    x_offset_mm = float(nodemap["Scan3dCoordinateOffset"].value)

    # Y offset
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateB"
    y_offset_mm = float(nodemap["Scan3dCoordinateOffset"].value)

    # Z offset
    nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
    z_offset_mm = float(nodemap["Scan3dCoordinateOffset"].value)

    print()
    print("Helios XYZ conversion")
    print("---------------------")
    print(f"Scale:    {xyz_scale_mm} mm")
    print(f"X offset: {x_offset_mm} mm")
    print(f"Y offset: {y_offset_mm} mm")
    print(f"Z offset: {z_offset_mm} mm")
    print()

    return (
        xyz_scale_mm,
        x_offset_mm,
        y_offset_mm,
        z_offset_mm,
    )

def buffer_to_xyz(
    buffer,
    xyz_scale_mm,
    x_offset_mm,
    y_offset_mm,
    z_offset_mm,
):
    """
    Convert a Coord3D_ABCY16 Arena buffer into:

        xyz:       (H, W, 3) float32, metres
        intensity: (H, W) uint16

    Coord3D_ABCY16 layout for each pixel:

        A = X
        B = Y
        C = Z
        Y = intensity

    Each channel is uint16.
    """

    height = int(buffer.height)
    width = int(buffer.width)

    if int(buffer.bits_per_pixel) != 64:
        raise RuntimeError(
            "Expected Coord3D_ABCY16 to have "
            f"64 bits/pixel, got {buffer.bits_per_pixel}"
        )

    number_of_uint16_values = (height * width * 4)

    # Arena buffer is raw memory.
    # Interpret it as uint16 values.
    data_pointer = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16),)
    raw = np.ctypeslib.as_array(data_pointer, shape=(number_of_uint16_values,))

    # Copy before Arena requeues the buffer.
    abcy = raw.reshape(height, width, 4,).copy()

    x_raw = abcy[:, :, 0]
    y_raw = abcy[:, :, 1]
    z_raw = abcy[:, :, 2]

    intensity = abcy[:, :, 3].copy()

    # Helios invalid coordinate marker
    invalid = (
        (x_raw == 0xFFFF)
        | (y_raw == 0xFFFF)
        | (z_raw == 0xFFFF)
    )

    xyz_mm = np.empty((height, width, 3), dtype=np.float32,)

    xyz_mm[:, :, 0] = x_raw.astype(np.float32) * xyz_scale_mm + x_offset_mm
    xyz_mm[:, :, 1] = y_raw.astype(np.float32) * xyz_scale_mm + y_offset_mm
    xyz_mm[:, :, 2] = z_raw.astype(np.float32) * xyz_scale_mm+ z_offset_mm

    # Use NaN for invalid XYZ.
    xyz_mm[invalid] = np.nan

    # Standardize the rest of the project on metres.
    xyz_m = xyz_mm * 0.001

    return xyz_m, intensity

def find_next_index(output_directory, prefix):
        """
        Find next available index for selected prefix.

        Example: 
        scene_0001.npz 
        scene_0002.npz 
        
        or: 
        background_0001.npz 
        background_0002.npz
        """
        pattern = f"{prefix}_*.npz"
        highest_index = 0

        for path in output_directory.glob(pattern):
            try:
                index = int(path.stem.rsplit("_", 1)[1])

                highest_index = max(highest_index, index)
            except (IndexError, ValueError):
                continue

        return highest_index + 1

def print_statistics(xyz):
        """
        Print simple point-cloud statistics.
        """
        xyz_flat = xyz.reshape(-1, 3)

        valid = np.isfinite(xyz_flat).all(axis=1) 
        valid_xyz = xyz_flat[valid]

        print() 
        print("Point cloud statistics") 
        print("----------------------") 
        print(f"Shape: {xyz.shape}") 
        print(f"Total points: {len(xyz_flat)}") 
        print(f"Valid points: {len(valid_xyz)}")

        if len(valid_xyz) > 0: 
            print("XYZ minimum:", np.min(valid_xyz, axis=0)) 
            print("XYZ maximum:", np.max(valid_xyz, axis=0)) 
            print("XYZ median:", np.median(valid_xyz, axis=0)) 
            print()

def capture_frame(
    device,
    xyz_parameters,
):
    """
    Grab one current Helios frame and convert it to XYZ.
    """
    buffer = None

    try:
        buffer = device.get_buffer()
        xyz, intensity = buffer_to_xyz(buffer,*xyz_parameters,)

    finally:
        if buffer is not None:
            device.requeue_buffer(buffer)

    return xyz, intensity

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Capture Helios2 XYZ point clouds "
            "directly using Arena SDK."
        )
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
        help="Helios2 YAML config file."
    )
    parser.add_argument(
        "--prefix",
        choices=("scene", "background"),
        default="scene",
        help="Capture type. Default: scene",
    )

    parser.add_argument(
        "--serial",
        default=None,
        help=(
            "Optional Helios serial number. "
            "If omitted, first detected LUCID camera is used."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Override outputroot directory from YAML.",
    )

    parser.add_argument(
        "--exposure",
        choices=(
            "Exp62_5Us",
            "Exp250Us",
            "Exp1000Us",
        ),
        default=None,
        help="Helios exposure time.",
    )
    
    args = parser.parse_args()
    config = load_config(args.config)

    helios_config = config.get("helios2", {})
    serial_config = helios_config.get("serial", None)
    output_config = helios_config.get("output_root", str(DEFAULT_OUTPUT_ROOT))
    exposure_config = helios_config.get("exposure", "Exp250Us")

    # Overriding YAML
    if args.serial is not None:
        serial = args.serial
    else:
        serial = serial_config

    if args.output is not None:
        output_root = args.output
    else:
        output_root = output_config

    if args.exposure is not None:
        exposure = args.exposure
    else:
        exposure = exposure_config

    output_directory = Path(output_root).expanduser() / args.prefix
    output_directory.mkdir(parents=True, exist_ok=True,)

    device = None
    stream_started = False

    try:
        device = find_device(serial=serial)

        inspect_nodes(
            device,
            keywords=[
                "exposure",
                "gain",
                "confidence",
                "filter",
                "flying",
                "spatial",
                "scan3d",
                "accum",
            ],
        )
        
        xyz_parameters = configure_helios(device, exposure=exposure)
        print("Starting Helios stream...")

        device.start_stream()
        stream_started = True

        # Allow the camera to stabilize briefly.
        time.sleep(0.5)

        print()
        print("==============================")
        print("Helios2 Point Cloud Capture")
        print("==============================")
        print()
        print(f"Prefix: {args.prefix}")
        print(f"Output: {output_directory}")
        print()
        print("Press ENTER to capture.")
        print("Type q + ENTER to quit.")
        print()

        while True:

            command = input()
            if command.strip().lower() in {
                "q",
                "quit",
                "exit",
            }:
                break

            xyz, intensity = capture_frame(device, xyz_parameters,)

            index = find_next_index(output_directory, args.prefix)
            output_path = output_directory / f"{args.prefix}_{index:04d}.npz"


            np.savez(
                output_path,
                xyz=xyz,
                intensity=intensity,
                xyz_unit=np.array("m"),
                width=np.array(xyz.shape[1]),
                height=np.array(xyz.shape[0]),
                exposure=np.array(device.nodemap["ExposureTimeSelector"].value),
            )

            print_statistics(xyz)
            print(f"Saved: {output_path}")

    except KeyboardInterrupt:
        print("\nStopping capture...")

    finally:
        if device is not None:
            if stream_started:
                device.stop_stream()

            system.destroy_device(device)

        print("Helios2 disconnected.")

if __name__ == "__main__":
    main()
