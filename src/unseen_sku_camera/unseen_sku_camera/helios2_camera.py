#!/usr/bin/env python3

import ctypes
import threading
import time
from dataclasses import dataclass

import numpy as np

from arena_api.system import system


@dataclass
class Helios2Frame:
    """
    One captured Helios2 frame.

    xyz_m:
        Organized point cloud with shape (H, W, 3), in metres.
    intensity:
        Intensity image with shape (H, W), uint16.
    exposure:
        Active Helios2 exposure mode.
    width:
        Image width.
    height:
        Image height.
    """
    xyz_m: np.ndarray
    intensity: np.ndarray
    exposure: str
    width: int
    height: int

class Helios2Camera:
    """
    LUCID Helios2 camera wrapper using Arena SDK.

    Responsibilities:
        - Find and connect to the Helios2.
        - Configure Arena stream settings.
        - Configure Coord3D_ABCY16.
        - Read XYZ coordinate scale/offset.
        - Start and stop the camera stream.
        - Capture one organized XYZ + intensity frame.
        - Convert Helios raw coordinates to metres.
    """
    def __init__(
        self,
        serial=None,
        exposure="Exp250Us",
        warmup_seconds=0.5,
        log_callback=None,
        auto_connect=True,
    ):
        if serial is not None:
            self.serial = str(serial)
        else:
            self.serial = None

        self.exposure = str(exposure)
        self.warmup_seconds = float(warmup_seconds)

        self._log_callback = log_callback
        self._device = None

        self._xyz_scale_mm = None
        self._x_offset_mm = None
        self._y_offset_mm = None
        self._z_offset_mm = None

        # Prevent two captures from accessing Arena simultaneously.
        self._capture_lock = threading.Lock()

        if auto_connect:
            self.connect()

    # Logging
    def _log(self, message):
        if self._log_callback is not None:
            self._log_callback(str(message))

    # Properties
    @property
    def connected(self):
        return self._device is not None

    @property
    def activate_exposure(self):
        if self._device is None:
            return None

        return str(self._device.nodemap["ExposureTimeSelector"].value)

    # Connect camera
    def connect(self):
        """
        Connect to configured Helios2.
        """
        if self.connected:
            return

        device_infos = system.device_infos

        if not device_infos:
            raise RuntimeError("No LUCID camera detected by Arena SDK.")
    
        self._log("Detected LUCID cameras:")
    
        for index, info in enumerate(device_infos):
            self._log(
                f"  [{index}] "
                f"model={info.get('model')} "
                f"serial={info.get('serial')}"
            )

        selected_info = self._select_device_info(device_infos)

        self._log(
            "Connecting to "
            f"{selected_info.get('model')} "
            f"serial={selected_info.get('serial')}"
        )

        devices = system.create_device(device_infos=selected_info)

        if not devices:
            raise RuntimeError(
                "Arena SDK failed to create Helios2 device."
            )

        self._device = devices[0]
        try:
            self._configure_camera()
            self._log("Helios2 connected and ready for snapshot capture.")

        except Exception:
            self.close()
            raise

    def _select_device_info(self, device_infos):
        """
        Select requested camera serial.

        If no serial was provided, use the first LUCID camera.
        """
        if self.serial is None:
            return device_infos[0]

        for info in device_infos:
            if (str(info.get("serial"))) == self.serial:
                return info

        available_serials = [str(info.get("serial")) for info in device_infos]

        raise RuntimeError(
            f"Helios2 serial {self.serial} not found. "
            f"Available serials: {available_serials}"
        )

    # Configure Helios2 camera
    def _configure_camera(self):
        """
        Configure the Helios2 for XYZ + intensity output.
        """
        if self._device is None:
            raise RuntimeError("Helios2 is not connected.")

        nodemap = self._device.nodemap
        stream_nodemap = self._device.tl_stream_nodemap
        
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
        self._log(f"Current Helios2 exposure: {exposure_node.value}")
    
        # XYZ conversion scale
        self._xyz_scale_mm = float(nodemap["Scan3dCoordinateScale"].value)
    
        # X offset
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateA"
        self._x_offset_mm = float(nodemap["Scan3dCoordinateOffset"].value)
    
        # Y offset
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateB"
        self._y_offset_mm = float(nodemap["Scan3dCoordinateOffset"].value)
    
        # Z offset
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
        self._z_offset_mm = float(nodemap["Scan3dCoordinateOffset"].value)
    
        self._log("Helios XYZ conversion")
        self._log(f"Scale:    {self._xyz_scale_mm} mm")
        self._log(f"X offset: {self._x_offset_mm} mm")
        self._log(f"Y offset: {self._y_offset_mm} mm")
        self._log(f"Z offset: {self._z_offset_mm} mm")

    # Capture
    def capture(self):
        """
        Capture exactly one Helios2 point cloud.

        Acquisition starts for this capture and is stopped
        immediately afterwards.
        """
        if not self.connected:
            raise RuntimeError("Helios2 is not connected.")

        with self._capture_lock:
            buffer = None

            try:
                # Start acquisition only for this scene
                self._device.start_stream()
                self._stream_started = True

                # Optional short stabilization delay.
                if self.warmup_seconds > 0.0:
                    time.sleep(self.warmup_seconds)

                # Capture exactly one frame
                buffer = self._device.get_buffer()

                xyz_m, intensity = (self._buffer_to_xyz(buffer))

            finally:
                # Return Arena buffer
                if buffer is not None:
                    self._device.requeue_buffer(buffer)

                # Stop acquisition immediately
                if self._stream_started:
                    self._device.stop_stream()
                    self._stream_started = False

        height, width = xyz_m.shape[:2]

        return Helios2Frame(
            xyz_m=xyz_m,
            intensity=intensity,
            exposure=self.activate_exposure,
            width=width,
            height=height
        )

    def _buffer_to_xyz(self, buffer):
        """
        Convert Arena Coord3D_ABCY16 buffer into:
            xyz_m:
                (H, W, 3) float32 in metres.
            intensity:
                (H, W) uint16.

        Coord3D_ABCY16 channels:
            A = X
            B = Y
            C = Z
            Y = intensity
        """

        height = int(buffer.height)
        width = int(buffer.width)

        if int(buffer.bits_per_pixel) != 64:
            raise RuntimeError(
                "Expected Coord3D_ABCY16 to have "
                "64 bits/pixel, but received "
                f"{buffer.bits_per_pixel}."
            )

        number_of_values = height * width * 4

        # Arena exposes its image as raw memory.
        data_pointer = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16),)
        raw = np.ctypeslib.as_array(data_pointer, shape=(number_of_values,),)

        # Important:
        # Copy the data before the Arena buffer is requeued.
        abcy = raw.reshape(height, width, 4).copy()

        x_raw = abcy[:, :, 0]
        y_raw = abcy[:, :, 1]
        z_raw = abcy[:, :, 2]

        intensity = abcy[:, :, 3].copy()

        # Helios invalid coordinate marker.
        invalid = (
            (x_raw == 0xFFFF)
            | (y_raw == 0xFFFF)
            | (z_raw == 0xFFFF)
        )

        xyz_mm = np.empty((height, width, 3), dtype=np.float32,)

        xyz_mm[:, :, 0] = (x_raw.astype(np.float32) * self._xyz_scale_mm + self._x_offset_mm)
        xyz_mm[:, :, 1] = (y_raw.astype(np.float32) * self._xyz_scale_mm + self._y_offset_mm)
        xyz_mm[:, :, 2] = (z_raw.astype(np.float32) * self._xyz_scale_mm + self._z_offset_mm)

        # Preserve image organization.
        # Invalid points become NaN rather than being removed.
        xyz_mm[invalid] = np.nan

        # Standardize ROS/project coordinates on metres.
        xyz_m = xyz_mm * 0.001

        return xyz_m, intensity

    # Stop camera
    def close(self):
        """
        Stop the camera stream and destroy the Arena device.
        """
        if self._device is None:
            return

        try:
            if self._stream_started:
                self._device.stop_stream()
                self._stream_started = False

        finally:
            try:
                system.destroy_device(self._device)
            finally:
                self._device = None

        self._log("Helios2 disconnected.")

    # Context-manager support
    def __enter__(self):
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exception_type, exception_value, traceback,):
        self.close()
        return False