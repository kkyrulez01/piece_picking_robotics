from eli_common_interface.srv import SetIO


class VacuumController:
    """
    Controls the physical vacuum output on the Elite robot.

    This class is NOT a ROS node.
    It uses the ROS node supplied by DepalletizingCoordinator.

    It only handles:
        - vacuum ON
        - vacuum OFF
        - Elite SetIO service response
        - commanded vacuum state
    """

    def __init__(
        self,
        node,
        do_pin=0,
        service_name="/io_and_status_controller/set_io",
    ):
        self.node = node

        self.do_pin = do_pin
        self.service_name = service_name

        # This represents the commanded DO state.
        # It does NOT prove that a carton has been successfully grasped.
        self.enabled = False

        self.client = self.node.create_client(
            SetIO,
            self.service_name,
        )

        self.node.get_logger().info(
            f"VacuumController initialized: "
            f"DO{self.do_pin}, service='{self.service_name}'"
        )

    # ============================================================
    # Public interface
    # ============================================================

    def activate(self, callback=None):
        """
        Turn vacuum ON.

        callback signature:
            callback(success: bool, message: str)
        """

        self.node.get_logger().info(f"Requesting vacuum ON using DO{self.do_pin}.")

        self._set_vacuum(enabled=True, callback=callback)

    def release(self, callback=None):
        """
        Turn vacuum OFF.

        callback signature:
            callback(success: bool, message: str)
        """

        self.node.get_logger().info(f"Requesting vacuum OFF using DO{self.do_pin}.")

        self._set_vacuum(enabled=False, callback=callback)

    def is_enabled(self):
        """
        Return the last successfully commanded vacuum state.

        True does not mean that vacuum pressure has been physically verified.
        """

        return self.enabled

    # ============================================================
    # Internal SetIO handling
    # ============================================================

    def _set_vacuum(self, enabled, callback):

        if not self.client.service_is_ready():
            message = (
                f"Vacuum SetIO service is unavailable: "
                f"{self.service_name}"
            )

            self.node.get_logger().error(message)

            if callback is not None:
                callback(False, message)

            return

        request = SetIO.Request()

        # Standard digital output
        request.fun = SetIO.Request.FUN_SET_DIGITAL_OUT

        # Vacuum is currently connected to DO0
        request.pin = self.do_pin

        # Not used for normal digital output
        request.analog_type = 0

        # 1.0 = ON
        # 0.0 = OFF
        request.state = 1.0 if enabled else 0.0

        state_text = "ON" if enabled else "OFF"

        self.node.get_logger().warn(f"Elite DO{self.do_pin} -> {state_text}")

        future = self.client.call_async(request)

        future.add_done_callback(
            lambda future: self._set_io_response(
                future,
                requested_state=enabled,
                callback=callback,
            )
        )

    # ============================================================
    # Elite SetIO response
    # ============================================================

    def _set_io_response(
        self,
        future,
        requested_state,
        callback
    ):
        try:
            response = future.result()

        except Exception as exception:
            message = (
                f"Vacuum SetIO service call failed: "
                f"{exception}"
            )

            self.node.get_logger().error(message)

            if callback is not None:
                callback(False, message)

            return

        if response is None:
            message = "Vacuum SetIO returned no response."

            self.node.get_logger().error(message)

            if callback is not None:
                callback(False, message)

            return

        if not response.success:
            message = (
                "Vacuum SetIO request was rejected "
                "or returned success=False."
            )

            self.node.get_logger().error(message)

            if callback is not None:
                callback(False, message)

            return

        # Only update our stored state after Elite accepts the command.
        self.enabled = requested_state

        state_text = "ON" if requested_state else "OFF"

        message = f"Vacuum DO{self.do_pin} successfully commanded {state_text}."

        self.node.get_logger().info(message)

        if callback is not None:
            callback(True, message)
