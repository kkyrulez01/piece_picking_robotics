"""Main ROS 2 node for the Perception Coordinator."""

import rclpy
from rclpy.node import Node

from .scene_context import SceneContext
from .perception_manager import PerceptionManager
from std_srvs.srv import Trigger


# Names of parameters required by the coordinator.
# Their actual values are defined in YAML.
REQUIRED_PARAMETERS = (
    "realsense_rgbd_topic",
    "helios_capture_service",
    "helios_cloud_topic",

    "sam2_process_service",
    "sam2_masks_topic",

    "fused_objects_topic",

    "capture_timeout_sec",
    "camera_inputs_timeout_sec",
    "sam2_timeout_sec",
    "fusion_timeout_sec",
    "scoring_timeout_sec",

    "allow_concurrent_requests",
    "service_readiness_service",
)


class PerceptionCoordinator(Node):
    """ROS 2 entry point for scene-processing transactions."""

    def __init__(self):

        super().__init__(
            "perception_coordinator",
            automatically_declare_parameters_from_overrides=True,
        )

        # Verify that all required parameters were loaded.
        for name in REQUIRED_PARAMETERS:

            if not self.has_parameter(name):
                raise RuntimeError(
                    f"Missing required configuration parameter: {name}"
                )

        # Read parameters from ROS 2.
        self.settings = {
            name: self.get_parameter(name).value
            for name in REQUIRED_PARAMETERS
        }

        self._validate_configuration()

        # Only one scene can be processed at a time.
        self.active_scene: SceneContext | None = None

        # Register camera, segmentation, and fusion interfaces.
        self.perception_manager = PerceptionManager(
            node=self,
            settings=self.settings,
        )

        # Diagnostic service for checking required ROS 2 services.
        self.service_readiness_server = self.create_service(
            Trigger,
            self.settings["service_readiness_service"],
            self._check_services_callback,
        )

        self.get_logger().info(
            "Perception Coordinator initialized."
        )

        self.get_logger().info(
            "Configuration loaded successfully."
        )

    def _validate_configuration(self):
        """Validate coordinator configuration."""

        for name, value in self.settings.items():

            if name.endswith("_timeout_sec"):

                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value <= 0
                ):
                    raise ValueError(
                        f"{name} must be a positive number"
                    )

        if self.settings["allow_concurrent_requests"] is not False:
            raise ValueError(
                "Phase 1 requires allow_concurrent_requests: false"
            )

    def _check_services_callback(self, request, response):
        """Report Helios2 and SAM2 service availability."""

        del request

        availability = (
            self.perception_manager.get_service_availability()
        )

        response.success = all(availability.values())

        response.message = "; ".join(
            f"{name}: {'ready' if ready else 'unavailable'}"
            for name, ready in availability.items()
        )

        return response
    


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:
        node = PerceptionCoordinator()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()