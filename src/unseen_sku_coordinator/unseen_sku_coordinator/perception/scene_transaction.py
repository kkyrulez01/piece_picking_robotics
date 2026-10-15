"""Coordinate camera, SAM2, fusion, and suction without blocking ROS."""

from time import monotonic

from ..common.coordinator_states import CoordinatorState, FailureCode, ResultStatus
from .scene_context import SceneContext
from .observation_validator import validate_observation_pair
from .fusion_result_validator import validate_fusion_output
from .suction_result_validator import validate_suction_result


class SceneTransaction:
    """Timer-driven workflow supervisor with separate result validators."""

    def __init__(self, manager, settings, logger):
        self.manager = manager
        self.settings = settings
        self.logger = logger
        self.scene = None
        self.restart_required = False
        self._capture_future = None
        self._sam2_future = None
        self._capture_ack = False
        self._sam2_ack = False
        self._capture_sent_at = None
        self._sam2_sent_at = None
        self._cloud_before = None
        self._mask_index_before = None

        # Observations selected for this transaction.
        self._selected_cloud = None
        self._selected_masks = None

        # Fusion request timing.
        self._fusion_sent_at = None

        # Suction scoring request timing.
        self._suction_sent_at = None

        # IDs of the fused objects approved for scoring.
        self._approved_suction_object_ids = None

        # Retain the validated scoring result after SUCCESS.
        self._validated_suction_result = None

    def begin(self):
        """Accept one manual test request, or give an immediate rejection."""
        if self.restart_required:
            return False, "RESTART_REQUIRED: a previous request timed out"
        if self.scene is not None and not self.scene.is_finished:
            return False, f"BUSY: scene_id={self.scene.scene_id}"

        availability = self.manager.get_service_availability()
        unavailable = [name for name, ready in availability.items() if not ready]
        if unavailable:
            return False, "UNAVAILABLE: " + ", ".join(unavailable)

        self.scene = SceneContext()
        self._capture_future = None
        self._sam2_future = None
        self._capture_ack = False
        self._sam2_ack = False
        self._capture_sent_at = None
        self._sam2_sent_at = None
        self._cloud_before = None
        self._mask_index_before = None
        # Do not reuse observations selected for a previous scene.
        self._selected_cloud = None
        self._selected_masks = None
        self._fusion_sent_at = None
        self._suction_sent_at = None
        self._approved_suction_object_ids = None
        self._validated_suction_result = None
        self.scene.enter_state(CoordinatorState.WAITING_FOR_CAMERA_INPUTS)
        self.logger.info(f"M1 test scene accepted: {self.scene.scene_id}")
        return True, f"ACCEPTED: scene_id={self.scene.scene_id}; M1 test only"

    def describe(self):
        """Report the current transaction and its observation timestamps."""

        if self.scene is None:
            return False, "No test scene has been requested"

        scene = self.scene

        result = (
            scene.result_status.value
            if scene.result_status
            else "running"
        )

        # Format the timestamps of all recorded observations.
        observation_stamps = {
            source: (
                f"{stamp.sec}.{stamp.nanosec:09d} "
                f"[{stamp.frame_id}]"
            )
            for source, stamp in scene.observations.items()
        }

        return True, (
            f"scene_id={scene.scene_id}; "
            f"state={scene.state.value}; "
            f"result={result}; "
            f"sam2_scene_index={scene.sam2_scene_index}; "
            f"selected_pair_ready={self.get_selected_pair() is not None}; "
            f"observations={list(scene.observations)}; "
            f"observation_stamps={observation_stamps}; "
            f"failure={scene.failure_code.value}; "
            f"detail={scene.failure_detail}"
        )

    # ==============================================
    # Selected Observation Pair
    # ==============================================

    def get_selected_pair(self):
        """Return selected observations after a successful M1 transaction."""

        if self.scene is None:
            return None

        if self.scene.result_status != ResultStatus.SUCCESS:
            return None

        cloud = self._selected_cloud
        masks = self._selected_masks

        # Do not expose an incomplete pair.
        if cloud is None or masks is None:
            return None

        # Both full ROS messages must be available.
        if cloud.payload is None or masks.payload is None:
            return None

        return cloud, masks

    # ==============================================
    # Validated Suction Result
    # ==============================================

    def get_validated_suction_result(self):
        """
        Return the validated SuctionResult from a
        successfully completed scoring transaction.

        Returns None for failed, incomplete, or
        fusion-only transactions.
        """

        if self.scene is None:
            return None

        if self.scene.result_status != ResultStatus.SUCCESS:
            return None

        return self._validated_suction_result

    def release_selected_pair(self):
        """Release selected messages after downstream consumption."""

        if self.scene is not None and not self.scene.is_finished:
            raise RuntimeError(
                "Cannot release selected pair during an active transaction"
            )

        self._selected_cloud = None
        self._selected_masks = None

    def _finish(self, status, failure, detail):

        # Release the expected FusionResult, whether this
        # transaction succeeded, failed, or timed out.
        if (
            self.scene is not None
            and self.settings.get("enable_mask_fusion", False)
        ):
            self.manager.clear_expected_fusion_result(
                self.scene.scene_id
            )

        # Release the expected suction result.
        if (
            self.scene is not None
            and self.settings.get("enable_suction_scoring", False)
        ):
            self.manager.clear_expected_suction_result(
                self.scene.scene_id
            )

        # Preserve the existing M2.1 behavior:
        # only successful transactions retain their
        # selected Helios2 and SAM2 messages.
        if status != ResultStatus.SUCCESS:

            self._selected_cloud = None
            self._selected_masks = None

            # Do not expose an invalid scoring result.
            self._validated_suction_result = None

        # Approved object IDs are needed only during scoring.
        self._approved_suction_object_ids = None

        self.scene.finish(status, failure, detail)

    def tick(self):
        """Called periodically by the ROS executor; never blocks."""
        if self.scene is None or self.scene.is_finished:
            return

        try:
            state = self.scene.state
            if state == CoordinatorState.WAITING_FOR_CAMERA_INPUTS:
                self._wait_for_realsense()
            elif state == CoordinatorState.REQUESTING_CAPTURE:
                self._wait_for_helios()
            elif state == CoordinatorState.WAITING_FOR_SAM2_MASKS:
                self._wait_for_sam2()
            elif state == CoordinatorState.WAITING_FOR_MASK_FUSION:
                self._wait_for_fusion()
            elif state == CoordinatorState.WAITING_FOR_SUCTION_CANDIDATES:
                self._wait_for_suction()
        except Exception as error:
            self._finish(
                ResultStatus.FAILED, FailureCode.INTERNAL_ERROR, str(error)
            )

    def _wait_for_realsense(self):
        obs = self.manager.get_latest_observations()
        rgbd = obs.get("realsense")
        if rgbd is not None and (
            rgbd.received_monotonic >= self.scene.started_monotonic
        ):
            cloud = obs.get("helios2")
            self._cloud_before = cloud.stamp if cloud is not None else None
            self._capture_sent_at = monotonic()
            try:
                self._capture_future = self.manager.request_helios_capture()
            except RuntimeError as error:
                self._finish(
                    ResultStatus.FAILED, FailureCode.CAMERA_UNAVAILABLE,
                    str(error),
                )
                return
            self.scene.enter_state(CoordinatorState.REQUESTING_CAPTURE)
            return

        if (monotonic() - self.scene.stage_started_monotonic
                >= self.settings["camera_inputs_timeout_sec"]):
            self._finish(
                ResultStatus.TIMEOUT, FailureCode.RGBD_TIMEOUT,
                "No new RealSense observation received after request",
            )

    def _wait_for_helios(self):
        if self._capture_future.done() and not self._capture_ack:
            reply = self._capture_future.result()
            if not reply.success:
                self._finish(
                    ResultStatus.FAILED, FailureCode.CAMERA_UNAVAILABLE,
                    f"Helios2 capture failed: {reply.message}",
                )
                return
            self._capture_ack = True

        cloud = self.manager.get_latest_observations().get("helios2")
        if (
            self._capture_ack
            and cloud is not None
            and cloud.received_monotonic >= self._capture_sent_at
            and cloud.stamp != self._cloud_before
        ):
            # Preserve the exact observation accepted by this transaction.
            self._selected_cloud = cloud

            self.scene.record_observation("helios2", cloud.stamp)
            old_masks = self.manager.get_latest_observations().get("sam2")
            self._mask_index_before = (
                old_masks.scene_index if old_masks is not None else None
            )
            self._sam2_sent_at = monotonic()
            try:
                self._sam2_future = self.manager.request_sam2_processing()
            except RuntimeError as error:
                self._finish(
                    ResultStatus.FAILED, FailureCode.SAM2_UNAVAILABLE,
                    str(error),
                )
                return
            self.scene.enter_state(CoordinatorState.WAITING_FOR_SAM2_MASKS)
            return

        if (monotonic() - self._capture_sent_at
                >= self.settings["capture_timeout_sec"]):
            self.restart_required = True
            self._finish(
                ResultStatus.TIMEOUT, FailureCode.CAPTURE_TIMEOUT,
                "Capture response or new cloud missing; restart before retry",
            )

    def _wait_for_sam2(self):
        if self._sam2_future.done() and not self._sam2_ack:
            reply = self._sam2_future.result()
            if not reply.success:
                self._finish(
                    ResultStatus.FAILED, FailureCode.SAM2_FAILED,
                    f"SAM2 processing failed: {reply.message}",
                )
                return
            self._sam2_ack = True

        masks = self.manager.get_latest_observations().get("sam2")
        fresh_masks = (
            masks is not None
            and masks.received_monotonic >= self._sam2_sent_at
            and masks.scene_index is not None
            and masks.scene_index != self._mask_index_before
        )

        if self._sam2_ack and fresh_masks:
            rgbd = self.manager.find_realsense_frame_for_sam2(
                mask_stamp=masks.stamp,
                scene_started_monotonic=self.scene.started_monotonic,
            )
            if rgbd is not None:

                # ------------------------------------------
                # M2.1B — Preserve selected SAM2 observation
                # ------------------------------------------

                self._selected_masks = masks

                # Both observations must be available.
                if self._selected_cloud is None:
                    self._finish(
                        ResultStatus.FAILED,
                        FailureCode.INVALID_RESULT,
                        "Missing selected Helios2 cloud",
                    )
                    return

                # ------------------------------------------
                # M2.2B — Validate selected observation pair
                # ------------------------------------------

                validation = validate_observation_pair(
                    cloud=self._selected_cloud,
                    masks=self._selected_masks,
                    realsense_stamp=rgbd.stamp,
                    sam2_scene_index=int(masks.scene_index),
                    scene_started_monotonic=self.scene.started_monotonic,
                    expected_cloud_frame="helios2_optical_frame",
                )

                if not validation.valid:
                    self._finish(
                        ResultStatus.FAILED,
                        FailureCode.INVALID_RESULT,
                        "Observation pair invalid: "
                        + "; ".join(validation.errors),
                    )
                    return

                # ------------------------------------------
                # Record successfully validated observations
                # ------------------------------------------

                self.scene.record_observation(
                    "realsense",
                    rgbd.stamp,
                )

                self.scene.record_observation(
                    "sam2",
                    masks.stamp,
                )

                self.scene.sam2_scene_index = masks.scene_index

                # ------------------------------------------
                # Classify transaction result
                # ------------------------------------------

                if masks.object_count == 0:
                    self._finish(
                        ResultStatus.NO_OBJECTS_DETECTED,
                        FailureCode.NONE,
                        "SAM2 produced zero masks; "
                        "this does not prove the bin is empty",
                    )
                else:
                    if self.settings.get("enable_mask_fusion", False):

                        # M2.3C.4: Start controlled Mask Fusion.
                        self._start_fusion()

                    else:

                        # Preserve the earlier M1/M2.2 test behavior.
                        self._finish(
                            ResultStatus.SUCCESS,
                            FailureCode.NONE,
                            "M2.2 verified: observation pair passed "
                            "structural validation; "
                            "fusion not enabled",
                        )

                return

        if (monotonic() - self._sam2_sent_at
                >= self.settings["sam2_timeout_sec"]):
            self.restart_required = True
            self._finish(
                ResultStatus.TIMEOUT, FailureCode.SAM2_TIMEOUT,
                "SAM2 response, masks, or matching RGB-D metadata missing; "
                "restart before retry",
            )

    def _start_fusion(self):
        """Publish the validated observation pair exactly once."""

        scene_id = self.scene.scene_id

        try:
            # Register the expected result BEFORE publishing.
            self.manager.expect_fusion_result(scene_id)

            # Start measuring fusion response time.
            self._fusion_sent_at = monotonic()

            # Publish the exact pair accepted by this transaction.
            self.manager.publish_fusion_input(
                scene_id=scene_id,
                cloud=self._selected_cloud.payload,
                masks=self._selected_masks.payload,
            )

        except Exception as error:
            self._finish(
                ResultStatus.FAILED,
                FailureCode.MASK_FUSION_FAILED,
                f"Could not submit FusionInput: {error}",
            )
            return

        self.scene.enter_state(
            CoordinatorState.WAITING_FOR_MASK_FUSION
        )

    def _wait_for_fusion(self):
        """Wait for Mask Fusion, classify its result, and advance the workflow."""
        result = self.manager.get_fusion_result(self.scene.scene_id)

        if result is None:
            if (monotonic() - self._fusion_sent_at
                    >= self.settings["fusion_timeout_sec"]):
                self._finish(
                    ResultStatus.TIMEOUT,
                    FailureCode.MASK_FUSION_TIMEOUT,
                    "No matching FusionResult received "
                    "before the fusion timeout",
                )
            return

        # PerceptionManager also filters unrelated transactions.
        if result.scene_id != self.scene.scene_id:
            return

        expected_index = self.scene.sam2_scene_index
        if int(result.scene_index) != int(expected_index):
            self._finish(
                ResultStatus.FAILED,
                FailureCode.INVALID_RESULT,
                "FusionResult scene_index does not match "
                "the selected SAM2 scene",
            )
            return

        # Processing failure is distinct from a malformed successful result.
        if not result.success:
            self._finish(
                ResultStatus.FAILED,
                FailureCode.MASK_FUSION_FAILED,
                f"Mask Fusion reported failure: {result.detail}",
            )
            return

        fused = result.fused_objects
        validation = validate_fusion_output(
            fused,
            expected_scene_index=expected_index,
            expected_cloud_stamp=self._selected_cloud.stamp,
            approved_mask_objects=self._selected_masks.payload.objects,
        )
        if not validation.valid:
            self._finish(
                ResultStatus.FAILED,
                FailureCode.INVALID_RESULT,
                validation.reason,
            )
            return

        self.scene.record_observation("mask_fusion", validation.fused_stamp)

        if self.settings.get("enable_suction_scoring", False):
            # Sends one approved scene and switches state to wait for scoring.
            self._start_suction(fused)
            return

        # Preserve existing fusion-only success path when scoring is disabled.
        self._finish(
            ResultStatus.SUCCESS,
            FailureCode.NONE,
            f"Mask Fusion completed; "
            f"fused_objects={len(fused.objects)}; "
            "geometric accuracy not yet verified",
        )


    # ==============================================
    # Start Controlled Suction Scoring
    # ==============================================

    def _start_suction(self, fused_objects):
        """
        Submit the validated FusedObjectArray to Sim-Suction.

        This method is called once after successful fusion.
        """

        scene_id = self.scene.scene_id

        # Remember which fused objects belong to this transaction.
        # Store IDs only, not additional copies of the point clouds.
        self._approved_suction_object_ids = {
            int(obj.object_id)
            for obj in fused_objects.objects
        }

        try:

            # Register the expected response BEFORE publishing.
            self.manager.expect_suction_result(scene_id)

            # Start the scoring timeout clock.
            self._suction_sent_at = monotonic()

            # Submit the exact validated fused objects.
            self.manager.publish_suction_input(
                scene_id=scene_id,
                fused_objects=fused_objects,
            )

        except Exception as error:

            self._finish(
                ResultStatus.FAILED,
                FailureCode.SCORING_FAILED,
                f"Could not submit SuctionInput: {error}",
            )

            return

        # Publishing succeeded. Wait for the corresponding result.
        self.scene.enter_state(
            CoordinatorState.WAITING_FOR_SUCTION_CANDIDATES
        )

    # ==============================================
    # Invalid Suction Result Handler
    # ==============================================

    def _reject_invalid_suction(self, reason: str):
        """Reject a structurally invalid suction result."""

        self._finish(
            ResultStatus.FAILED,
            FailureCode.INVALID_RESULT,
            f"Invalid SuctionResult: {reason}",
        )

    # ==============================================
    # M3.4C — Validate Sim-Suction Result
    # ==============================================

    def _wait_for_suction(self):
        """Wait for Sim-Suction, validate its output, and finish the scene.

        A scoring SUCCESS is not authorization for robot motion.
        """
        scene_id = self.scene.scene_id
        result = self.manager.get_suction_result(scene_id)

        if result is None:
            if (monotonic() - self._suction_sent_at
                    >= self.settings["scoring_timeout_sec"]):
                self._finish(
                    ResultStatus.TIMEOUT,
                    FailureCode.SCORING_TIMEOUT,
                    "No matching SuctionResult received "
                    "before the scoring timeout",
                )
            return

        if result.scene_id != scene_id:
            return

        validation = validate_suction_result(
            result,
            expected_scene_index=self.scene.sam2_scene_index,
            expected_cloud_stamp=self._selected_cloud.stamp,
            approved_object_ids=self._approved_suction_object_ids,
            expected_candidate_frame=self.settings["suction_result_frame"],
        )
        if not validation.valid:
            if validation.internal_error:
                self._finish(
                    ResultStatus.FAILED,
                    FailureCode.INTERNAL_ERROR,
                    validation.reason,
                )
            else:
                self._reject_invalid_suction(validation.reason)
            return

        output = result.candidates
        status = str(output.status)

        if status == "ERROR":
            self._finish(
                ResultStatus.FAILED,
                FailureCode.SCORING_FAILED,
                f"Sim-Suction reported ERROR: {output.error_message}",
            )
            return

        if status == "NO_CANDIDATES":
            self._finish(
                ResultStatus.NO_VALID_CANDIDATES,
                FailureCode.NONE,
                "Sim-Suction completed but produced "
                "zero suction candidates; "
                "this does not prove the bin is empty",
            )
            return

        # Validator only accepts OK, ERROR or NO_CANDIDATES, and checks
        # candidate geometry/IDs/ranks for OK before we can report success.
        self._validated_suction_result = result
        self._finish(
            ResultStatus.SUCCESS,
            FailureCode.NONE,
            f"Sim-Suction completed successfully; "
            f"candidates={len(output.candidates)}; "
            f"approved_objects={len(self._approved_suction_object_ids)}; "
            f"geometry_is_mask_limited="
            f"{output.geometry_is_mask_limited}; "
            f"robot collision safety not verified",
        )
