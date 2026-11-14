"""Pruebas sintéticas de la Fase 1 y contratos numéricos posteriores."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np

from app.camera import CameraError, CameraRig, FramePacket, RigFrame
from app.vision_lab.background_model import BackgroundModel
from app.vision_lab.biomechanics import calculate_elbow_angle, calculate_kinematics
from app.vision_lab.heatmaps import (
    HeatmapAccumulator,
    accumulate_motion,
    enhance_reference_frame,
    overlay_heatmap_on_frame,
    resize_bilinear,
)
from app.vision_lab.morphology import connected_components, dilate, erode, keep_components
from app.vision_lab.motion_detector import MotionDetector, MotionResult
from app.vision_lab.session_recorder import SessionRecorder
from app.vision_lab.segmentation import SegmentationResult, segment_foreground, segment_foreground_result
from app.vision_lab.trajectories import WristPoint
from app.vision_lab.ui_integration import ExternalVisionLab
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from app.trainer.results import ClassificationResult, PunchEvent


def _frame(height: int = 32, width: int = 48, value: int = 40) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class MorphologyTests(unittest.TestCase):
    def test_erode_and_dilate_are_numpy_binary_operations(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True
        self.assertEqual(int(np.count_nonzero(erode(mask))), 1)
        self.assertEqual(int(np.count_nonzero(dilate(mask))), 25)

    def test_keep_components_removes_small_noise(self) -> None:
        mask = np.zeros((8, 10), dtype=bool)
        mask[1:4, 1:4] = True
        mask[7, 9] = True
        cleaned = keep_components(mask, min_area=4)
        self.assertEqual(int(np.count_nonzero(cleaned)), 9)
        self.assertFalse(cleaned[7, 9])
        self.assertEqual(len(connected_components(cleaned, min_area=1)), 1)


class SegmentationTests(unittest.TestCase):
    def test_robust_noise_floor_rejects_small_sensor_variation(self) -> None:
        frames = [_frame(value=value) for value in (40, 41, 40, 41, 40, 41)]
        model = BackgroundModel()
        model.calibrate(frames)
        current = _frame(value=42)
        result = segment_foreground_result(
            current,
            model.background_rgb,
            threshold=1.0,
            noise_floor=model.noise_floor,
            noise_scale=2.0,
            min_area=4,
        )
        self.assertEqual(int(np.count_nonzero(result.foreground_mask)), 0)

    def test_background_update_respects_freeze_and_reliable_pixels(self) -> None:
        background = _frame(value=40)
        model = BackgroundModel(update_alpha=0.5)
        model.calibrate([background])
        changed = _frame(value=200)
        model.freeze_background = True
        model.update(changed)
        self.assertEqual(int(model.background_rgb[0, 0, 0]), 40)
        model.freeze_background = False
        reliable = np.zeros(background.shape[:2], dtype=bool)
        reliable[0, 0] = True
        model.update(changed, reliable)
        self.assertEqual(int(model.background_rgb[0, 0, 0]), 120)
        self.assertEqual(int(model.background_rgb[0, 1, 0]), 40)

    def test_median_background_and_foreground_component(self) -> None:
        background = _frame(value=40)
        model = BackgroundModel()
        model.calibrate([background, background.copy(), background.copy()])
        current = background.copy()
        current[10:20, 18:28] = 220
        result = segment_foreground_result(current, model.background_rgb, threshold=30, min_area=12)
        self.assertTrue(result.foreground_mask[14, 22])
        self.assertFalse(result.foreground_mask[2, 2])
        self.assertGreater(result.confidence, 0.0)

    def test_roi_rejects_motion_outside_training_region(self) -> None:
        background = _frame(value=40)
        current = background.copy()
        current[2:8, 2:8] = 220
        current[20:27, 30:38] = 220
        mask = segment_foreground(current, background, threshold=30, roi=(0.45, 0.45, 0.95, 0.95), min_area=5)
        self.assertFalse(mask[4, 4])
        self.assertTrue(mask[23, 33])

    def test_foreground_can_remain_when_local_motion_stops(self) -> None:
        background = _frame(value=40)
        current = background.copy()
        current[10:20, 18:28] = 220
        mask = segment_foreground(current, background, threshold=30, min_area=12)
        motion = MotionDetector(threshold=10, min_area=4)
        first = motion.process(current, 1_000_000_000)
        second = motion.process(current.copy(), 1_033_000_000)
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertEqual(first.score, 0.0)
        self.assertEqual(second.score, 0.0)

    def test_motion_detector_rejects_global_exposure_change(self) -> None:
        detector = MotionDetector(threshold=10, min_area=1)
        first = detector.process(_frame(value=40), 1_000_000_000)
        second_frame = _frame(value=90)
        second = detector.process(second_frame, 1_033_000_000)
        self.assertEqual(first.score, 0.0)
        self.assertEqual(second.score, 0.0)
        third_frame = second_frame.copy()
        third_frame[10:18, 18:26] = 180
        third = detector.process(third_frame, 1_066_000_000)
        self.assertGreater(np.count_nonzero(third.motion_mask), 0)


class VisionLabUiTests(unittest.TestCase):
    def test_activity_mask_is_empty_until_background_is_calibrated(self) -> None:
        lab = ExternalVisionLab(process_width=32, calibration_frames=20)
        lab._last_motion = MotionResult(
            np.ones((12, 16), dtype=bool),
            1.0,
            12 * 16,
            0.03,
        )
        self.assertIsNone(lab.activity_mask)

    def test_activity_mask_keeps_motion_inside_the_silhouette(self) -> None:
        lab = ExternalVisionLab(process_width=32, calibration_frames=20)
        motion_mask = np.zeros((12, 16), dtype=bool)
        motion_mask[2:5, 2:5] = True
        motion_mask[7:10, 10:13] = True
        foreground = np.zeros_like(motion_mask)
        foreground[7:10, 10:13] = True
        lab._last_motion = MotionResult(motion_mask, 0.1, int(np.count_nonzero(motion_mask)), 0.03)
        lab._last_segmentation = SegmentationResult(
            foreground_mask=foreground,
            raw_mask=foreground,
            difference=np.zeros_like(foreground, dtype=np.float32),
            component=None,
            confidence=0.8,
        )
        activity = lab.activity_mask
        self.assertIsNotNone(activity)
        assert activity is not None
        self.assertFalse(activity[3, 3])
        self.assertTrue(activity[8, 11])

    def test_heatmap_overlay_lifts_dark_reference_and_keeps_context(self) -> None:
        frame = np.full((12, 16, 3), 6, dtype=np.uint8)
        values = np.zeros((6, 8), dtype=np.float32)
        values[2:4, 3:5] = 1.0
        enhanced = enhance_reference_frame(frame)
        rendered = overlay_heatmap_on_frame(enhanced, values)
        self.assertGreater(float(np.mean(enhanced)), float(np.mean(frame)))
        self.assertGreater(float(np.mean(rendered[4:8, 6:10])), float(np.mean(rendered[:2, :2])))

    def test_calibration_is_24_frames_and_disconnect_is_empty_state(self) -> None:
        lab = ExternalVisionLab(process_width=48, calibration_frames=20)
        primary_frame = _frame()
        primary = FramePacket(primary_frame, 0, 1000, 48, 32, 1_000_000_000)
        external = FramePacket(primary_frame.copy(), 0, 1001, 48, 32, 1_001_000_000)
        rig = RigFrame(primary=primary, by_role={"front": primary, "depth": external})
        lab.request_recalibration()
        for _ in range(19):
            telemetry = lab.observe(rig)
            self.assertFalse(telemetry.calibrated)
        telemetry = lab.observe(rig)
        self.assertTrue(telemetry.calibrated)
        self.assertTrue(telemetry.sync_valid)
        disconnected = RigFrame(primary=primary, by_role={"front": primary})
        self.assertEqual(lab.observe(disconnected).status, "STEREN SIN CÁMARA")

    def test_auxiliary_read_failure_does_not_stop_front_camera(self) -> None:
        class FakeCamera:
            def __init__(self, packet: FramePacket | None = None) -> None:
                self.packet = packet
                self.released = False

            def read(self) -> FramePacket:
                if self.packet is None:
                    raise CameraError("auxiliar desconectada")
                return self.packet

            def release(self) -> None:
                self.released = True

        frame = _frame()
        primary = FramePacket(frame, 0, 1000, 48, 32)
        rig = CameraRig(front_index=0, depth_index=1, setup="depth")
        auxiliary = FakeCamera()
        rig._cameras = {"front": FakeCamera(primary), "depth": auxiliary}  # type: ignore[assignment]
        rig._resolved_indices = {"front": 0, "depth": 1}
        rig.read()
        self.assertEqual(set(rig.last_frame.by_role), {"front"})
        self.assertTrue(auxiliary.released)


class FutureContractTests(unittest.TestCase):
    def test_kinematics_uses_real_irregular_dt(self) -> None:
        points = (
            WristPoint(0, 0.0, 0.0, 1.0, "left"),
            WristPoint(100_000_000, 1.0, 0.0, 1.0, "left"),
            WristPoint(300_000_000, 3.0, 0.0, 1.0, "left"),
        )
        result = calculate_kinematics(points, smoothing_alpha=None)
        self.assertAlmostEqual(result.peak_speed_px_s, 10.0)
        self.assertAlmostEqual(result.path_length_px, 3.0)

    def test_kinematics_marks_lost_points_without_nan(self) -> None:
        points = (
            WristPoint(0, 0.0, 0.0, 1.0, "left"),
            WristPoint(100_000_000, float("nan"), 0.0, 0.0, "left", tracking_valid=False),
            WristPoint(200_000_000, 2.0, 0.0, 1.0, "left"),
        )
        result = calculate_kinematics(points, smoothing_alpha=None)
        self.assertEqual(result.positions[1], None)
        self.assertEqual(result.velocities_px_s[1], None)
        self.assertEqual(result.velocities_px_s[2], None)

    def test_elbow_angle_handles_degenerate_segments(self) -> None:
        self.assertAlmostEqual(calculate_elbow_angle((0, 0), (1, 0), (1, 1)), 90.0)
        self.assertIsNone(calculate_elbow_angle((1, 1), (1, 1), (2, 2)))

    def test_heatmap_accumulates_time_and_radial_hand_points(self) -> None:
        heatmap = np.zeros((12, 12), dtype=np.float32)
        mask = np.zeros((12, 12), dtype=bool)
        mask[5, 5] = True
        accumulate_motion(heatmap, mask, 0.25)
        accumulator = HeatmapAccumulator((12, 12), heatmap)
        accumulator.add_point(5, 5, weight=2.0, radius=1)
        self.assertGreater(float(np.max(accumulator.values)), 0.25)
        self.assertAlmostEqual(float(np.max(accumulator.normalized())), 1.0)

    def test_hand_heatmaps_are_independent_and_bilinear_resize_is_smooth(self) -> None:
        left = HeatmapAccumulator((4, 4))
        right = HeatmapAccumulator((4, 4))
        self.assertFalse(np.shares_memory(left.values, right.values))
        left.add_point(0.25, 0.25, radius=1)
        self.assertEqual(float(np.max(right.values)), 0.0)
        source = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        resized = resize_bilinear(source, 3, 3)
        self.assertAlmostEqual(float(resized[1, 1]), 0.5, places=5)

    def test_session_recorder_persists_events_trajectories_and_maps(self) -> None:
        points = [LandmarkObservation(0.5, 0.5, 0.0, 1.0, 1.0) for _ in range(33)]
        points[int(LandmarkId.LEFT_SHOULDER)] = LandmarkObservation(0.4, 0.3, 0.0, 1.0, 1.0)
        points[int(LandmarkId.RIGHT_SHOULDER)] = LandmarkObservation(0.6, 0.3, 0.0, 1.0, 1.0)
        points[int(LandmarkId.LEFT_ELBOW)] = LandmarkObservation(0.35, 0.45, 0.0, 1.0, 1.0)
        points[int(LandmarkId.RIGHT_ELBOW)] = LandmarkObservation(0.65, 0.45, 0.0, 1.0, 1.0)
        points[int(LandmarkId.RIGHT_WRIST)] = LandmarkObservation(0.7, 0.55, 0.0, 1.0, 1.0)
        result = ClassificationResult(
            "heuristic",
            "completado",
            4,
            "Gancho derecho",
            events=(PunchEvent(4, "Gancho derecho", "right", event_id="right:0", started_ms=0, ended_ms=100),),
        )
        mask = np.zeros((12, 16), dtype=bool)
        mask[4:8, 6:10] = True
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory), classifier="heuristic")
            for index, timestamp_ms in enumerate((0, 50, 100)):
                points[int(LandmarkId.LEFT_WRIST)] = LandmarkObservation(0.3 + index * 0.03, 0.55, 0.0, 1.0, 1.0)
                pose = PoseObservation(timestamp_ms, tuple(points), tuple(points))
                recorder.record_frame(
                    timestamp_ns=timestamp_ms * 1_000_000,
                    pose=pose,
                    result=result if index == 2 else ClassificationResult("heuristic", "neutral", 0, "Guardia / Neutral"),
                    frame_size=(16, 12),
                    primary_frame=np.full((12, 16, 3), 90, dtype=np.uint8),
                    active=True,
                    external_motion_mask=mask,
                    external_sync_valid=True,
                )
            report = recorder.finalize()
            self.assertTrue((report.directory / "punch_events.csv").is_file())
            self.assertTrue((report.directory / "wrist_trajectories.csv").is_file())
            self.assertTrue((report.directory / "heatmap_external_activity.npy").is_file())
            self.assertTrue((report.directory / "heatmap_left.npy").is_file())
            self.assertTrue((report.directory / "heatmap_right.npy").is_file())
            left_values = np.load(report.directory / "heatmap_left.npy")
            right_values = np.load(report.directory / "heatmap_right.npy")
            self.assertEqual(float(np.sum(left_values)), 0.0)
            self.assertGreater(float(np.sum(right_values)), 0.0)
            self.assertTrue((report.directory / "exports" / "heatmap_right.png").is_file())
            self.assertTrue((report.directory / "exports" / "reference_macbook.png").is_file())
            rendered = np.asarray(cv2.imread(str(report.directory / "exports" / "heatmap_right.png")))
            self.assertGreater(float(np.mean(rendered)), 20.0)
            self.assertEqual(report.summary["total_punches"], 1)
            self.assertFalse(report.summary["three_dimensional_metric_data"])


if __name__ == "__main__":
    unittest.main()
