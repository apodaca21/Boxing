"""Pruebas de la rutina de calibración y del preflight, sin webcam."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.config import AppConfig
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from app.trainer.calibration_session import (
    CalibrationProtocol,
    CalibrationStep,
    save_calibration,
)
from app.trainer.heuristic import HeuristicSettings
from app.trainer.preflight import tracking_snapshot


def _pose(timestamp_ms: int = 0) -> PoseObservation:
    points = [LandmarkObservation(0.5, 0.5, 0.0, 1.0, 1.0) for _ in range(33)]
    points[int(LandmarkId.LEFT_SHOULDER)] = LandmarkObservation(0.4, 0.3, 0.0, 1.0, 1.0)
    points[int(LandmarkId.RIGHT_SHOULDER)] = LandmarkObservation(0.6, 0.3, 0.0, 1.0, 1.0)
    points[int(LandmarkId.LEFT_ELBOW)] = LandmarkObservation(0.35, 0.45, 0.0, 1.0, 1.0)
    points[int(LandmarkId.RIGHT_ELBOW)] = LandmarkObservation(0.65, 0.45, 0.0, 1.0, 1.0)
    points[int(LandmarkId.LEFT_WRIST)] = LandmarkObservation(0.3, 0.6, 0.0, 1.0, 1.0)
    points[int(LandmarkId.RIGHT_WRIST)] = LandmarkObservation(0.7, 0.6, 0.0, 1.0, 1.0)
    points[int(LandmarkId.LEFT_HIP)] = LandmarkObservation(0.43, 0.7, 0.0, 1.0, 1.0)
    points[int(LandmarkId.RIGHT_HIP)] = LandmarkObservation(0.57, 0.7, 0.0, 1.0, 1.0)
    return PoseObservation(timestamp_ms, tuple(points), tuple(points))


class CalibrationSessionTests(unittest.TestCase):
    def test_preflight_reports_both_anatomical_arms_and_body(self) -> None:
        snapshot = tracking_snapshot(_pose(), AppConfig(), fps=30.0, orientation="frontal")
        self.assertTrue(snapshot["body_visible"])
        self.assertEqual(snapshot["left_arm_tracking_quality"], 1.0)
        self.assertEqual(snapshot["right_arm_tracking_quality"], 1.0)
        self.assertEqual(snapshot["required_landmarks_quality"], 1.0)
        self.assertEqual(snapshot["orientation"], "frontal")

    def test_protocol_requires_human_confirmation(self) -> None:
        protocol = CalibrationProtocol(
            prepare_ms=10,
            return_ms=10,
            next_ms=10,
            steps=(CalibrationStep("jab", "Jab", "Jab", 1, 1, 10),),
        )
        protocol.start(0)
        protocol.advance(10)
        protocol.advance(20)
        protocol.advance(30)
        self.assertTrue(protocol.view().review)
        self.assertFalse(protocol.finished)
        protocol.confirm()
        self.assertTrue(protocol.finished)
        self.assertEqual(protocol.records[0].status, "verified")
        self.assertEqual(protocol.records[0].verified_label_id, 1)

    def test_save_keeps_backup_and_does_not_apply_proposal(self) -> None:
        protocol = CalibrationProtocol(
            prepare_ms=0,
            return_ms=0,
            next_ms=0,
            steps=(CalibrationStep("jab", "Jab", "Jab", 1, 1, 1),),
        )
        protocol.start(0)
        protocol.advance(0)
        protocol.observe(
            0,
            {
                "pose_detected": True,
                "required_landmarks_quality": 1.0,
                "left_arm_tracking_quality": 1.0,
                "right_arm_tracking_quality": 1.0,
                "left_speed": 1.2,
                "right_speed": 0.1,
                "left_extension": 1.0,
                "right_extension": 0.8,
                "left_lateral": 0.1,
                "right_lateral": 0.1,
                "left_elbow_angle": 160.0,
                "right_elbow_angle": 90.0,
                "left_wrist_x": -0.3,
                "left_wrist_y": 0.2,
                "right_wrist_x": 0.3,
                "right_wrist_y": 0.2,
                "fps": 30.0,
                "cooldown_active": False,
            },
        )
        protocol.advance(1)
        protocol.advance(1)
        protocol.confirm()
        with tempfile.TemporaryDirectory() as directory:
            path = save_calibration(
                Path(directory),
                orientation="frontal",
                preflight_rows=[],
                records=protocol.records,
                current=HeuristicSettings(),
            )
            self.assertTrue((path / "calibration.json").is_file())
            self.assertTrue((path / "attempts.csv").is_file())
            self.assertTrue((path / "visibility.csv").is_file())
            self.assertTrue((path / "params.json").is_file())


if __name__ == "__main__":
    unittest.main()
