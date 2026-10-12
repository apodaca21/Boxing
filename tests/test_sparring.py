"""Secuencias sintéticas del modo sparring. No miden precisión en video real."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from app.trainer.acm_boxing import (
    INPUT_SIZE,
    SEQUENCE_LENGTH,
    ExternalSettings,
    SequenceRecognizer,
)
from app.trainer.calibration import MotionCalibrator
from app.trainer.diagnostics import DiagnosticLog
from app.trainer.heuristic import HeuristicClassifier, HeuristicSettings
from app.trainer.results import ClassificationResult, PunchEvent
from app.trainer.sensitivity import profile_settings
from app.trainer.session import TrainingSession
from tests.test_trainer import _features, _feed


def _high() -> HeuristicClassifier:
    return HeuristicClassifier(replace(profile_settings("high"), smooth_alpha=1.0))


def _rise(start_ms: int, *, base: float, side: str = "left") -> list:
    offsets = (0.0, 0.0, 0.18, 0.32, 0.40, 0.28)
    elbows = (90.0, 90.0, 110.0, 130.0, 150.0, 125.0)
    frames = []
    for index, (offset, elbow) in enumerate(zip(offsets, elbows, strict=True)):
        timestamp = start_ms + index * 40
        extension = base + offset
        if side == "left":
            frames.append(_features(timestamp, left_extension=extension, left_elbow=elbow))
        else:
            frames.append(_features(timestamp, right_extension=extension, right_elbow=elbow))
    return frames


def _partial_jab(start_ms: int, *, side: str = "left") -> list:
    extensions = (0.90, 0.90, 1.08, 1.22, 1.30, 1.18)
    elbows = (90.0, 90.0, 110.0, 130.0, 150.0, 125.0)
    frames = []
    for index, (extension, elbow) in enumerate(zip(extensions, elbows, strict=True)):
        timestamp = start_ms + index * 40
        if side == "left":
            frames.append(_features(timestamp, left_extension=extension, left_elbow=elbow))
        else:
            frames.append(_features(timestamp, right_extension=extension, right_elbow=elbow))
    return frames


def _pose_from_points(timestamp_ms: int, points: dict[int, tuple[float, float, float]]) -> PoseObservation:
    size = max(points) + 1
    slots = [LandmarkObservation(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(size)]
    for index, point in points.items():
        slots[index] = LandmarkObservation(point[0], point[1], point[2], 1.0, 1.0)
    packed = tuple(slots)
    return PoseObservation(timestamp_ms, packed, packed)


def _guard_points() -> dict[int, tuple[float, float, float]]:
    return {
        int(LandmarkId.LEFT_SHOULDER): (-0.18, -0.40, 0.0),
        int(LandmarkId.RIGHT_SHOULDER): (0.18, -0.40, 0.0),
        int(LandmarkId.LEFT_ELBOW): (-0.22, -0.10, 0.02),
        int(LandmarkId.RIGHT_ELBOW): (0.22, -0.10, 0.02),
        int(LandmarkId.LEFT_WRIST): (-0.20, 0.05, 0.04),
        int(LandmarkId.RIGHT_WRIST): (0.20, 0.05, 0.04),
        int(LandmarkId.LEFT_HIP): (-0.12, 0.20, 0.0),
        int(LandmarkId.RIGHT_HIP): (0.12, 0.20, 0.0),
    }


def _rotate_y(point: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, y, z = point
    return (cosine * x + sine * z, y, -sine * x + cosine * z)


class SparringCombinationTests(unittest.TestCase):
    def test_normal_profile_drops_a_jab_that_does_not_return(self) -> None:
        classifier = HeuristicClassifier(HeuristicSettings(smooth_alpha=1.0))
        results = _feed(classifier, _partial_jab(0))
        self.assertFalse(any(result.counted for result in results))

    def test_short_jab_counts_once_on_high_sensitivity(self) -> None:
        results = _feed(_high(), _partial_jab(0))
        events = [event for result in results for event in result.events]
        self.assertEqual([event.label_id for event in events], [1])
        self.assertTrue(any(result.moving and result.candidate and not result.counted for result in results))
        self.assertTrue(all(result.latency_ms is not None and result.latency_ms >= 0 for result in results))

    def test_fast_cross_counts_on_the_right_arm(self) -> None:
        results = _feed(_high(), _partial_jab(0, side="right"))
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [2],
        )

    def test_two_jabs_count_when_the_arm_is_ready_again(self) -> None:
        frames = _partial_jab(0) + _rise(400, base=1.18)
        results = _feed(_high(), frames)
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1, 1],
        )

    def test_jab_then_cross_does_not_wait_for_a_shared_guard(self) -> None:
        left = _partial_jab(0, side="left")
        right = _partial_jab(80, side="right")
        merged = []
        for index in range(max(len(left), len(right))):
            timestamp = index * 40
            left_frame = left[index] if index < len(left) else _features(timestamp)
            right_frame = right[index] if index < len(right) else _features(timestamp)
            merged.append(
                _features(
                    timestamp,
                    left_extension=left_frame.left_shoulder_wrist_distance or 0.90,
                    left_elbow=left_frame.left_elbow_angle or 90.0,
                    right_extension=right_frame.right_shoulder_wrist_distance or 0.90,
                    right_elbow=right_frame.right_elbow_angle or 90.0,
                )
            )
        results = _feed(_high(), merged)
        labels = [event.label_id for result in results for event in result.events]
        self.assertEqual(labels, [1, 2])

    def test_hook_can_follow_a_straight_punch(self) -> None:
        frames = _partial_jab(0)
        for index, position in enumerate((-0.30, -0.55, -0.85, -0.70), start=1):
            frames.append(
                _features(
                    400 + index * 40,
                    left_x=position,
                    left_elbow=105.0,
                    left_extension=1.18,
                )
            )
        results = _feed(_high(), frames)
        labels = [event.label_id for result in results for event in result.events]
        self.assertEqual(labels, [1, 3])

    def test_guard_sway_is_not_a_punch(self) -> None:
        frames = [_features(0)]
        for index, extension in enumerate((0.94, 0.98, 0.94, 0.90, 0.90), start=1):
            frames.append(_features(index * 40, left_extension=extension, left_elbow=95))
        results = _feed(_high(), frames)
        self.assertFalse(any(result.counted for result in results))

    def test_body_translation_and_torso_yaw_do_not_count(self) -> None:
        classifier = _high()
        translated = []
        for index in range(8):
            shift = index * 0.08
            points = {
                key: (point[0] + shift, point[1], point[2])
                for key, point in _guard_points().items()
            }
            translated.append(
                classifier.update(
                    _features(index * 40),
                    pose_detected=True,
                    pose=_pose_from_points(index * 40, points),
                )
            )
        self.assertFalse(any(result.counted for result in translated))

        turned = _high()
        yaw_results = []
        for index in range(8):
            angle = index * 0.18
            points = {
                key: _rotate_y(point, angle)
                for key, point in _guard_points().items()
            }
            yaw_results.append(
                turned.update(
                    _features(index * 40),
                    pose_detected=True,
                    pose=_pose_from_points(index * 40, points),
                )
            )
        self.assertFalse(any(result.counted for result in yaw_results))

    def test_tracking_loss_does_not_invent_a_punch(self) -> None:
        frames = _partial_jab(0)[:4]
        frames.extend(
            _features(160 + index * 40, left_missing=True, right_missing=True)
            for index in range(10)
        )
        results = _feed(_high(), frames)
        self.assertFalse(any(result.counted for result in results))
        self.assertEqual(results[-1].status, "no_detection")

    def test_cooldown_suppresses_a_duplicate_of_the_same_motion(self) -> None:
        classifier = HeuristicClassifier(replace(profile_settings("high"), smooth_alpha=1.0, cooldown_ms=1000))
        results = _feed(classifier, _partial_jab(0) + _rise(400, base=1.18))
        self.assertEqual(sum(len(result.events) for result in results), 1)
        self.assertTrue(any(result.rejection == "cooldown" for result in results))

    def test_sensitivity_can_change_without_restarting(self) -> None:
        classifier = _high()
        self.assertEqual(classifier.settings.profile_name, "high")
        low_elbow = []
        extensions = (0.90, 0.90, 1.02, 1.08, 1.10, 1.04)
        for index, extension in enumerate(extensions):
            low_elbow.append(_features(index * 40, left_extension=extension, left_elbow=90))
        before = _feed(classifier, low_elbow)
        self.assertFalse(any(result.counted for result in before))

        classifier.set_sensitivity("maximum")
        classifier.reset()
        after = _feed(classifier, low_elbow)
        self.assertEqual(classifier.settings.profile_name, "maximum")
        self.assertEqual(
            [event.label_id for result in after for event in result.events],
            [1],
        )
        classifier.set_sensitivity("normal")
        self.assertFalse(classifier.settings.allow_early_confirm)
        self.assertEqual(classifier.settings.cooldown_ms, 280)

    def test_candidate_does_not_increment_the_counter(self) -> None:
        session = TrainingSession()
        session.training_style = "sparring"
        session.advance(
            40,
            ClassificationResult(
                source="heuristic",
                status="extendiendo",
                label_id=None,
                display_name="Jab izquierdo",
                candidate="Jab izquierdo",
                moving=True,
                sensitivity="high",
            ),
        )
        view = session.view(40)
        self.assertEqual(view.total, 0)
        self.assertEqual(view.candidate, "Jab izquierdo")
        self.assertTrue(view.moving)
        self.assertEqual(view.sensitivity, "HIGH")
        session.advance(
            80,
            ClassificationResult(
                source="heuristic",
                status="completado",
                label_id=1,
                display_name="Jab izquierdo",
                events=(PunchEvent(1, "Jab izquierdo", "left"),),
                sensitivity="high",
            ),
        )
        self.assertEqual(session.view(80).total, 1)
        self.assertEqual(session.view(80).last_punch, "Jab izquierdo")


class ExternalWindowTests(unittest.TestCase):
    def test_default_recovery_is_shorter_than_the_original_two_second_lock(self) -> None:
        self.assertEqual(ExternalSettings().recovery_frames, 20)
        self.assertEqual(SEQUENCE_LENGTH, 25)
        self.assertEqual(INPUT_SIZE, 16)

    def test_sliding_window_counts_again_without_a_fresh_buffer(self) -> None:
        def predict(sequence: np.ndarray) -> np.ndarray:
            self.assertEqual(sequence.shape[1], INPUT_SIZE)
            self.assertLess(sequence.shape[0], 25)
            return np.asarray([6.0, 0.0, 0.0, 0.0], dtype=np.float32)

        recognizer = SequenceRecognizer(
            predict,
            settings=ExternalSettings(sequence_length=4, recovery_frames=2, min_punch_speed=1.0),
        )
        from tests.test_external import _feed as feed_pose

        first = feed_pose(recognizer, 4, left_speed=30.0)
        self.assertEqual(sum(len(result.events) for result in first), 1)
        blocked = feed_pose(recognizer, 2, left_speed=30.0)
        self.assertFalse(any(result.counted for result in blocked))
        self.assertTrue(any(result.rejection == "cooldown" for result in blocked))
        second = feed_pose(recognizer, 1, left_speed=30.0)
        self.assertEqual(sum(len(result.events) for result in second), 1)


class CalibrationTests(unittest.TestCase):
    def test_calibration_proposes_thresholds_and_is_not_a_training_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calibrator = MotionCalibrator(Path(directory) / "calibration.json")
            calibrator.start(0)
            for timestamp in range(0, 14000, 100):
                calibrator.observe(
                    timestamp,
                    {
                        "left_extension": 0.9 if timestamp < 6000 else 1.5,
                        "right_extension": 0.9,
                        "left_speed": 0.05 if timestamp < 6000 else 1.2,
                        "right_speed": 0.05,
                        "left_lateral": 0.05 if timestamp < 10000 else 0.60,
                        "right_lateral": 0.05,
                    },
                )
            self.assertTrue(calibrator.finished(14000))
            proposed = calibrator.propose(profile_settings("high"))
            calibrator.save(proposed)
            saved = Path(directory, "calibration.json").read_text(encoding="utf-8")
            self.assertIn("No son etiquetas para entrenar NEAT", saved)
            self.assertNotIn("frames", saved)
            self.assertEqual(proposed.profile_name, "calibrado")
            self.assertTrue(proposed.allow_early_confirm)

            log = DiagnosticLog(Path(directory) / "diagnostics.jsonl")
            log.write({"timestamp_ms": 0, "kind": "rechazo", "rejection": "cooldown"})
            log.write({"timestamp_ms": 100, "kind": "rechazo", "rejection": "cooldown"})
            log.write({"timestamp_ms": 700, "kind": "confirmado", "display_name": "Jab izquierdo"})
            lines = Path(directory, "diagnostics.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
