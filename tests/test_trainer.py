"""Pruebas del detector provisional, la sesión y los clasificadores."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

from app.config import AppConfig
from app.evolution.features import vector_size
from app.geometry import BiomechanicalFeatures, RelativePoint
from app.overlay import render_frame
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from app.trainer.factory import create_classifier
from app.trainer.heuristic import HeuristicClassifier, HeuristicSettings, _arm_observation
from app.trainer.neat_classifier import ClassifierUnavailable, NeatClassifier
from app.trainer.results import (
    STATUS_NO_DETECTION,
    STATUS_UNKNOWN,
    ClassificationResult,
    PunchEvent,
)
from app.trainer.session import TrainingSession
from main import parse_args


def _features(
    timestamp_ms: int,
    *,
    left_extension: float = 0.90,
    left_elbow: float = 90.0,
    left_x: float = -0.30,
    left_y: float = 0.20,
    right_extension: float = 0.90,
    right_elbow: float = 90.0,
    right_x: float = 0.30,
    right_y: float = 0.20,
    left_missing: bool = False,
    right_missing: bool = False,
) -> BiomechanicalFeatures:
    return BiomechanicalFeatures(
        timestamp_ms=timestamp_ms,
        left_elbow_angle=None if left_missing else left_elbow,
        right_elbow_angle=None if right_missing else right_elbow,
        left_shoulder_wrist_distance=None if left_missing else left_extension,
        right_shoulder_wrist_distance=None if right_missing else right_extension,
        left_wrist_relative=None if left_missing else RelativePoint(left_x, left_y),
        right_wrist_relative=None if right_missing else RelativePoint(right_x, right_y),
        wrist_separation=None if left_missing or right_missing else abs(left_x - right_x),
    )


def _feed(
    classifier: HeuristicClassifier | NeatClassifier,
    frames: list[BiomechanicalFeatures],
    *,
    pose_detected: bool = True,
) -> list[ClassificationResult]:
    return [
        classifier.update(frame, pose_detected=pose_detected)
        for frame in frames
    ]


def _guard(timestamp_ms: int) -> BiomechanicalFeatures:
    return _features(timestamp_ms)


def _straight_cycle(
    start_ms: int,
    *,
    side: str,
    step_ms: int = 40,
) -> list[BiomechanicalFeatures]:
    extensions = (0.90, 0.90, 1.20, 1.55, 1.90, 2.10, 1.70, 1.30, 1.00, 0.90, 0.90)
    elbows = (90, 90, 130, 155, 168, 172, 160, 120, 95, 90, 90)
    frames = []
    for index, (extension, elbow) in enumerate(zip(extensions, elbows, strict=True)):
        timestamp = start_ms + index * step_ms
        if side == "left":
            frames.append(_features(timestamp, left_extension=extension, left_elbow=elbow))
        else:
            frames.append(
                _features(timestamp, right_extension=extension, right_elbow=elbow)
            )
    return frames


def _hook_cycle(start_ms: int, *, side: str, step_ms: int = 40) -> list[BiomechanicalFeatures]:
    positions = (-0.30, -0.30, -0.62, -0.90, -1.15, -1.15, -0.80, -0.50, -0.32, -0.30)
    frames = []
    for index, position in enumerate(positions):
        timestamp = start_ms + index * step_ms
        if side == "left":
            frames.append(
                _features(timestamp, left_x=position, left_elbow=100.0, left_extension=0.98)
            )
        else:
            frames.append(
                _features(
                    timestamp,
                    right_x=-position,
                    right_elbow=100.0,
                    right_extension=0.98,
                )
            )
    return frames


class HeuristicDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = HeuristicClassifier(HeuristicSettings(smooth_alpha=1.0))

    def test_neutral_sequence_does_not_count(self) -> None:
        results = _feed(self.classifier, [_guard(index * 40) for index in range(12)])

        self.assertTrue(all(result.label_id == 0 for result in results))
        self.assertTrue(all(not result.counted for result in results))
        self.assertEqual(results[-1].source, "heuristic")

    def test_default_smoothing_still_counts_a_clear_jab(self) -> None:
        results = _feed(HeuristicClassifier(), _straight_cycle(0, side="left"))
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1],
        )

    def test_left_extension_and_return_counts_one_jab(self) -> None:
        results = _feed(self.classifier, _straight_cycle(0, side="left"))

        events = [event for result in results for event in result.events]
        self.assertEqual([event.label_id for event in events], [1])
        self.assertEqual(events[0].arm, "left")

    def test_right_extension_and_return_counts_one_cross(self) -> None:
        results = _feed(self.classifier, _straight_cycle(0, side="right"))

        events = [event for result in results for event in result.events]
        self.assertEqual([event.label_id for event in events], [2])
        self.assertEqual(events[0].arm, "right")

    def test_left_and_right_hooks_are_distinguished(self) -> None:
        left = _feed(HeuristicClassifier(HeuristicSettings(smooth_alpha=1.0)), _hook_cycle(0, side="left"))
        right = _feed(
            HeuristicClassifier(HeuristicSettings(smooth_alpha=1.0)),
            _hook_cycle(0, side="right"),
        )

        self.assertEqual([event.label_id for result in left for event in result.events], [3])
        self.assertEqual([event.label_id for result in right for event in result.events], [4])

    def test_small_motion_is_not_a_punch(self) -> None:
        frames = [_guard(0)]
        for index, extension in enumerate((0.94, 1.00, 0.96, 0.90), start=1):
            frames.append(_features(index * 40, left_extension=extension, left_elbow=100))

        results = _feed(self.classifier, frames)
        self.assertFalse(any(result.counted for result in results))

    def test_held_extension_counts_only_when_the_arm_returns(self) -> None:
        frames = [_guard(0), _guard(40)]
        for index in range(2, 12):
            frames.append(_features(index * 40, left_extension=2.0, left_elbow=170))
        results = _feed(self.classifier, frames)
        self.assertFalse(any(result.counted for result in results))

        frames.extend(
            _features(timestamp, left_extension=extension, left_elbow=elbow)
            for timestamp, extension, elbow in (
                (480, 1.4, 140),
                (520, 1.0, 100),
                (560, 0.90, 90),
                (600, 0.90, 90),
            )
        )
        results = _feed(self.classifier, frames)
        self.assertEqual(sum(len(result.events) for result in results), 1)

    def test_cooldown_blocks_a_second_count(self) -> None:
        classifier = HeuristicClassifier(
            HeuristicSettings(smooth_alpha=1.0, cooldown_ms=1000)
        )
        frames = _straight_cycle(0, side="left") + _straight_cycle(500, side="left")
        results = _feed(classifier, frames)
        self.assertEqual(sum(len(result.events) for result in results), 1)

        later = HeuristicClassifier(HeuristicSettings(smooth_alpha=1.0, cooldown_ms=200))
        separated = _straight_cycle(0, side="left") + _straight_cycle(1200, side="left")
        later_results = _feed(later, separated)
        self.assertEqual(sum(len(result.events) for result in later_results), 2)

    def test_single_frame_spike_is_smoothed_away(self) -> None:
        classifier = HeuristicClassifier(HeuristicSettings(smooth_alpha=0.2))
        frames = [_guard(index * 40) for index in range(4)]
        frames.append(_features(160, left_extension=2.2, left_elbow=175, left_x=-0.30))
        frames.extend(_guard(200 + index * 40) for index in range(6))

        results = _feed(classifier, frames)
        self.assertFalse(any(result.counted for result in results))

    def test_tracking_loss_cancels_an_open_cycle(self) -> None:
        frames = _straight_cycle(0, side="left")[:6]
        frames.extend(
            _features(240 + index * 40, left_missing=True, right_missing=True)
            for index in range(8)
        )
        results = _feed(self.classifier, frames)

        self.assertFalse(any(result.counted for result in results))
        self.assertEqual(results[-1].status, STATUS_NO_DETECTION)

        hidden = _feed(self.classifier, [_guard(0), _guard(40)], pose_detected=False)
        self.assertTrue(all(result.status == STATUS_NO_DETECTION for result in hidden))
        self.assertFalse(any(result.counted for result in hidden))

    def test_ambiguous_motion_stays_unknown(self) -> None:
        frames = [_guard(0)]
        extensions = (1.2, 1.6, 2.0, 2.0, 1.4, 1.0, 0.9)
        for index, extension in enumerate(extensions, start=1):
            frames.append(_features(index * 40, left_extension=extension, left_elbow=95))
        results = _feed(self.classifier, frames)

        self.assertFalse(any(result.counted for result in results))
        self.assertTrue(any(result.status == STATUS_UNKNOWN for result in results))
        self.assertTrue(all(result.label_id != 1 for result in results))


    def test_camera_facing_jab_counts_from_depth(self) -> None:
        classifier = HeuristicClassifier()
        guard_elbow = (-0.13, -0.18, 0.06)
        guard_wrist = (-0.22, -0.28, 0.14)
        punch_elbow = (-0.16, -0.36, -0.18)
        punch_wrist = (-0.15, -0.34, -0.48)

        def mix(start: tuple[float, float, float], end: tuple[float, float, float], amount: float):
            return tuple((1.0 - amount) * start[axis] + amount * end[axis] for axis in range(3))

        amounts = (0.0, 0.0, 0.35, 0.7, 1.0, 1.0, 0.65, 0.3, 0.0)
        results = []
        for index, amount in enumerate(amounts):
            elbow = mix(guard_elbow, punch_elbow, amount)
            wrist = mix(guard_wrist, punch_wrist, amount)
            features = _guard(index * 40)
            results.append(
                classifier.update(
                    features,
                    pose_detected=True,
                    pose=_arm_pose(index * 40, elbow, wrist),
                )
            )

        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1],
        )

    def test_brief_tracking_gap_still_counts_the_return(self) -> None:
        frames = _straight_cycle(0, side="left")
        gap_at = 5
        results = []
        for index, frame in enumerate(frames):
            detected = index not in {gap_at, gap_at + 1}
            sample = frame if detected else _features(frame.timestamp_ms, left_missing=True, right_missing=True)
            results.append(self.classifier.update(sample, pose_detected=detected))

        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1],
        )

    def test_lateral_projection_with_progressive_elbow_opening_is_a_jab(self) -> None:
        """Una proyección lateral no debe convertir un recto en gancho."""

        frames = []
        positions = (-0.30, -0.30, -0.52, -0.78, -1.00, -1.00, -0.70, -0.42, -0.30)
        extensions = (0.90, 0.90, 1.20, 1.55, 1.90, 2.10, 1.70, 1.30, 0.90)
        elbows = (90, 90, 130, 155, 168, 172, 160, 120, 90)
        for index, (x, extension, elbow) in enumerate(zip(positions, extensions, elbows, strict=True)):
            frames.append(
                _features(
                    index * 40,
                    left_x=x,
                    left_extension=extension,
                    left_elbow=elbow,
                )
            )

        results = _feed(self.classifier, frames)
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1],
        )
        completed = [result for result in results if result.counted]
        self.assertIn("recto", completed[0].diagnostics["left_classification_reason"])

    def test_two_jabs_and_jab_cross_use_independent_arm_recovery(self) -> None:
        first = _straight_cycle(0, side="left")
        guard = [_guard(480), _guard(520), _guard(560)]
        second = _straight_cycle(600, side="left")
        results = _feed(self.classifier, first + guard + second)
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1, 1],
        )

        classifier = HeuristicClassifier(HeuristicSettings(smooth_alpha=1.0))
        left = _straight_cycle(0, side="left")
        right = _straight_cycle(600, side="right")
        results = _feed(classifier, left + [_guard(440), _guard(480), _guard(520)] + right)
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1, 2],
        )

    def test_partial_return_closes_one_event_without_duplicate(self) -> None:
        frames = _straight_cycle(0, side="left")[:-2]
        frames.extend(
            _features(timestamp, left_extension=extension, left_elbow=elbow)
            for timestamp, extension, elbow in (
                (360, 1.20, 130),
                (400, 1.10, 110),
            )
        )
        results = _feed(self.classifier, frames)
        self.assertEqual(sum(len(result.events) for result in results), 1)

    def test_fast_cycle_uses_timestamps_and_still_closes(self) -> None:
        results = _feed(self.classifier, _straight_cycle(0, side="right", step_ms=10))
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [2],
        )

    def test_brief_wrist_loss_is_tolerated_but_insufficient_tracking_is_not(self) -> None:
        frames = _straight_cycle(0, side="left")
        results = []
        for index, frame in enumerate(frames):
            if index == 4:
                frame = _features(frame.timestamp_ms, left_missing=True)
            results.append(self.classifier.update(frame, pose_detected=True))
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1],
        )

        classifier = HeuristicClassifier(HeuristicSettings(smooth_alpha=1.0))
        frames = _straight_cycle(0, side="left")[:5]
        frames.extend(
            _features(200 + index * 40, left_missing=True, right_missing=True)
            for index in range(8)
        )
        results = _feed(classifier, frames)
        self.assertFalse(any(result.counted for result in results))
        self.assertEqual(results[-1].status, STATUS_NO_DETECTION)

    def test_unreliable_world_landmarks_do_not_create_depth(self) -> None:
        slots = [LandmarkObservation(0.0, 0.0, 99.0, 0.0, 0.0) for _ in range(17)]
        normalized_points = {
            int(LandmarkId.LEFT_SHOULDER): (-0.18, -0.40, 0.0),
            int(LandmarkId.RIGHT_SHOULDER): (0.18, -0.40, 0.0),
            int(LandmarkId.LEFT_ELBOW): (-0.22, -0.10, 0.0),
            int(LandmarkId.LEFT_WRIST): (-0.20, 0.05, 0.0),
        }
        for index, point in normalized_points.items():
            slots[index] = LandmarkObservation(*point, 1.0, 1.0)
        pose = PoseObservation(0, tuple(slots), tuple(slots))
        observation = _arm_observation(_guard(0), "left", pose)
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.depth, 0.0)

    def test_ambiguous_cycle_does_not_renew_cooldown(self) -> None:
        classifier = HeuristicClassifier(
            HeuristicSettings(smooth_alpha=1.0, cooldown_ms=300)
        )
        frames = _straight_cycle(0, side="left")
        frames.extend(_guard(timestamp) for timestamp in (440, 480, 520, 560))
        frames.extend(
            _features(timestamp, left_extension=extension, left_elbow=95.0)
            for timestamp, extension in (
                (600, 0.90),
                (640, 1.20),
                (680, 1.50),
                (720, 1.00),
                (760, 0.90),
                (800, 0.90),
            )
        )
        frames.extend(_straight_cycle(840, side="left"))
        results = _feed(classifier, frames)
        self.assertEqual(
            [event.label_id for result in results for event in result.events],
            [1, 1],
        )
        self.assertTrue(any(result.rejection == "tipo no distinguible" for result in results))


def _arm_pose(
    timestamp_ms: int,
    elbow: tuple[float, float, float],
    wrist: tuple[float, float, float],
) -> PoseObservation:
    slots = [LandmarkObservation(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(17)]
    points = {
        int(LandmarkId.LEFT_SHOULDER): (-0.18, -0.4, 0.0),
        int(LandmarkId.RIGHT_SHOULDER): (0.18, -0.4, 0.0),
        int(LandmarkId.LEFT_ELBOW): elbow,
        int(LandmarkId.LEFT_WRIST): wrist,
        int(LandmarkId.RIGHT_ELBOW): (0.28, -0.15, 0.05),
        int(LandmarkId.RIGHT_WRIST): (0.22, 0.0, 0.08),
    }
    for index, point in points.items():
        slots[index] = LandmarkObservation(point[0], point[1], point[2], 1.0, 1.0)
    packed = tuple(slots)
    return PoseObservation(timestamp_ms, packed, packed)


class SessionTests(unittest.TestCase):
    def test_pause_and_reset_control_the_counters(self) -> None:
        session = TrainingSession()
        punch = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=1,
            display_name="Jab izquierdo",
            events=(PunchEvent(1, "Jab izquierdo", "left"),),
            active_arm="left",
        )
        self.assertTrue(session.running)
        session.handle_key(ord("p"))
        session.advance(0, punch)
        self.assertEqual(session.counts[1], 0)

        session.handle_key(ord("p"))
        session.advance(1000, punch)
        session.advance(2500, punch)
        self.assertEqual(session.counts[1], 2)
        self.assertEqual(session.view(2500).elapsed_ms, 1500)
        self.assertAlmostEqual(session.view(2500).punches_per_minute or 0.0, 80.0)

        session.handle_key(ord(" "))
        session.advance(3000, punch)
        self.assertEqual(session.counts[1], 2)
        self.assertFalse(session.view(3000).running)

        session.handle_key(ord("r"))
        view = session.view(3000)
        self.assertEqual(view.total, 0)
        self.assertEqual(view.elapsed_ms, 0)
        self.assertIsNone(view.punches_per_minute)
        self.assertTrue(view.running)

    def test_classifier_results_share_the_same_session_contract(self) -> None:
        session = TrainingSession()
        session.handle_key(ord("p"))
        for source, provisional in (("heuristic", True), ("neat", False)):
            session.advance(
                100,
                ClassificationResult(
                    source=source,
                    status="completado",
                    label_id=2,
                    display_name="Cross derecho",
                    events=(PunchEvent(2, "Cross derecho", "right"),),
                    active_arm="right",
                ),
            )
            view = session.view(100)
            self.assertEqual(view.source, source)
            self.assertEqual(view.provisional, provisional)
            self.assertNotIn("%", view.recognition)


class ClassifierAvailabilityTests(unittest.TestCase):
    def test_factory_switches_available_classifiers(self) -> None:
        heuristic = create_classifier("heuristic")
        self.assertEqual(heuristic.name, "heuristic")
        self.assertTrue(heuristic.provisional)

        with self.assertRaises(ClassifierUnavailable) as external:
            create_classifier(
                "external",
                external_dir=Path("/tmp/evobox-sin-lstm"),
            )
        self.assertIn("no está disponible", str(external.exception))
        self.assertNotIn("NEAT", heuristic.__class__.__name__)

        with self.assertRaises(ClassifierUnavailable) as neat:
            create_classifier("neat", results_dir=__import__("pathlib").Path("/tmp/evobox-sin-modelo"))
        self.assertIn("NEAT", str(neat.exception))

    def test_neat_window_preserves_missing_values_and_counts_once(self) -> None:
        captured: dict[str, tuple[float | None, ...] | tuple[float, ...]] = {}
        labels = iter((1, 1, 1, 0, 0, 0, 1, 1))

        def transform(raw: tuple[float | None, ...]) -> tuple[float, ...]:
            captured["raw"] = raw
            return tuple(-4.0 if value is None else float(value) for value in raw)

        def activate(vector: tuple[float, ...]) -> tuple[float, ...]:
            captured["vector"] = vector
            label = next(labels)
            outputs = [0.0, 0.0, 0.0, 0.0, 0.0]
            outputs[label] = 1.0
            return tuple(outputs)

        classifier = NeatClassifier(
            transform,
            activate,
            time_steps=8,
            cooldown_ms=0,
            neutral_frames_to_release=3,
        )
        frames = []
        for index in range(9):
            frames.append(
                _features(
                    index * 50,
                    left_elbow=None,
                    left_extension=1.2,
                )
            )
        results = _feed(classifier, frames)
        raw = captured["raw"]
        vector = captured["vector"]
        assert isinstance(raw, tuple)
        assert isinstance(vector, tuple)
        self.assertEqual(len(raw), vector_size(8))
        self.assertEqual(len(vector), vector_size(8))
        missing_index = raw.index(None)
        self.assertEqual(vector[missing_index], -4.0)
        self.assertNotEqual(vector[missing_index], 0.0)
        self.assertEqual(sum(len(result.events) for result in results), 2)
        self.assertTrue(all(result.source == "neat" for result in results[1:]))

    def test_live_hud_keeps_the_camera_frame_intact(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = PoseObservation(timestamp_ms=1)
        session = TrainingSession()
        session.handle_key(ord("p"))
        session.advance(
            1000,
            ClassificationResult(
                source="heuristic",
                status="completado",
                label_id=1,
                display_name="Jab izquierdo",
                events=(PunchEvent(1, "Jab izquierdo", "left"),),
                active_arm="left",
            ),
        )
        output = render_frame(
            frame,
            pose,
            _guard(1),
            30.0,
            AppConfig(),
            live=session.view(1200),
        )
        self.assertEqual(output.shape, frame.shape)
        self.assertFalse(np.shares_memory(output, frame))
        self.assertTrue(session.view(1200).provisional)
        self.assertGreater(int(output[:, :, 0].sum()), 0)

    def test_live_mode_parses_without_replacing_previous_modes(self) -> None:
        original = sys.argv
        try:
            sys.argv = ["main.py", "--mode", "live", "--classifier", "heuristic"]
            args = parse_args()
            self.assertEqual(args.mode, "live")
            self.assertEqual(args.classifier, "heuristic")
            sys.argv = ["main.py", "--mode", "train"]
            self.assertEqual(parse_args().mode, "train")
            sys.argv = ["main.py", "--mode", "collect"]
            self.assertEqual(parse_args().mode, "collect")
            sys.argv = ["main.py"]
            self.assertEqual(parse_args().mode, "track")
        finally:
            sys.argv = original

    def test_external_and_missing_neat_do_not_open_training(self) -> None:
        from main import main

        original = sys.argv
        try:
            expectations = {
                "external": "externo no está disponible",
                "neat": "No hay un modelo NEAT",
            }
            for classifier, expected in expectations.items():
                sys.argv = [
                    "main.py",
                    "--mode",
                    "live",
                    "--classifier",
                    classifier,
                    "--external-model",
                    "/tmp/evobox-sin-lstm",
                    "--results-dir",
                    "/tmp/evobox-sin-modelo",
                ]
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    code = main()
                self.assertEqual(code, 2, errors.getvalue())
                self.assertIn(expected, errors.getvalue())
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
