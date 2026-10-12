"""Pruebas del LSTM real y del conteo de eventos, sin abrir la webcam."""

from __future__ import annotations

import time
import unittest

import numpy as np

from app.config import PROJECT_ROOT
from app.trainer.acm_boxing import (
    INPUT_SIZE,
    KEYPOINT_MAP,
    SEQUENCE_LENGTH,
    ExternalSettings,
    SequenceRecognizer,
    normalize_keypoints,
    punch_event,
)
from app.trainer.external import load_lstm, select_torch_device
from app.trainer.session import TrainingSession
from app.trainer.results import ClassificationResult


WEIGHTS = PROJECT_ROOT / "models" / "external"
PUNCH = WEIGHTS / "punch_classifier.pth"


def _person(left_wrist: tuple[float, float], right_wrist: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((17, 2), dtype=np.float64)
    confidences = np.ones(17, dtype=np.float64)
    keypoints[KEYPOINT_MAP["left_shoulder"]] = (100.0, 100.0)
    keypoints[KEYPOINT_MAP["right_shoulder"]] = (200.0, 100.0)
    keypoints[KEYPOINT_MAP["left_elbow"]] = (110.0, 150.0)
    keypoints[KEYPOINT_MAP["right_elbow"]] = (190.0, 150.0)
    keypoints[KEYPOINT_MAP["left_wrist"]] = left_wrist
    keypoints[KEYPOINT_MAP["right_wrist"]] = right_wrist
    keypoints[KEYPOINT_MAP["left_hip"]] = (120.0, 230.0)
    keypoints[KEYPOINT_MAP["right_hip"]] = (180.0, 230.0)
    return keypoints, confidences


def _feed(recognizer: SequenceRecognizer, steps: int, *, left_speed: float, right_speed: float = 0.0):
    results = []
    for step in range(steps):
        keypoints, confidences = _person(
            (80.0 + step * left_speed, 140.0),
            (210.0 + step * right_speed, 140.0),
        )
        results.append(recognizer.push(keypoints, confidences))
    return results


class PreprocessingTests(unittest.TestCase):
    def test_normalization_matches_the_original_contract(self) -> None:
        keypoints, confidences = _person((150.0, 200.0), (210.0, 140.0))
        normalized = normalize_keypoints(keypoints, confidences)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized.shape, (INPUT_SIZE,))
        self.assertNotEqual(normalized.shape[0], 92)
        # Hombros en x=100 y x=200: centro 150, escala 100. La muñeca izquierda queda en (0, 1).
        self.assertAlmostEqual(float(normalized[8]), 0.0, places=5)
        self.assertAlmostEqual(float(normalized[9]), 1.0, places=5)

        confidences[KEYPOINT_MAP["left_elbow"]] = 0.1
        hidden = normalize_keypoints(keypoints, confidences)
        assert hidden is not None
        self.assertEqual(float(hidden[4]), 0.0)
        self.assertEqual(float(hidden[5]), 0.0)

        confidences[KEYPOINT_MAP["left_shoulder"]] = 0.1
        self.assertIsNone(normalize_keypoints(keypoints, confidences))

    def test_labels_do_not_invent_sides(self) -> None:
        jab = punch_event("jab", "left")
        cross = punch_event("cross", "right")
        uppercut = punch_event("uppercut", "left")
        hook = punch_event("hook", None)
        left_hook = punch_event("hook", "left")
        right_hook = punch_event("hook", "right")

        self.assertEqual(jab.display_name, "Jab")
        self.assertIsNone(jab.label_id)
        self.assertEqual(cross.display_name, "Cross")
        self.assertIsNone(cross.label_id)
        self.assertEqual(uppercut.display_name, "Uppercut")
        self.assertEqual(uppercut.counter_key, "uppercut")
        self.assertEqual(hook.display_name, "HOOK")
        self.assertEqual(hook.counter_key, "hook")
        self.assertEqual(left_hook.label_id, 3)
        self.assertEqual(right_hook.label_id, 4)


class EventTests(unittest.TestCase):
    def _recognizer(self, logits: list[float], *, recovery: int = 3) -> SequenceRecognizer:
        def predict(sequence: np.ndarray) -> np.ndarray:
            self.assertEqual(sequence.shape, (4, INPUT_SIZE))
            return np.asarray(logits, dtype=np.float32)

        return SequenceRecognizer(
            predict,
            settings=ExternalSettings(
                sequence_length=4,
                recovery_frames=recovery,
                min_punch_speed=1.0,
                min_arm_pixels=5.0,
            ),
        )

    def test_one_motion_counts_once_until_recovery_ends(self) -> None:
        recognizer = self._recognizer([6.0, 0.0, 0.0, 0.0], recovery=3)
        first = _feed(recognizer, 4, left_speed=30.0)
        self.assertEqual(sum(len(result.events) for result in first), 1)
        self.assertEqual(first[-1].events[0].display_name, "Jab")
        held = _feed(recognizer, 3, left_speed=30.0)
        self.assertFalse(any(result.counted for result in held))
        again = _feed(recognizer, 1, left_speed=30.0)
        self.assertEqual(sum(len(result.events) for result in again), 1)

    def test_hook_side_and_uppercut_stay_distinct(self) -> None:
        left = self._recognizer([0.0, 0.0, 6.0, 0.0])
        left_results = _feed(left, 4, left_speed=40.0, right_speed=1.0)
        self.assertEqual(left_results[-1].events[0].label_id, 3)

        ambiguous = self._recognizer([0.0, 0.0, 6.0, 0.0])
        ambiguous_results = _feed(ambiguous, 4, left_speed=30.0, right_speed=30.0)
        self.assertEqual(ambiguous_results[-1].events[0].display_name, "HOOK")
        self.assertIsNone(ambiguous_results[-1].events[0].label_id)

        upper = self._recognizer([0.0, 0.0, 0.0, 6.0])
        upper_results = _feed(upper, 4, left_speed=30.0)
        event = upper_results[-1].events[0]
        self.assertEqual(event.display_name, "Uppercut")
        self.assertIsNone(event.label_id)

    def test_tracking_loss_and_model_errors_do_not_count(self) -> None:
        recognizer = self._recognizer([6.0, 0.0, 0.0, 0.0])
        hidden = [recognizer.lost() for _ in range(3)]
        self.assertTrue(all(result.status == "no_detection" for result in hidden))
        self.assertFalse(any(result.counted for result in hidden))

        def fail(sequence: np.ndarray) -> np.ndarray:
            del sequence
            raise RuntimeError("fallo de inferencia")

        broken = SequenceRecognizer(
            fail,
            settings=ExternalSettings(sequence_length=4, min_punch_speed=1.0),
        )
        results = _feed(broken, 4, left_speed=30.0)
        self.assertEqual(results[-1].display_name, "ERROR DE MODELO")
        self.assertFalse(any(result.counted for result in results))

    def test_external_counters_do_not_reuse_the_heuristic_jab_slot(self) -> None:
        session = TrainingSession()
        session.advance(
            1000,
            ClassificationResult(
                source="external",
                status="completado",
                label_id=None,
                display_name="Jab",
                events=(punch_event("jab", "left"),),
            ),
        )
        view = session.view(1000)
        self.assertEqual(view.source, "external")
        self.assertEqual(view.counts[1], 0)
        self.assertEqual(view.total, 1)
        self.assertEqual(view.rows[0], ("Jab", 1, "neutral"))
        self.assertFalse(view.provisional)


class CheckpointTests(unittest.TestCase):
    def test_real_punch_weights_match_the_published_architecture(self) -> None:
        import torch

        self.assertTrue(PUNCH.is_file(), "Falta punch_classifier.pth descargado del repositorio original.")
        device = select_torch_device(torch)
        model = load_lstm(PUNCH, 4, "cpu")
        state = torch.load(PUNCH, map_location="cpu", weights_only=True)
        self.assertEqual(tuple(state["lstm.weight_ih_l0"].shape), (512, 16))
        self.assertEqual(tuple(state["lstm.weight_ih_l1"].shape), (512, 128))
        self.assertEqual(tuple(state["fc.weight"].shape), (4, 128))
        sample = torch.zeros(1, SEQUENCE_LENGTH, INPUT_SIZE)
        started = time.perf_counter()
        with torch.no_grad():
            output = model(sample)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(latency_ms, 0.0)
        print(f"LSTM cpu latencia {latency_ms:.2f} ms, dispositivo disponible {device}")

        if device == "mps":
            mps_model = load_lstm(PUNCH, 4, "mps")
            mps_sample = sample.to("mps")
            with torch.no_grad():
                mps_model(mps_sample)
            started = time.perf_counter()
            with torch.no_grad():
                mps_output = mps_model(mps_sample)
            mps_ms = (time.perf_counter() - started) * 1000.0
            self.assertEqual(tuple(mps_output.shape), (1, 4))
            self.assertTrue(torch.isfinite(mps_output).all())
            print(f"LSTM mps latencia {mps_ms:.2f} ms")

    def test_yolo_pose_loads_with_restricted_deserialization(self) -> None:
        pose = WEIGHTS / "yolo11m-pose.pt"
        self.assertTrue(pose.is_file())
        from app.trainer.external import ExternalClassifier

        classifier = ExternalClassifier(WEIGHTS)
        self.addCleanup(classifier.close)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        started = time.perf_counter()
        result = classifier.analyze_frame(frame)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.assertEqual(result.source, "external")
        self.assertFalse(result.counted)
        self.assertIsNotNone(classifier.yolo_latency_ms)
        self.assertGreater(classifier.yolo_latency_ms or 0.0, 0.0)
        self.assertIn(classifier.device, {"cpu", "mps"})
        inference_fps = 1000.0 / classifier.yolo_latency_ms if classifier.yolo_latency_ms else 0.0
        print(
            f"YOLO {classifier.device} latencia {classifier.yolo_latency_ms:.1f} ms, "
            f"FPS de esa inferencia {inference_fps:.1f}, "
            f"preparacion {classifier.warmup_latency_ms:.1f} ms, "
            f"ciclo medido {latency_ms:.1f} ms"
        )


if __name__ == "__main__":
    unittest.main()
