"""Evaluación guiada con eventos sintéticos. No mide golpes reales."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.benchmark.matching import evaluate_repetitions
from app.benchmark.protocol import BenchmarkProtocol, PredictionMark, RepetitionRecord
from app.benchmark.report import (
    _diagnose_record,
    reconciliation_metrics,
    update_comparison,
    write_session_report,
)
from app.benchmark.samples import VerifiedSampleStore
from app.config import AppConfig
from app.dataset_manager import (
    TRACKED_LANDMARK_NAMES,
    DatasetManager,
    FeatureRecord,
    FrameRecord,
    LandmarkRecord,
    MOVEMENTS,
    PlaneCoordinate,
)
from app.evolution.features import extract_raw_features, vector_size
from app.trainer.results import PunchEvent


def _mark(timestamp_ms: int, label_id: int | None, latency_ms: float | None = 12.0) -> PredictionMark:
    name = "sin clase" if label_id is None else MOVEMENTS[label_id].display_name
    return PredictionMark(timestamp_ms, label_id, name, latency_ms)


def _verified(
    label: int,
    start: int,
    end: int,
    marks: list[PredictionMark],
    *,
    status: str = "verified",
) -> RepetitionRecord:
    record = RepetitionRecord(
        requested_label_id=label,
        repetition_index=1,
        window_start_ms=start,
        window_end_ms=end,
        status=status,
        verified_label_id=None if status != "verified" else label,
    )
    record.predictions.extend(marks)
    record.fps_samples.append(30.0)
    return record


class MatchingTests(unittest.TestCase):
    def test_reconciliation_separates_wrong_classes_from_guard_false_positives(self) -> None:
        records = (
            _verified(0, 0, 1000, [_mark(100, 1)]),
            _verified(1, 1000, 2000, [_mark(1500, 2)]),
            _verified(2, 2000, 3000, []),
        )
        metrics = evaluate_repetitions(records, tolerance_ms=0)
        reconciliation = reconciliation_metrics(metrics, records)
        self.assertEqual(reconciliation["punch_wrong_classifications"], 1)
        self.assertEqual(reconciliation["punch_no_detection"], 1)
        self.assertEqual(reconciliation["guard_false_positives"], 1)
        self.assertEqual(reconciliation["reported_incorrect_detections"], 2)

    def test_one_prediction_matches_only_one_verified_punch(self) -> None:
        records = (
            _verified(1, 0, 1000, [_mark(500, 1)]),
            _verified(1, 2000, 3000, []),
        )
        metrics = evaluate_repetitions(records, tolerance_ms=0)
        self.assertEqual(metrics.matched_pairs, 1)
        self.assertEqual(metrics.scores[1].true_positives, 1)
        self.assertEqual(metrics.scores[1].false_negatives, 1)
        self.assertAlmostEqual(metrics.scores[1].recall or 0, 0.5)

    def test_diagnosis_uses_terminal_evidence(self) -> None:
        record = _verified(1, 0, 1000, [])
        record.evidence = [
            {"rejection": "cooldown", "phase": "neutral"},
            {"rejection": "sin compromiso", "phase": "neutral"},
        ]
        self.assertEqual(_diagnose_record(record), "activacion_insuficiente")

        record.evidence = [
            {"rejection": "cooldown", "phase": "neutral", "cooldown_active": True, "cooldown_remaining_ms": 120},
        ]
        self.assertEqual(_diagnose_record(record), "cooldown")

    def test_overlapping_windows_do_not_share_a_prediction(self) -> None:
        records = (
            _verified(1, 1000, 2000, [_mark(1600, 1)]),
            _verified(1, 1500, 2500, []),
        )
        metrics = evaluate_repetitions(records, tolerance_ms=0)
        self.assertEqual(metrics.matched_pairs, 1)
        self.assertEqual(metrics.scores[1].true_positives, 1)
        self.assertEqual(metrics.scores[1].false_negatives, 1)

    def test_duplicate_prediction_lowers_precision(self) -> None:
        record = _verified(1, 1000, 3000, [_mark(1500, 1, 30.0), _mark(1800, 1, 10.0)])
        metrics = evaluate_repetitions((record,), tolerance_ms=0)
        self.assertEqual(metrics.duplicates, 1)
        self.assertEqual(metrics.scores[1].true_positives, 1)
        self.assertEqual(metrics.scores[1].false_positives, 1)
        self.assertAlmostEqual(metrics.scores[1].precision or 0, 0.5)
        self.assertAlmostEqual(metrics.scores[1].recall or 0, 1.0)
        self.assertAlmostEqual(metrics.mean_latency_ms or 0, 10.0)

    def test_guard_prediction_is_a_false_positive_not_a_hit(self) -> None:
        record = _verified(0, 0, 2000, [_mark(1000, 2)])
        metrics = evaluate_repetitions((record,), tolerance_ms=0)
        self.assertEqual(metrics.neutral_false_positives, 1)
        self.assertEqual(metrics.neutral_clean, 0)
        self.assertEqual(metrics.scores[2].false_positives, 1)
        self.assertEqual(metrics.scores[2].true_positives, 0)
        self.assertIsNone(metrics.scores[2].recall)
        self.assertEqual(metrics.scores[0].recall, 0.0)
        self.assertIsNone(metrics.scores[0].precision)

    def test_wrong_class_is_not_a_true_positive(self) -> None:
        record = _verified(1, 0, 2000, [_mark(800, 2)])
        metrics = evaluate_repetitions((record,), tolerance_ms=0)
        self.assertEqual(metrics.scores[1].true_positives, 0)
        self.assertEqual(metrics.scores[1].false_negatives, 1)
        self.assertEqual(metrics.scores[2].false_positives, 1)
        self.assertEqual(metrics.incorrect, 1)

    def test_unconfirmed_repetition_is_not_scored(self) -> None:
        pending = _verified(1, 0, 2000, [_mark(500, 1)], status="pending")
        metrics = evaluate_repetitions((pending,), tolerance_ms=0)
        self.assertEqual(metrics.verified_repetitions, 0)
        self.assertEqual(metrics.matched_pairs, 0)
        self.assertEqual(metrics.scores[1].true_positives, 0)


class ProtocolTests(unittest.TestCase):
    def _protocol(self) -> BenchmarkProtocol:
        protocol = BenchmarkProtocol(
            repetitions=1,
            prepare_ms=100,
            active_ms=100,
            return_ms=100,
            next_ms=50,
            classes=(1,),
        )
        protocol.advance(0)
        return protocol

    def test_instructions_and_confirmation_gate(self) -> None:
        protocol = self._protocol()
        self.assertEqual(protocol.view(sensitivity="high", classifier_name="heuristic").instruction, "Prepárate.")
        protocol.advance(100)
        self.assertEqual(protocol.view(sensitivity="high", classifier_name="heuristic").instruction, "Ejecuta un jab izquierdo.")
        protocol.observe(120, (PunchEvent(1, "Jab izquierdo", "left"),), latency_ms=8.0, fps=25.0)
        protocol.advance(200)
        self.assertEqual(protocol.view(sensitivity="normal", classifier_name="heuristic").instruction, "Regresa a guardia.")
        protocol.observe(210, (PunchEvent(1, "Jab izquierdo", "left"),))
        protocol.advance(300)
        self.assertTrue(protocol.view(sensitivity="normal", classifier_name="heuristic").review)
        self.assertEqual(len(protocol.records[0].predictions), 1)
        self.assertEqual(evaluate_repetitions(protocol.records, tolerance_ms=0).verified_repetitions, 0)
        confirmed = protocol.confirm()
        assert confirmed is not None
        self.assertEqual(confirmed.verified_label_id, 1)
        self.assertFalse(confirmed.corrected)
        metrics = evaluate_repetitions(protocol.records, tolerance_ms=0)
        self.assertEqual(metrics.scores[1].true_positives, 1)

    def test_correction_and_exclusion(self) -> None:
        protocol = self._protocol()
        protocol.advance(100)
        protocol.advance(200)
        protocol.advance(300)
        self.assertTrue(protocol.correct(3))
        confirmed = protocol.confirm()
        assert confirmed is not None
        self.assertEqual(confirmed.verified_label_id, 3)
        self.assertTrue(confirmed.corrected)
        corrected_metrics = evaluate_repetitions(protocol.records, tolerance_ms=0)
        self.assertEqual(corrected_metrics.scores[3].support, 1)
        self.assertEqual(corrected_metrics.scores[1].support, 0)

        excluded = self._protocol()
        excluded.advance(100)
        excluded.advance(200)
        excluded.advance(300)
        excluded.observe(150, (PunchEvent(1, "Jab izquierdo", "left"),))
        self.assertIsNotNone(excluded.exclude())
        self.assertEqual(evaluate_repetitions(excluded.records).verified_repetitions, 0)
        self.assertEqual(evaluate_repetitions(excluded.records).excluded_repetitions, 1)

    def test_next_repetition_message(self) -> None:
        protocol = BenchmarkProtocol(
            repetitions=2,
            prepare_ms=10,
            active_ms=10,
            return_ms=10,
            next_ms=10,
            classes=(2,),
        )
        protocol.advance(0)
        protocol.advance(10)
        protocol.advance(20)
        protocol.advance(30)
        protocol.confirm()
        self.assertEqual(
            protocol.view(sensitivity="maximum", classifier_name="heuristic").instruction,
            "Siguiente repetición.",
        )

    def test_repeat_keeps_the_attempt_out_of_verified_metrics(self) -> None:
        protocol = self._protocol()
        protocol.advance(100)
        protocol.advance(200)
        protocol.advance(300)
        repeated = protocol.repeat(300)
        assert repeated is not None
        self.assertEqual(protocol.records[0].status, "repeated")
        self.assertEqual(protocol.phase, "prepare")
        self.assertEqual(protocol.records[1].requested_label_id, 1)
        self.assertEqual(len(protocol.verified_records()), 0)


class StorageTests(unittest.TestCase):
    def test_only_verified_windows_join_the_neat_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(data_dir=root / "data")
            store = VerifiedSampleStore(config, DatasetManager(config.data_dir))
            frames = []
            for index in range(20):
                frames.append(
                    FrameRecord(
                        timestamp_ms=index * 100,
                        elapsed_ms=0,
                        pose_detected=True,
                        landmarks={
                            name: LandmarkRecord(0.2, 0.3, 0.0, 1.0, 1.0, True)
                            for name in TRACKED_LANDMARK_NAMES
                        },
                        features=FeatureRecord(
                            90.0,
                            90.0,
                            1.0,
                            1.0,
                            PlaneCoordinate(-0.2, 0.1),
                            PlaneCoordinate(0.2, 0.1),
                            0.4,
                        ),
                    )
                )
            verified = _verified(1, 0, 1900, [])
            verified.verified_label_id = 4
            verified.corrected = True
            saved = store.save(verified, frames, width=640, height=480, created_at="2026-10-11T21:00:00-07:00")
            self.assertTrue(saved.saved)
            sample = DatasetManager(config.data_dir).load_sample(
                DatasetManager(config.data_dir).sample_path(store.session_id, saved.sample_id or "")
            )
            self.assertEqual(sample.label_id, 4)
            self.assertEqual(sample.label_name, "gancho_derecho")
            self.assertFalse(sample.contains_images or sample.contains_video or sample.contains_audio)
            self.assertEqual(sample.session_id, store.session_id)
            self.assertEqual(len(extract_raw_features(sample)), vector_size(8))

            excluded = _verified(1, 0, 1900, [], status="excluded")
            with self.assertRaises(ValueError):
                store.save(excluded, frames, width=640, height=480)

    def test_report_does_not_rank_a_profile_for_counting_more(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sensitivity, detections in (("normal", 0), ("maximum", 3)):
                protocol = BenchmarkProtocol(repetitions=1, classes=(0, 1, 2, 3, 4), prepare_ms=1, active_ms=1, return_ms=1, next_ms=1)
                for label in (0, 1, 2, 3, 4):
                    record = _verified(label, 0, 1000, [_mark(200, 1)] * detections)
                    record.repetition_index = 1
                    protocol.records.append(record)
                write_session_report(
                    protocol,
                    root / sensitivity / "session",
                    classifier_name="heuristic",
                    sensitivity=sensitivity,
                    tolerance_ms=0,
                )
            comparison = update_comparison(root)
            assert comparison is not None
            payload = json.loads(comparison.read_text(encoding="utf-8"))
            self.assertIsNone(payload["winner"])
            self.assertIn("no es mejor", payload["note"])
            self.assertTrue((root / "normal" / "session" / "confusion.png").is_file())
            self.assertTrue((root / "normal" / "session" / "metrics.csv").is_file())
            self.assertTrue((root / "normal" / "session" / "events.csv").is_file())
            self.assertTrue((root / "normal" / "session" / "attempts.csv").is_file())
            report = json.loads((root / "normal" / "session" / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("recognition_metrics", report)
            self.assertIn("counter_metrics", report)


if __name__ == "__main__":
    unittest.main()
