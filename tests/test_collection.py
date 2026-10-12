"""Pruebas sintéticas del flujo de recopilación, sin webcam ni capturas reales."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from uuid import uuid4

import numpy as np

from app.config import AppConfig
from app.data_collector import DataCollector
from app.dataset_manager import DatasetManager
from app.geometry import extract_biomechanical_features
from app.overlay import render_frame
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from main import parse_args


def synthetic_pose() -> PoseObservation:
    coordinates = {
        LandmarkId.LEFT_SHOULDER: (0.40, 0.40),
        LandmarkId.RIGHT_SHOULDER: (0.60, 0.40),
        LandmarkId.LEFT_ELBOW: (0.34, 0.55),
        LandmarkId.RIGHT_ELBOW: (0.66, 0.55),
        LandmarkId.LEFT_WRIST: (0.30, 0.72),
        LandmarkId.RIGHT_WRIST: (0.70, 0.72),
        LandmarkId.LEFT_HIP: (0.43, 0.78),
        LandmarkId.RIGHT_HIP: (0.57, 0.78),
    }
    landmarks = [
        LandmarkObservation(0.5, 0.5, 0.0, 0.0, 0.0) for _ in range(33)
    ]
    for landmark_id, (x, y) in coordinates.items():
        landmarks[int(landmark_id)] = LandmarkObservation(x, y, 0.0, 1.0, 1.0)
    return PoseObservation(timestamp_ms=0, landmarks=tuple(landmarks))


class CollectionFlowTests(unittest.TestCase):
    def test_valid_synthetic_sequence_is_saved_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(data_dir=Path(temporary), mirror_view=True)
            dataset = DatasetManager(config.data_dir)
            session_id = str(uuid4())
            collector = DataCollector(
                config,
                dataset,
                clock=lambda: datetime(
                    2026, 10, 10, 22, 0, tzinfo=timezone(timedelta(hours=-7))
                ),
                session_id_factory=lambda: session_id,
            )
            pose = synthetic_pose()
            features = extract_biomechanical_features(
                pose,
                min_visibility=config.min_landmark_visibility,
                min_presence=config.min_landmark_presence,
                image_size=(640, 480),
            )
            self.assertIsNotNone(features.left_elbow_angle)
            self.assertIsNotNone(features.wrist_separation)

            collector.handle_key(ord("1"), 0)
            collector.handle_key(ord(" "), 0)
            for timestamp in range(0, 3000, 100):
                self.assertIsNone(
                    collector.advance(timestamp, pose, features, 640, 480)
                )
            self.assertEqual(collector.view().countdown_seconds, 1)
            collector.handle_key(ord("4"), 2900)
            self.assertEqual(collector.view().movement_name, "Jab izquierdo")

            messages = [
                message
                for timestamp in range(3000, 5001, 100)
                if (
                    message := collector.advance(timestamp, pose, features, 640, 480)
                )
            ]

            self.assertEqual(messages, ["Captura válida guardada."])
            self.assertEqual(collector.saved_count, 1)
            session = dataset.load_session(session_id)
            self.assertEqual(session.saved_sample_count, 1)
            self.assertEqual(session.stance, "orthodox")
            sample = dataset.load_sample(
                dataset.sample_path(session_id, session.sample_ids[0])
            )
            self.assertEqual(sample.label_name, "jab_izquierdo")
            self.assertEqual(sample.split_unit, "sample")
            self.assertEqual(sample.frames[0].landmarks["left_shoulder"].x, 0.40)  # type: ignore[union-attr]
            self.assertGreater(sample.duration_ms, sample.frames[0].elapsed_ms)
            self.assertEqual(
                sample.frames[-1].elapsed_ms,
                sample.frames[-1].timestamp_ms - sample.frames[0].timestamp_ms,
            )

    def test_short_capture_is_rejected_and_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(data_dir=Path(temporary))
            dataset = DatasetManager(config.data_dir)
            collector = DataCollector(config, dataset)
            pose = synthetic_pose()
            features = extract_biomechanical_features(
                pose,
                min_visibility=config.min_landmark_visibility,
                min_presence=config.min_landmark_presence,
                image_size=(640, 480),
            )
            collector.handle_key(ord(" "), 0)
            collector.advance(3000, pose, features, 640, 480)
            message = collector.advance(5000, pose, features, 640, 480)

            self.assertIsNotNone(message)
            self.assertIn("fotogramas", message or "")
            self.assertEqual(collector.view().outcome, "rejected")
            self.assertEqual(collector.saved_count, 0)
            self.assertFalse(Path(temporary, "samples").exists())
            self.assertEqual(
                dataset.load_session(collector.session_id).saved_sample_count,
                0,
            )

    def test_collection_overlay_preserves_frame(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = PoseObservation(timestamp_ms=1)
        config = AppConfig()
        features = extract_biomechanical_features(pose, image_size=(640, 480))
        with tempfile.TemporaryDirectory() as temporary:
            collector = DataCollector(
                AppConfig(data_dir=Path(temporary)),
                DatasetManager(Path(temporary)),
            )
            output = render_frame(
                frame,
                pose,
                features,
                30.0,
                config,
                collector.view(),
            )

        self.assertEqual(output.shape, frame.shape)
        self.assertFalse(np.shares_memory(output, frame))


class CommandLineTests(unittest.TestCase):
    def test_track_remains_the_default_mode(self) -> None:
        original = sys.argv
        try:
            sys.argv = ["main.py"]
            self.assertEqual(parse_args().mode, "track")
            sys.argv = ["main.py", "--mode", "collect"]
            self.assertEqual(parse_args().mode, "collect")
            sys.argv = ["main.py", "--mode", "neat"]
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()