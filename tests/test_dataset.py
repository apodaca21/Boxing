"""Pruebas sintéticas del dataset. No representan capturas reales."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from app.config import STANCE_DESCRIPTION, AppConfig
from app.dataset_manager import (
    TRACKED_LANDMARK_NAMES,
    CameraMetadata,
    DatasetManager,
    DuplicateDatasetEntryError,
    FeatureRecord,
    FrameRecord,
    LandmarkRecord,
    MOVEMENTS,
    PlaneCoordinate,
    SessionRecord,
    assemble_sample,
    movement_by_id,
)
from app.sample_validator import validate_sample


def camera() -> CameraMetadata:
    return CameraMetadata(
        index=0,
        requested_width=640,
        requested_height=480,
        actual_width=640,
        actual_height=480,
        mirror_view=True,
    )


def reliable_landmark() -> LandmarkRecord:
    return LandmarkRecord(0.42, 0.36, -0.05, 0.99, 0.98, True)


def reliable_features() -> FeatureRecord:
    return FeatureRecord(
        left_elbow_angle=92.0,
        right_elbow_angle=168.0,
        left_shoulder_wrist_distance=1.25,
        right_shoulder_wrist_distance=0.80,
        left_wrist_relative=PlaneCoordinate(0.15, 0.70),
        right_wrist_relative=PlaneCoordinate(-0.20, 0.55),
        wrist_separation=1.10,
    )


def frame(
    timestamp_ms: int,
    *,
    missing_wrist: bool = False,
    detected: bool = True,
) -> FrameRecord:
    if detected:
        landmarks: dict[str, LandmarkRecord | None] = {
            name: reliable_landmark() for name in TRACKED_LANDMARK_NAMES
        }
        if missing_wrist:
            landmarks["left_wrist"] = None
        features = reliable_features()
    else:
        landmarks = {name: None for name in TRACKED_LANDMARK_NAMES}
        features = FeatureRecord.missing()
    return FrameRecord(
        timestamp_ms=timestamp_ms,
        elapsed_ms=0,
        pose_detected=detected,
        landmarks=landmarks,
        features=features,
    )


def synthetic_sample(session_id: str, *, count: int = 20, step_ms: int = 100):
    return assemble_sample(
        session_id=session_id,
        label=MOVEMENTS[1],
        frames=tuple(frame(index * step_ms) for index in range(count)),
        stance="orthodox",
        stance_description=STANCE_DESCRIPTION,
        camera=camera(),
        created_at="2026-10-10T22:00:00-07:00",
        sample_id=str(uuid4()),
    )


class DatasetPersistenceTests(unittest.TestCase):
    def test_roundtrip_preserves_sample_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = DatasetManager(Path(temporary))
            session_id = str(uuid4())
            manager.create_session(self.session(session_id))
            sample = synthetic_sample(session_id)

            path = manager.save_sample(sample)
            loaded = manager.load_sample(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            index = manager.load_index()

            self.assertEqual(loaded, sample)
            self.assertEqual(payload["split_unit"], "sample")
            self.assertFalse(payload["privacy"]["contains_images"])
            self.assertFalse(payload["privacy"]["contains_audio"])
            self.assertFalse(payload["privacy"]["contains_video"])
            self.assertEqual(len(index), 1)
            self.assertEqual(index[0]["sample_id"], sample.sample_id)
            self.assertEqual(
                index[0]["relative_path"],
                f"samples/{session_id}/{sample.sample_id}.json",
            )
            self.assertEqual(
                sum(line.startswith("sample_id,") for line in manager.index_path.read_text().splitlines()),
                1,
            )

    def test_missing_coordinate_is_explicit_null_and_can_remain_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = DatasetManager(Path(temporary))
            session_id = str(uuid4())
            manager.create_session(self.session(session_id))
            sample = synthetic_sample(session_id)
            frames = list(sample.frames)
            landmarks = dict(frames[0].landmarks)
            landmarks["left_wrist"] = None
            frames[0] = replace(frames[0], landmarks=landmarks)
            sample = replace(sample, frames=tuple(frames))

            result = validate_sample(sample, AppConfig())
            path = manager.save_sample(sample)
            loaded = manager.load_sample(path)

            self.assertTrue(result.accepted)
            self.assertIsNone(loaded.frames[0].landmarks["left_wrist"])
            self.assertIn('"left_wrist": null', path.read_text(encoding="utf-8"))

    def test_invalid_label_duplicate_name_and_incomplete_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = DatasetManager(Path(temporary))
            session_id = str(uuid4())
            manager.create_session(self.session(session_id))
            sample = synthetic_sample(session_id)
            config = AppConfig()

            with self.assertRaises(ValueError):
                movement_by_id(8)
            invalid_label = validate_sample(
                replace(sample, label_id=8, label_name="inventada"),
                config,
            )
            mismatched_name = validate_sample(
                replace(sample, label_name="cross_derecho"),
                config,
            )
            incomplete = validate_sample(synthetic_sample(session_id, count=2), config)
            hidden_body = validate_sample(
                replace(
                    sample,
                    frames=tuple(
                        frame(item.timestamp_ms, detected=False)
                        for item in sample.frames
                    ),
                ),
                config,
            )
            nan_frame = replace(
                sample.frames[0],
                features=replace(sample.frames[0].features, left_elbow_angle=math.nan),
            )
            invalid_number = validate_sample(
                replace(sample, frames=(nan_frame, *sample.frames[1:])),
                config,
            )

            self.assertFalse(invalid_label.accepted)
            self.assertTrue(any("no es válida" in reason for reason in invalid_label.reasons))
            self.assertTrue(
                any("no coincide" in reason for reason in mismatched_name.reasons)
            )
            self.assertTrue(any("fotogramas" in reason for reason in incomplete.reasons))
            self.assertTrue(any("1.5 a 2.8" in reason for reason in incomplete.reasons))
            self.assertTrue(any("persona" in reason for reason in hidden_body.reasons))
            self.assertTrue(any("visibles" in reason for reason in hidden_body.reasons))
            self.assertTrue(
                any("numéricos" in reason for reason in invalid_number.reasons)
            )

            path = manager.save_sample(sample)
            original = path.read_text(encoding="utf-8")
            with self.assertRaises(DuplicateDatasetEntryError):
                manager.save_sample(sample)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

            second = synthetic_sample(session_id)
            manager.save_sample(second)
            identifiers = [row["sample_id"] for row in manager.load_index()]
            self.assertEqual(identifiers, [sample.sample_id, second.sample_id])
            self.assertEqual(len(set(identifiers)), 2)

    def test_session_persistence_and_timestamp_coherence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = DatasetManager(Path(temporary))
            session_id = str(uuid4())
            manager.create_session(self.session(session_id))
            with self.assertRaises(DuplicateDatasetEntryError):
                manager.create_session(self.session(session_id))

            sample = synthetic_sample(session_id)
            manager.save_sample(sample)
            reloaded = manager.load_session(session_id)
            self.assertEqual(reloaded.saved_sample_count, 1)
            self.assertEqual(reloaded.sample_ids, (sample.sample_id,))
            self.assertEqual(reloaded.stance, "orthodox")

            decreasing = list(sample.frames)
            decreasing[2] = replace(
                decreasing[2],
                timestamp_ms=decreasing[1].timestamp_ms,
            )
            timestamps = validate_sample(
                replace(sample, frames=tuple(decreasing)),
                AppConfig(),
            )
            elapsed = list(sample.frames)
            elapsed[4] = replace(elapsed[4], elapsed_ms=1)
            relative_times = validate_sample(
                replace(sample, frames=tuple(elapsed)),
                AppConfig(),
            )
            duration = validate_sample(
                replace(sample, duration_ms=sample.duration_ms + 25),
                AppConfig(),
            )

            self.assertTrue(
                any("estrictamente crecientes" in reason for reason in timestamps.reasons)
            )
            self.assertTrue(
                any("tiempos relativos" in reason for reason in relative_times.reasons)
            )
            self.assertTrue(
                any("duración no coincide" in reason for reason in duration.reasons)
            )

    def session(self, session_id: str) -> SessionRecord:
        return SessionRecord(
            session_id=session_id,
            started_at="2026-10-10T22:00:00-07:00",
            stance="orthodox",
            stance_description=STANCE_DESCRIPTION,
            camera=camera(),
            saved_sample_count=0,
            sample_ids=(),
        )


if __name__ == "__main__":
    unittest.main()