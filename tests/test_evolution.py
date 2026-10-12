"""Pruebas del pipeline evolutivo con fixtures sintéticos, nunca muestras reales."""

from __future__ import annotations

from argparse import Namespace
import math
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from app.config import STANCE_DESCRIPTION
from app.dataset_manager import (
    TRACKED_LANDMARK_NAMES,
    CameraMetadata,
    DatasetManager,
    FeatureRecord,
    FrameRecord,
    LandmarkRecord,
    MOVEMENTS,
    PlaneCoordinate,
    SessionRecord,
    assemble_sample,
)
from app.evolution.audit import audit_dataset
from app.evolution.experiment import run_training
from app.evolution.features import (
    MISSING_SENTINEL,
    FeatureScaler,
    extract_raw_features,
    feature_names,
    vector_size,
)
from app.evolution.metrics import fitness_from_predictions, predict_label
from app.evolution.metrics import FitnessError
from app.evolution.model_store import load_model
from app.evolution.neat_trainer import PRESETS, load_neat_config, render_config, train_population
from app.evolution.splitting import SplitError, split_by_group
from app.evolution.experiment import CONFIG_TEMPLATE


def _sample(angles: tuple[float | None, ...], *, label: int = 1, session_id: str | None = None):
    frames = []
    for index, angle in enumerate(angles):
        frames.append(
            FrameRecord(
                timestamp_ms=index * 1000,
                elapsed_ms=index * 1000,
                pose_detected=True,
                landmarks={},
                features=FeatureRecord(
                    left_elbow_angle=angle,
                    right_elbow_angle=90.0,
                    left_shoulder_wrist_distance=1.0,
                    right_shoulder_wrist_distance=1.0,
                    left_wrist_relative=PlaneCoordinate(0.0, float(index)),
                    right_wrist_relative=PlaneCoordinate(1.0, float(index)),
                    wrist_separation=1.0,
                ),
            )
        )
    return assemble_sample(
        session_id=session_id or str(uuid4()),
        label=MOVEMENTS[label],
        frames=tuple(frames),
        stance="orthodox",
        stance_description=STANCE_DESCRIPTION,
        camera=CameraMetadata(0, 640, 480, 640, 480, True),
        created_at="2026-10-10T22:00:00-07:00",
        sample_id=str(uuid4()),
    )


def _valid_dataset(root: Path) -> None:
    manager = DatasetManager(root)
    for session_number in range(3):
        session_id = str(uuid4())
        manager.create_session(
            SessionRecord(
                session_id=session_id,
                started_at="2026-10-10T22:00:00-07:00",
                stance="orthodox",
                stance_description=STANCE_DESCRIPTION,
                camera=CameraMetadata(0, 640, 480, 640, 480, True),
                saved_sample_count=0,
                sample_ids=(),
            )
        )
        for label in range(5):
            frames = []
            for index in range(20):
                frames.append(
                    FrameRecord(
                        timestamp_ms=index * 100,
                        elapsed_ms=0,
                        pose_detected=True,
                        landmarks={
                            name: LandmarkRecord(0.4, 0.4, 0.0, 1.0, 1.0, True)
                            for name in TRACKED_LANDMARK_NAMES
                        },
                        features=FeatureRecord(
                            40.0 + label * 20 + index,
                            150.0 - label * 10,
                            0.8 + label / 10,
                            1.2,
                            PlaneCoordinate(-0.2 + index / 100, 0.5 + label / 10),
                            PlaneCoordinate(0.3, 0.4 + index / 100),
                            1.0 + label / 5,
                        ),
                    )
                )
            manager.save_sample(
                assemble_sample(
                    session_id=session_id,
                    label=MOVEMENTS[label],
                    frames=tuple(frames),
                    stance="orthodox",
                    stance_description=STANCE_DESCRIPTION,
                    camera=CameraMetadata(0, 640, 480, 640, 480, True),
                    created_at="2026-10-10T22:00:00-07:00",
                )
            )


class FeatureTests(unittest.TestCase):
    def test_vector_size_and_temporal_resampling(self) -> None:
        self.assertEqual(vector_size(8), 92)
        self.assertEqual(len(feature_names(8)), 92)
        sample = _sample((0.0, 10.0))
        raw = extract_raw_features(sample, 3)
        self.assertEqual(raw[0], 0.0)
        self.assertAlmostEqual(raw[8] or 0.0, 5.0)
        self.assertEqual(raw[16], 10.0)
        self.assertEqual(len(raw), vector_size(3))

    def test_missing_values_are_not_replaced_by_zero(self) -> None:
        sample = _sample((None, None))
        raw = extract_raw_features(sample, 4)
        elbow_indexes = [step * 8 for step in range(4)]
        self.assertTrue(all(raw[index] is None for index in elbow_indexes))
        observed = _sample((0.0, 0.0))
        scaler = FeatureScaler.fit(
            [extract_raw_features(observed, 4), extract_raw_features(sample, 4)],
            4,
        )
        transformed = scaler.transform(raw)
        self.assertTrue(all(transformed[index] == MISSING_SENTINEL for index in elbow_indexes))
        real_zero = scaler.transform(extract_raw_features(observed, 4))
        self.assertNotEqual(real_zero[0], MISSING_SENTINEL)
        self.assertLessEqual(abs(real_zero[0]), scaler.clip_limit)

    def test_scaler_uses_only_training_rows(self) -> None:
        training = [(0.0, 0.0), (10.0, 10.0)]
        scaler = FeatureScaler.fit(
            [tuple([value, value] + [None] * (vector_size(2) - 2)) for value in (0.0, 10.0)],
            2,
        )
        other = FeatureScaler.fit(
            [tuple([value, value] + [None] * (vector_size(2) - 2)) for value in (100.0, 110.0)],
            2,
        )
        row = tuple([10.0, 10.0] + [None] * (vector_size(2) - 2))
        self.assertNotEqual(scaler.transform(row)[0], other.transform(row)[0])
        self.assertEqual(len(training), 2)


class SplitAndFitnessTests(unittest.TestCase):
    def test_session_split_is_reproducible_and_has_no_leakage(self) -> None:
        mapping = {f"sample-{session}-{item}": f"session-{session}" for session in range(4) for item in range(3)}
        first = split_by_group(mapping, seed=42, group_kind="session")
        second = split_by_group(mapping, seed=42, group_kind="session")
        self.assertEqual(first, second)
        groups = (set(first.train_groups), set(first.validation_groups), set(first.test_groups))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        samples = set(first.train_ids) | set(first.validation_ids) | set(first.test_ids)
        self.assertEqual(samples, set(mapping))
        with self.assertRaises(SplitError):
            split_by_group({"a": "only"}, seed=1, group_kind="session")

    def test_fitness_formula_and_required_labels(self) -> None:
        perfect = fitness_from_predictions([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 0)
        self.assertAlmostEqual(perfect.value, 1.0)
        neutral_only = fitness_from_predictions([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], 0)
        self.assertAlmostEqual(neutral_only.value, 0.35)
        penalized = fitness_from_predictions([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 1)
        self.assertAlmostEqual(penalized.value, 0.975)
        self.assertEqual(predict_label([0.1, 0.2, 0.9, 0.3, 0.4]), 2)
        with self.assertRaises(FitnessError):
            fitness_from_predictions([0, 1, 2, 3], [0, 1, 2, 3], 0)


class TrainingPipelineTests(unittest.TestCase):
    def test_real_dataset_absence_blocks_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = run_training(
                Namespace(
                    data_dir=root / "data",
                    results_dir=root / "results",
                    time_steps=8,
                    seed=42,
                    experiment="quick",
                    population=None,
                    generations=None,
                    split="session",
                )
            )
            audit = (root / "results" / "data_audit.json").read_text(encoding="utf-8")
            self.assertEqual(code, 2)
            self.assertIn("Faltan muestras reales", audit)
            self.assertEqual(list((root / "results").glob("**/fitness_mejor.png")), [])

    def test_saved_model_reloads_with_identical_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _valid_dataset(root / "data")
            code = run_training(
                Namespace(
                    data_dir=root / "data",
                    results_dir=root / "results",
                    time_steps=8,
                    seed=5,
                    experiment="quick",
                    population=6,
                    generations=1,
                    split="session",
                )
            )
            self.assertEqual(code, 0)
            model_dir = next((root / "results").glob("*/model"))
            model = load_model(model_dir)
            audit = audit_dataset(root / "data")
            sample = audit.records[0].sample
            vector = model.scaler.transform(extract_raw_features(sample, 8))
            self.assertEqual(model.predict_sample(sample), model.predict_sample(sample))
            self.assertEqual(model.predict_vector(vector), model.predict_vector(vector))
            self.assertTrue((model_dir.parent / "fitness_mejor.png").is_file())
            self.assertTrue((model_dir.parent / "confusion_prueba.png").is_file())
            self.assertIn("experiment_executed", (model_dir.parent / "resumen.json").read_text())

    def test_same_seed_reproduces_evolution(self) -> None:
        config_text = render_config(
            CONFIG_TEMPLATE.read_text(encoding="utf-8"),
            population=5,
            inputs=vector_size(2),
            max_stagnation=PRESETS["quick"]["max_stagnation"],
        )
        vectors = []
        labels = []
        for label in range(5):
            row = [float(label)] * vector_size(2)
            vectors.append(tuple(row))
            labels.append(label)
        histories = []
        outputs = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "neat.cfg"
            path.write_text(config_text, encoding="utf-8")
            for _ in range(2):
                winner, history = train_population(
                    vectors,
                    labels,
                    load_neat_config(path),
                    generations=1,
                    seed=11,
                )
                histories.append(history[0].best_fitness)
                network_config = load_neat_config(path)
                from app.evolution.neat_trainer import activation_vectors

                outputs.append(activation_vectors(winner, network_config, [vectors[0]]))
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(outputs[0], outputs[1])
        self.assertFalse(math.isnan(histories[0]))
