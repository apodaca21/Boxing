"""Pruebas del entrenamiento con objetivos. No miden precisión en video real."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.config import AppConfig
from app.dataset_manager import (
    TRACKED_LANDMARK_NAMES,
    DatasetManager,
    FeatureRecord,
    FrameRecord,
    LandmarkRecord,
    PlaneCoordinate,
)
from app.evolution.audit import audit_dataset
from app.evolution.features import extract_raw_features, vector_size
from app.evolution.splitting import split_by_group
from app.geometry import BiomechanicalFeatures
from app.overlay import render_frame
from app.pose_tracker import PoseObservation
from app.targets.generator import BLOCK_LABELS, BlockPlan, TargetGenerator
from app.targets.impact import judge_target
from app.targets.placement import (
    MAX_REACH,
    PointSmoother,
    body_offset,
    clamp_offset,
    offset_for,
    project_target,
    radius_for,
)
from app.targets.review import (
    CaptureDraft,
    ReviewStore,
    dataset_gap_lines,
    list_review_sessions,
    open_review_session,
    reason_text,
)
from app.targets.training import TargetTraining
from app.targets.view import TargetView


def _frame(timestamp_ms: int) -> FrameRecord:
    return FrameRecord(
        timestamp_ms=timestamp_ms,
        elapsed_ms=0,
        pose_detected=True,
        landmarks={
            name: LandmarkRecord(0.4, 0.4, 0.0, 1.0, 1.0, True)
            for name in TRACKED_LANDMARK_NAMES
        },
        features=FeatureRecord(
            40.0,
            150.0,
            0.8,
            1.2,
            PlaneCoordinate(-0.2, 0.5),
            PlaneCoordinate(0.3, 0.4),
            1.0,
        ),
    )


def _frames(start: int = 0, count: int = 21, step: int = 100) -> list[FrameRecord]:
    return [_frame(start + index * step) for index in range(count)]


def _draft(label: int, outcome: str, frames: list[FrameRecord], detector: int | None) -> CaptureDraft:
    return CaptureDraft(
        requested_label=label,
        outcome=outcome,
        detector_label=detector,
        detector_name="detector",
        frames=tuple(frames),
        window_start_ms=frames[0].timestamp_ms,
        window_ms=2000,
        target_offset=offset_for(label),
        radius=radius_for(label),
        game_score=300,
    )


class PlacementTests(unittest.TestCase):
    def test_targets_stay_within_reach_and_differ_by_punch(self) -> None:
        jab = offset_for(1)
        hook = offset_for(3)
        cross = offset_for(2)
        self.assertLess(jab[0], 0)
        self.assertGreater(cross[0], 0)
        self.assertLess(hook[0], jab[0])
        self.assertLess(abs(hook[1]), abs(jab[1]))
        for label in range(5):
            point = offset_for(label)
            self.assertLessEqual((point[0] ** 2 + point[1] ** 2) ** 0.5, MAX_REACH + 1e-9)

    def test_far_offset_is_clamped_before_it_leaves_the_body(self) -> None:
        clamped = clamp_offset(5.0, 0.0)
        self.assertAlmostEqual(clamped[0], MAX_REACH)
        left = (180.0, 120.0)
        right = (300.0, 120.0)
        projected = project_target(left, right, (5.0, 0.0), 0.4, 640, 480)
        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertGreaterEqual(projected[0], int(640 * 0.06))
        self.assertLess(projected[0], 640)

    def test_smoother_rejects_a_single_spike(self) -> None:
        smoother = PointSmoother(alpha=0.2)
        smoother.update((0.0, 0.0))
        moved = smoother.update((100.0, 0.0))
        self.assertAlmostEqual(moved[0], 20.0)
        wrist = body_offset((250.0, 120.0), (200.0, 120.0), (300.0, 120.0))
        self.assertIsNotNone(wrist)
        assert wrist is not None
        self.assertAlmostEqual(wrist[0], 0.0)


class ImpactAndGeneratorTests(unittest.TestCase):
    def test_circle_without_event_is_not_a_hit(self) -> None:
        target = offset_for(1)
        pending = judge_target(
            requested=1,
            event_present=False,
            event_label=None,
            event_key=None,
            consumed_key=None,
            wrist=target,
            target=target,
            radius=radius_for(1),
            tracking=True,
            elapsed_ms=200,
            limit_ms=2000,
        )
        self.assertEqual(pending.status, "pendiente")
        far = judge_target(
            requested=1,
            event_present=True,
            event_label=1,
            event_key="a",
            consumed_key=None,
            wrist=(0.0, 0.0),
            target=target,
            radius=radius_for(1),
            tracking=True,
            elapsed_ms=200,
            limit_ms=2000,
        )
        self.assertEqual(far.status, "lejos")

    def test_wrong_duplicate_guard_and_tracking(self) -> None:
        target = offset_for(2)
        wrong = judge_target(
            requested=2,
            event_present=True,
            event_label=4,
            event_key="b",
            consumed_key=None,
            wrist=target,
            target=target,
            radius=radius_for(2),
            tracking=True,
            elapsed_ms=100,
            limit_ms=2000,
        )
        duplicate = judge_target(
            requested=2,
            event_present=True,
            event_label=2,
            event_key="b",
            consumed_key="b",
            wrist=target,
            target=target,
            radius=radius_for(2),
            tracking=True,
            elapsed_ms=120,
            limit_ms=2000,
        )
        lost = judge_target(
            requested=2,
            event_present=False,
            event_label=None,
            event_key=None,
            consumed_key=None,
            wrist=None,
            target=target,
            radius=radius_for(2),
            tracking=False,
            elapsed_ms=5000,
            limit_ms=2000,
        )
        guard = judge_target(
            requested=0,
            event_present=False,
            event_label=None,
            event_key=None,
            consumed_key=None,
            wrist=None,
            target=offset_for(0),
            radius=0.0,
            tracking=True,
            elapsed_ms=2000,
            limit_ms=2000,
        )
        broken = judge_target(
            requested=0,
            event_present=True,
            event_label=1,
            event_key="c",
            consumed_key=None,
            wrist=None,
            target=offset_for(0),
            radius=0.0,
            tracking=True,
            elapsed_ms=400,
            limit_ms=2000,
        )
        self.assertEqual(wrong.status, "equivocado")
        self.assertEqual(duplicate.status, "duplicado")
        self.assertFalse(duplicate.failure)
        self.assertTrue(lost.pause)
        self.assertFalse(lost.failure)
        self.assertEqual(guard.status, "guardia")
        self.assertEqual(broken.status, "guardia_rota")

    def test_generator_covers_every_class_and_skips_a_completed_single(self) -> None:
        generator = TargetGenerator(goal=10, seed=2)
        seen: set[int] = set()
        for _ in range(40):
            seen.update(generator.choose_combo({label: 0 for label in range(5)}))
        self.assertEqual(seen, {0, 1, 2, 3, 4})
        balanced = [generator.next_label({label: 0 for label in range(5)}) for _ in range(15)]
        self.assertEqual([balanced.count(label) for label in range(5)], [3, 3, 3, 3, 3])
        completed = {0: 0, 1: 10, 2: 0, 3: 0, 4: 0}
        chosen = [generator.choose_combo(completed) for _ in range(40)]
        self.assertNotIn((1,), chosen)
        self.assertTrue(any(4 in combo for combo in chosen))


class ReviewAndRoundTests(unittest.TestCase):
    def test_provisional_sample_is_not_training_data_until_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            store = ReviewStore(root)
            store.ingest(_draft(1, "acierto", _frames(), detector=2), config, width=640, height=480)
            self.assertEqual(audit_dataset(root).valid_count, 0)
            payload = store.entries()[0][1]
            sample = payload["sample"]
            self.assertIsInstance(sample, dict)
            assert isinstance(sample, dict)
            self.assertEqual(sample["label_id"], 1)
            self.assertEqual(payload["detector_label_id"], 2)
            self.assertNotIn("game_score", sample)
            self.assertEqual(payload["game_score"], 300)
            self.assertEqual(store.approve_provisional(DatasetManager(root), config), 0)
            store.correct(store.entries()[0][0], 1)
            saved = store.approve(store.entries()[0][0], DatasetManager(root), config)
            self.assertTrue(saved)
            audit = audit_dataset(root)
            self.assertEqual(audit.valid_count, 1)
            self.assertEqual(len(extract_raw_features(audit.records[0].sample, 8)), vector_size(8))
            self.assertEqual(store.entries()[0][1]["detector_label_id"], 2)

    def test_correction_changes_label_without_rewriting_detector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            store = ReviewStore(root)
            store.ingest(_draft(1, "equivocado", _frames(), detector=4), config, width=640, height=480)
            path, payload = store.entries()[0]
            self.assertEqual(payload["verification_status"], "needs_review")
            self.assertEqual(store.approve_provisional(DatasetManager(root), config), 0)
            store.correct(path, 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["detector_label_id"], 4)
            self.assertTrue(store.approve(path, DatasetManager(root), config))
            sample = audit_dataset(root).records[0].sample
            self.assertEqual(sample.label_id, 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["detector_label_id"], 4)

    def test_short_window_is_invalid_and_sessions_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            store = ReviewStore(root)
            store.ingest(_draft(0, "guardia", _frames(count=3), detector=None), config, width=640, height=480)
            path, payload = store.entries()[0]
            self.assertEqual(payload["verification_status"], "invalid")
            self.assertFalse(store.approve(path, DatasetManager(root), config))
            dataset = DatasetManager(root)
            groups = {}
            for _ in range(3):
                other = ReviewStore(root)
                other.ingest(_draft(0, "guardia", _frames(), None), config, width=640, height=480)
                sample_path, _ = other.entries()[0]
                other.correct(sample_path, 0)
                self.assertTrue(other.approve(sample_path, dataset, config))
                groups[other.session_id] = other.session_id
            self.assertEqual(len(groups), 3)
            samples = {
                record.sample.sample_id: record.sample.session_id
                for record in audit_dataset(root).records
            }
            splits = split_by_group(samples, seed=42, group_kind="session")
            assigned = set(splits.train_ids) | set(splits.validation_ids) | set(splits.test_ids)
            self.assertEqual(assigned, set(samples))
            self.assertFalse(set(splits.train_ids) & set(splits.test_ids))

    def test_round_timers_tracking_and_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            session = TargetTraining(
                review=ReviewStore(root),
                config=config,
                round_size=1,
                countdown_ms=1000,
                active_ms=2000,
                gap_ms=100,
            )
            session.handle_key(ord(" "), 0)
            self.assertEqual(session.phase, "countdown")
            session.tick(0, tracking=False, event_present=False, event_label=None, event_key=None, event_name="", candidate=None, wrist=None, frame=None)
            session.tick(5000, tracking=False, event_present=False, event_label=None, event_key=None, event_name="", candidate=None, wrist=None, frame=None)
            self.assertEqual(session.phase, "countdown")
            self.assertEqual(session.misses, 0)
            session.phase = "active"
            session.current_label = 1
            session._elapsed = 100
            session._last_ms = 100
            session.tick(
                8000,
                tracking=False,
                event_present=False,
                event_label=None,
                event_key=None,
                event_name="",
                candidate=None,
                wrist=None,
                frame=None,
            )
            self.assertEqual(session.misses, 0)
            self.assertEqual(session._elapsed, 100)
            session.paused = True
            session.tracking = True
            session.tick(
                9000,
                tracking=True,
                event_present=False,
                event_label=None,
                event_key=None,
                event_name="",
                candidate=None,
                wrist=None,
                frame=None,
            )
            self.assertEqual(session._elapsed, 100)
            session.paused = False
            session.phase = "active"
            session.current_label = 0
            session._elapsed = 0
            session._last_ms = 0
            session._window_start = 0
            session._locked = False
            for frame in _frames():
                session.tick(
                    frame.timestamp_ms,
                    tracking=True,
                    event_present=False,
                    event_label=None,
                    event_key=None,
                    event_name="",
                    candidate=None,
                    wrist=None,
                    frame=frame,
                )
            self.assertEqual(session.hits, 1)
            self.assertEqual(session.review.counts()["needs_review"], 1)
            session.phase = "active"
            session.handle_key(ord("r"), 10_000)
            session.handle_key(ord("r"), 10_100)
            self.assertEqual(session.phase, "idle")
            self.assertEqual(session.score, 0)

    def test_saved_session_can_be_reviewed_without_new_punches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            original = ReviewStore(root)
            original.ingest(_draft(1, "equivocado", _frames(), detector=4), config, width=640, height=480)
            before = list((root / "review" / original.session_id).glob("*.json"))
            found = list_review_sessions(root)
            self.assertEqual(found[0]["session_id"], original.session_id)
            self.assertEqual(found[0]["needs_review"], 1)
            reopened = open_review_session(root, original.session_id)
            session = TargetTraining(review=reopened, config=config)
            session.phase = "review"
            session.handle_key(ord("g"), 0)
            self.assertEqual(audit_dataset(root).valid_count, 0)
            session.handle_key(ord("1"), 0)
            session.handle_key(ord("g"), 0)
            audit = audit_dataset(root)
            self.assertEqual(audit.valid_count, 1)
            self.assertEqual(audit.records[0].sample.label_id, 1)
            stored = json.loads(before[0].read_text(encoding="utf-8"))
            self.assertEqual(stored["detector_label_id"], 4)
            self.assertEqual(stored["verification_status"], "verified")
            self.assertFalse(session.review.approve(before[0], DatasetManager(root), config))
            self.assertEqual(len(DatasetManager(root).load_index()), 1)
            self.assertIn("Faltan sesiones", " ".join(dataset_gap_lines(root, 10)))
            self.assertEqual(len(list((root / "review" / original.session_id).glob("*.json"))), 1)
            self.assertIn("clase distinta", reason_text("equivocado"))
            self.assertIn("fuera del objetivo", reason_text("lejos"))
            self.assertIn("no reconocido", reason_text("fallo").lower())

    def test_hud_renders_without_claiming_model_accuracy(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        view = TargetView(
            phase="active",
            title="EN ENTRENAMIENTO",
            combo_text="Jab izquierdo > Cross derecho",
            countdown_s=None,
            remaining_ms=1200,
            score=100,
            hits=1,
            misses=0,
            tracking=True,
            status_line="Candidato",
            progress_lines=("Guardia / Neutral: 0/10",),
            review_lines=(),
            controls="ESPACIO inicia o pausa",
            target_name="Jab izquierdo",
            target_color=(50, 210, 255),
            screen_xy=(200, 140),
            screen_radius=28,
            fade=0.0,
            show_ring=True,
        )
        output = render_frame(
            image,
            PoseObservation(timestamp_ms=0),
            BiomechanicalFeatures(0, None, None, None, None),
            None,
            AppConfig(),
            targets=view,
        )
        self.assertEqual(output.shape, image.shape)


def _quiet_tick(session: TargetTraining, timestamp: int, frame: FrameRecord | None = None, **extra: object) -> None:
    values = {
        "tracking": True,
        "event_present": False,
        "event_label": None,
        "event_key": None,
        "event_name": "",
        "candidate": None,
        "wrist": None,
        "frame": frame,
    }
    values.update(extra)
    session.tick(timestamp, **values)  # type: ignore[arg-type]


def _complete_open_attempt(session: TargetTraining, opened_at: int, event_label: int | None) -> None:
    for step in range(21):
        timestamp = opened_at + 100 + step * 100
        event = step == 4 and event_label is not None
        label = session.current_label
        _quiet_tick(
            session,
            timestamp,
            frame=_frame(timestamp),
            event_present=event,
            event_label=event_label if event else None,
            event_key="evento" if event else None,
            event_name="Gancho izquierdo" if event else "",
            wrist=None if label in (None, 0) or not event else offset_for(label),
        )


class BlockedTrainingTests(unittest.TestCase):
    def test_blocks_follow_ten_attempts_of_each_class(self) -> None:
        plan = BlockPlan(10)
        self.assertEqual(plan.labels(), tuple(label for label in BLOCK_LABELS for _ in range(10)))
        self.assertEqual(plan.labels()[:10], (1,) * 10)
        self.assertEqual(plan.labels()[-10:], (0,) * 10)
        for _ in range(9):
            self.assertEqual(plan.record_attempt(), "continue")
        self.assertEqual(plan.record_attempt(), "next")
        plan.advance()
        self.assertEqual(plan.label, 2)

    def test_blocked_session_changes_class_without_verifying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            first = ReviewStore(root)
            second = ReviewStore(root)
            self.assertNotEqual(first.session_id, second.session_id)
            session = TargetTraining(
                review=first,
                config=config,
                goal=1,
                countdown_ms=1000,
                active_ms=2000,
                gap_ms=100,
                order="blocked",
            )
            view = session.view(None, 0)
            self.assertIn("BLOQUE 1/5", view.combo_text)
            self.assertIn("PROGRESO: 0/1", view.progress_lines)
            session.handle_key(ord(" "), 0)
            _quiet_tick(session, 0)
            _quiet_tick(session, 1000)
            self.assertEqual(session.current_label, 1)
            _quiet_tick(session, 1100)
            _quiet_tick(session, 1300)
            self.assertEqual(session._elapsed, 200)
            session.handle_key(ord(" "), 1300)
            _quiet_tick(session, 4000)
            self.assertEqual(session._elapsed, 200)
            self.assertTrue(session.paused)
            session.handle_key(ord(" "), 4000)
            session.handle_key(ord("s"), 4100)
            self.assertEqual(session.attempts, 0)
            self.assertEqual(first.counts()["needs_review"], 0)
            _complete_open_attempt(session, 4100, 3)
            self.assertEqual(session.phase, "ready")
            self.assertIn("SIGUE: CROSS DERECHO", session.message)
            self.assertEqual(session.attempts, 1)
            self.assertEqual(first.verified_counts()[1], 0)
            payload = first.entries()[0][1]
            self.assertEqual(payload["requested_label_id"], 1)
            self.assertEqual(payload["detector_label_id"], 3)
            self.assertEqual(payload["verification_status"], "needs_review")
            self.assertEqual(audit_dataset(root).valid_count, 0)
            self.assertIn("SIGUE: CROSS DERECHO", session.view(None, 0).combo_text)
            _quiet_tick(session, 20000)
            self.assertEqual(session.phase, "ready")
            session.handle_key(ord(" "), 21000)
            self.assertEqual(session.phase, "active")
            self.assertEqual(session.current_label, 2)
            self.assertFalse(session.handle_key(ord("q"), 22000))
            self.assertIn("Guardia", session.message)
            session.phase = "review"
            session.handle_key(ord("g"), 23000)
            self.assertEqual(audit_dataset(root).valid_count, 0)
            session.handle_key(ord("1"), 23000)
            session.handle_key(ord("g"), 23000)
            audit = audit_dataset(root)
            self.assertEqual(audit.valid_count, 1)
            self.assertEqual(audit.records[0].sample.label_id, 1)
            self.assertEqual(first.entries()[0][1]["detector_label_id"], 3)
            self.assertEqual(len(extract_raw_features(audit.records[0].sample, 8)), vector_size(8))
            self.assertEqual(len(list((root / "review" / second.session_id).glob("*.json"))), 0)

    def test_ten_neutral_guards_need_no_punch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            store = ReviewStore(root)
            session = TargetTraining(
                review=store,
                config=config,
                goal=10,
                countdown_ms=1000,
                active_ms=2000,
                order="blocked",
                only_class=0,
            )
            session.handle_key(ord(" "), 0)
            _quiet_tick(session, 0)
            _quiet_tick(session, 1000)
            clock = 1000
            for index in range(10):
                self.assertEqual(session.current_label, 0)
                self.assertFalse(session.view(None, 0).show_ring)
                _complete_open_attempt(session, clock, None)
                if index < 9:
                    self.assertEqual(session.phase, "ready")
                    clock += 5000
                    session.handle_key(ord(" "), clock)
            self.assertEqual(session.phase, "summary")
            self.assertEqual(store.counts()["needs_review"], 10)
            self.assertEqual(store.counts()["invalid"], 0)
            self.assertTrue(all(item[1]["requested_label_id"] == 0 for item in store.entries()))
            self.assertTrue(all(item[1]["outcome"] == "guardia" for item in store.entries()))
            self.assertEqual(audit_dataset(root).valid_count, 0)
            session.handle_key(ord(" "), clock + 1000)
            session.handle_key(ord("g"), clock + 1000)
            self.assertEqual(audit_dataset(root).valid_count, 0)
            session.handle_key(ord("0"), clock + 1000)
            session.handle_key(ord("g"), clock + 1000)
            audit = audit_dataset(root)
            self.assertEqual(audit.valid_count, 1)
            self.assertEqual(audit.records[0].sample.label_id, 0)

    def test_full_block_order_reaches_guard_and_can_repeat_a_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            store = ReviewStore(root)
            session = TargetTraining(
                review=store,
                config=config,
                goal=1,
                countdown_ms=1000,
                active_ms=2000,
                order="blocked",
            )
            session.handle_key(ord(" "), 0)
            _quiet_tick(session, 0)
            _quiet_tick(session, 1000)
            clock = 1000
            for expected in (1, 2, 3, 4, 0):
                self.assertEqual(session.current_label, expected)
                _complete_open_attempt(session, clock, None)
                if expected != 0:
                    self.assertIn("SIGUE:", session.message)
                    clock += 5000
                    session.handle_key(ord(" "), clock)
            self.assertEqual(session.phase, "summary")
            self.assertEqual(sorted(item[1]["requested_label_id"] for item in store.entries()), [0, 1, 2, 3, 4])
            text = " ".join(session.view(None, 0).review_lines)
            self.assertIn("Guardia", text)
            self.assertIn("intentos", text.lower())
            solo = TargetTraining(
                review=ReviewStore(root),
                config=config,
                goal=1,
                countdown_ms=1000,
                active_ms=2000,
                order="blocked",
                only_class=4,
            )
            self.assertIn("BLOQUE 1/1", solo.view(None, 0).combo_text)
            self.assertEqual(solo.block.label, 4)
            rejected = TargetTraining(
                review=ReviewStore(root),
                config=config,
                goal=1,
                countdown_ms=1000,
                active_ms=2000,
                order="blocked",
                only_class=1,
            )
            rejected.handle_key(ord(" "), 0)
            _quiet_tick(rejected, 0)
            _quiet_tick(rejected, 1000)
            _quiet_tick(rejected, 1100, frame=_frame(1100))
            _quiet_tick(rejected, 1200, frame=_frame(1200))
            _quiet_tick(rejected, 3200, frame=_frame(3200))
            self.assertEqual(rejected.phase, "summary")
            self.assertEqual(rejected.review.counts()["invalid"], 1)
            self.assertIn("Faltan", " ".join(rejected.view(None, 0).review_lines))
            rejected.handle_key(ord("p"), 4000)
            self.assertEqual(rejected.phase, "ready")
            self.assertEqual(rejected.current_label, 1)
            self.assertEqual(rejected.review.session_id != store.session_id, True)


class GuardLabelTests(unittest.TestCase):
    def test_zero_is_saved_and_other_labels_do_not_regress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            store = ReviewStore(root)
            for label in range(5):
                store.ingest(_draft(label, "guardia" if label == 0 else "fallo", _frames(), detector=None), config, width=640, height=480)
            session = TargetTraining(review=store, config=config)
            session.phase = "review"
            session._load_player()
            session.handle_key(ord("g"), 0)
            self.assertEqual(audit_dataset(root).valid_count, 0)
            self.assertIsNone(session._label_choice)
            self.assertIn("no guardó", session.message.lower())
            session.handle_key(0, 0)
            self.assertEqual(session._label_choice, 0)
            self.assertIn("Etiqueta 0", " ".join(session.view(None, 0).review_lines))
            session.handle_key(ord("g"), 0)
            audit = audit_dataset(root)
            self.assertEqual(audit.class_counts[0], 1)
            self.assertEqual(audit.records[0].sample.label_name, "guardia_neutral")
            self.assertTrue(any((root / "samples").glob("*/*.json")))
            index = DatasetManager(root).load_index()
            self.assertEqual(index[0]["label_id"], "0")
            saved_path = next(path for path, payload in store.entries() if payload["verification_status"] == "verified")
            self.assertFalse(session.review.approve(saved_path, DatasetManager(root), config))
            self.assertEqual(len(DatasetManager(root).load_index()), 1)
            for label in (1, 2, 3, 4):
                session.handle_key(ord(str(label)), 0)
                session.handle_key(ord("g"), 0)
            audit = audit_dataset(root)
            self.assertEqual([audit.class_counts[label] for label in range(5)], [1, 1, 1, 1, 1])
            self.assertEqual(len(DatasetManager(root).load_index()), 5)

    def test_verified_mismatch_changes_only_after_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            store = ReviewStore(root)
            store.ingest(_draft(0, "guardia", _frames(), detector=None), config, width=640, height=480)
            path = store.entries()[0][0]
            store.correct(path, 1)
            self.assertTrue(store.approve(path, DatasetManager(root), config))
            session = TargetTraining(review=store, config=config, review_verified=True)
            session.phase = "review"
            session._load_player()
            session.handle_key(ord("g"), 0)
            self.assertEqual(DatasetManager(root).load_index()[0]["label_id"], "1")
            session.handle_key(0, 0)
            session.handle_key(ord("g"), 0)
            index = DatasetManager(root).load_index()
            self.assertEqual(len(index), 1)
            self.assertEqual(index[0]["label_id"], "0")
            self.assertEqual(index[0]["label_name"], "guardia_neutral")
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["requested_label_id"], 0)
            self.assertEqual(stored["approved_label_id"], 0)
            self.assertEqual(stored["sample"]["sample_id"], index[0]["sample_id"])


class PlaybackTests(unittest.TestCase):
    def test_player_keeps_timestamps_and_skips_missing_joints(self) -> None:
        from app.targets.playback import STORED_JOINTS, SkeletonPlayer

        frames = []
        for index, timestamp in enumerate((0, 100, 250)):
            landmarks = {
                name: {"x": 0.2, "y": 0.4, "z": 0.0, "visibility": 1.0, "presence": 1.0, "reliable": True}
                for name in STORED_JOINTS
            }
            landmarks["left_wrist"] = None
            frames.append(
                {
                    "timestamp_ms": timestamp,
                    "elapsed_ms": timestamp,
                    "landmarks": landmarks,
                    "features": {"left_elbow_angle": 80.0, "right_elbow_angle": None},
                }
            )
        player = SkeletonPlayer(frames, mirror=True)
        snapshot = player.snapshot()
        names = [name for name, _x, _y in snapshot.points]
        self.assertNotIn("left_wrist", names)
        self.assertNotIn("nose", names)
        self.assertEqual(len(snapshot.points), 7)
        self.assertAlmostEqual(snapshot.points[0][1], 0.8)
        player.toggle()
        player.advance(100)
        self.assertEqual(player.index, 1)
        player.toggle()
        player.advance(1000)
        self.assertEqual(player.index, 1)
        player.step(1)
        self.assertEqual(player.index, 2)
        broken = SkeletonPlayer(
            [{"timestamp_ms": 50, "landmarks": {}}, {"timestamp_ms": 40, "landmarks": {}}],
        )
        broken.toggle()
        broken.advance(30)
        self.assertIn("timestamps", broken.error)

    def test_failed_detection_stays_pending_and_discard_does_not_train(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = AppConfig(data_dir=root, mode="targets")
            store = ReviewStore(root)
            store.ingest(_draft(0, "fallo", _frames(), detector=None), config, width=640, height=480)
            path, payload = store.entries()[0]
            self.assertEqual(payload["verification_status"], "needs_review")
            self.assertEqual(audit_dataset(root).valid_count, 0)
            session = TargetTraining(review=store, config=config)
            session.phase = "review"
            session._load_player()
            assert session.player is not None
            names = [name for name, _x, _y in session.player.snapshot().points]
            self.assertIn("left_elbow", names)
            self.assertNotIn("nose", names)
            session.handle_key(ord("x"), 0)
            self.assertEqual(audit_dataset(root).valid_count, 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["verification_status"], "discarded")
            self.assertIsNone(session.player)


if __name__ == "__main__":
    unittest.main()
