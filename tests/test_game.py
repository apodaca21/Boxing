"""Lógica de juego y conteo. No afirma precisión con golpes reales."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from app.evolution.features import vector_size
from app.game.challenge import TargetChallenge
from app.game.free_round import FreeRound
from app.game.store import save_round
from app.geometry import BiomechanicalFeatures, RelativePoint
from app.targets.placement import display_point, offset_for, project_target, radius_for
from app.trainer.neat_classifier import NeatClassifier
from app.trainer.results import ClassificationResult, PunchEvent
from app.evolution.model_store import load_model
from main import parse_args


def _punch(label: int, name: str, arm: str) -> ClassificationResult:
    return ClassificationResult(
        source="test",
        status="completado",
        label_id=label,
        display_name=name,
        events=(PunchEvent(label, name, arm),),
        active_arm=arm,
    )


def _guard() -> ClassificationResult:
    return ClassificationResult(
        source="test",
        status="neutral",
        label_id=0,
        display_name="Guardia / Neutral",
    )


def _uncertain() -> ClassificationResult:
    return ClassificationResult(
        source="test",
        status="incierto",
        label_id=None,
        display_name="MOVIMIENTO NO CONFIRMADO",
    )


def _start(game: FreeRound) -> None:
    game.handle_key(ord(" "))
    game.tick(0, _guard())
    game.tick(3000, _guard())


class FreeRoundTests(unittest.TestCase):
    def test_one_punch_across_ten_frames_counts_once(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        punch = _punch(1, "Jab izquierdo", "left")
        for index in range(10):
            game.tick(4000 + index * 30, punch)
        self.assertEqual(game.snapshot().counts[1], 1)
        self.assertEqual(game.snapshot().total, 1)

    def test_two_distinct_jabs_count_twice(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        game.tick(4000, _punch(1, "Jab izquierdo", "left"))
        game.tick(4300, _guard())
        game.tick(4600, _punch(1, "Jab izquierdo", "left"))
        self.assertEqual(game.snapshot().counts[1], 2)

    def test_jab_and_cross_use_opposite_arms(self) -> None:
        game = FreeRound("neat", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        game.tick(4000, _punch(1, "Jab izquierdo", "left"))
        game.tick(4300, _punch(2, "Cross derecho", "right"))
        view = game.snapshot()
        self.assertEqual(view.counts[1], 1)
        self.assertEqual(view.counts[2], 1)
        self.assertIn("Jab", view.combo)
        self.assertIn("Cross", view.combo)

    def test_hooks_stay_on_their_side(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        game.tick(4000, _punch(3, "Gancho izquierdo", "left"))
        game.tick(4400, _punch(4, "Gancho derecho", "right"))
        self.assertEqual(game.snapshot().counts[3], 1)
        self.assertEqual(game.snapshot().counts[4], 1)

    def test_guard_and_uncertain_do_not_score(self) -> None:
        game = FreeRound("neat", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        game.tick(4000, _guard())
        game.tick(4300, _uncertain())
        view = game.snapshot()
        self.assertEqual(view.total, 0)
        self.assertEqual(view.uncertain, 1)
        self.assertEqual(view.combo, "—")

    def test_round_ends_and_saves_outside_the_dataset(self) -> None:
        game = FreeRound("heuristic", duration_ms=2000, countdown_ms=1000)
        _start(game)
        game.tick(5000, _punch(2, "Cross derecho", "right"))
        game.tick(8000, _guard())
        self.assertEqual(game.phase, "summary")
        with tempfile.TemporaryDirectory() as temporary:
            path = save_round(Path(temporary), game.record())
            self.assertTrue(path.is_file())
            self.assertNotIn("samples", path.parts)
            self.assertIn("score_is_not_neat_accuracy", path.read_text(encoding="utf-8"))


class TargetGameTests(unittest.TestCase):
    def _active(self, difficulty: str = "easy") -> TargetChallenge:
        game = TargetChallenge("heuristic", difficulty=difficulty, duration_ms=30_000, countdown_ms=1000)
        game.handle_key(ord(" "))
        game.tick(0, tracking=True, event_present=False, event_label=None, event_key=None, wrist=None, uncertain=False)
        game.tick(1000, tracking=True, event_present=False, event_label=None, event_key=None, wrist=None, uncertain=False)
        return game

    def test_hit_requires_class_and_wrist(self) -> None:
        game = self._active()
        label = game.label_id
        assert label is not None
        game.tick(
            1200,
            tracking=True,
            event_present=True,
            event_label=label,
            event_key="hit-1",
            wrist=offset_for(label),
            uncertain=False,
        )
        self.assertEqual(game.hits, 1)
        self.assertGreater(game.score, 0)

    def test_wrong_class_and_outside_circle_do_not_advance(self) -> None:
        game = self._active()
        label = game.label_id
        game.tick(
            1200,
            tracking=True,
            event_present=True,
            event_label=4 if label != 4 else 1,
            event_key="bad",
            wrist=offset_for(label or 1),
            uncertain=False,
        )
        self.assertEqual(game.message, "GOLPE INCORRECTO")
        self.assertEqual(game.label_id, label)
        game.tick(
            1400,
            tracking=True,
            event_present=False,
            event_label=None,
            event_key=None,
            wrist=None,
            uncertain=False,
        )
        game.tick(
            1600,
            tracking=True,
            event_present=True,
            event_label=label,
            event_key="far",
            wrist=(3.0, 3.0),
            uncertain=False,
        )
        self.assertEqual(game.message, "FUERA DEL OBJETIVO")
        self.assertEqual(game.hits, 0)
        self.assertEqual(game.label_id, label)

    def test_same_impact_cannot_score_twice(self) -> None:
        game = self._active()
        label = game.label_id
        assert label is not None
        kwargs = {
            "tracking": True,
            "event_present": True,
            "event_label": label,
            "event_key": "once",
            "wrist": offset_for(label),
            "uncertain": False,
        }
        game.tick(1200, **kwargs)
        game.tick(1250, **kwargs)
        self.assertEqual(game.hits, 1)

    def test_uncertain_does_not_hit(self) -> None:
        game = self._active()
        before = game.label_id
        game.tick(
            1200,
            tracking=True,
            event_present=False,
            event_label=None,
            event_key=None,
            wrist=offset_for(before or 1),
            uncertain=True,
        )
        self.assertEqual(game.message, "MOVIMIENTO NO CONFIRMADO")
        self.assertEqual(game.hits, 0)

    def test_targets_scale_with_body_and_stay_inside_the_frame(self) -> None:
        easy = TargetChallenge("heuristic", difficulty="easy")
        hard = TargetChallenge("heuristic", difficulty="hard")
        self.assertGreater(easy.scale, hard.scale)
        shoulders = ((200.0, 180.0), (360.0, 180.0))
        for width, height in ((640, 480), (1280, 720)):
            placed = project_target(
                shoulders[0],
                shoulders[1],
                offset_for(1),
                radius_for(1) * easy.scale,
                width,
                height,
            )
            assert placed is not None
            self.assertGreaterEqual(placed[0], 0)
            self.assertLess(placed[0], width)
            self.assertGreaterEqual(placed[1], 0)
            self.assertLess(placed[1], height)

    def test_mirror_flips_the_drawn_x(self) -> None:
        left = display_point(0.2, 0.4, 640, 480, mirror=False)
        right = display_point(0.2, 0.4, 640, 480, mirror=True)
        self.assertNotEqual(left[0], right[0])
        self.assertAlmostEqual(left[0] + right[0], 639)

    def test_round_reaches_summary(self) -> None:
        game = TargetChallenge("neat", difficulty="normal", duration_ms=1500, countdown_ms=500)
        game.handle_key(ord(" "))
        game.tick(0, tracking=True, event_present=False, event_label=None, event_key=None, wrist=None, uncertain=False)
        game.tick(500, tracking=True, event_present=False, event_label=None, event_key=None, wrist=None, uncertain=False)
        game.tick(2500, tracking=True, event_present=False, event_label=None, event_key=None, wrist=None, uncertain=False)
        self.assertEqual(game.phase, "summary")
        self.assertTrue(game.record()["score_is_not_neat_accuracy"])


class RecognitionGateTests(unittest.TestCase):
    def test_low_margin_is_not_a_punch_and_a_dip_allows_another(self) -> None:
        plan = iter((
            (1.0, 0.0),
            (0.40, 0.0),
            (1.0, 0.0),
            (0.55, 0.50),
        ))

        def transform(raw: tuple[float | None, ...]) -> tuple[float, ...]:
            return tuple(0.0 if value is None else float(value) for value in raw)

        def activate(_vector: tuple[float, ...]) -> tuple[float, ...]:
            jab, guard = next(plan)
            return (guard, jab, 0.0, 0.0, 0.0)

        classifier = NeatClassifier(transform, activate, time_steps=2, cooldown_ms=0, min_margin=0.08)
        frames = [
            BiomechanicalFeatures(
                timestamp_ms=index * 40,
                left_elbow_angle=90,
                right_elbow_angle=90,
                left_shoulder_wrist_distance=1.0,
                right_shoulder_wrist_distance=1.0,
                left_wrist_relative=RelativePoint(-0.2, 0.2),
                right_wrist_relative=RelativePoint(0.2, 0.2),
            )
            for index in range(5)
        ]
        results = [classifier.update(frame, pose_detected=True) for frame in frames]
        counted = [result for result in results if result.events]
        uncertain = [result for result in results if result.status == "incierto"]
        self.assertEqual(len(counted), 2)
        self.assertEqual(counted[0].events[0].arm, "left")
        self.assertGreaterEqual(len(uncertain), 1)
        self.assertEqual(vector_size(8), 92)

    def test_saved_neat_model_loads_without_rewriting_it(self) -> None:
        path = Path("results/20261012T001458_initial/model")
        if not (path / "winner.pkl").is_file():
            self.skipTest("el experimento principal no está en esta copia")
        before = (path / "winner.pkl").stat().st_mtime
        model = load_model(path)
        self.assertEqual(model.scaler.time_steps, 8)
        self.assertEqual(len(model.scaler.names), 92)
        self.assertEqual((path / "winner.pkl").stat().st_mtime, before)

    def test_previous_modes_still_parse(self) -> None:
        original = sys.argv
        try:
            for mode in ("track", "collect", "train", "live", "targets", "game"):
                sys.argv = ["main.py", "--mode", mode]
                self.assertEqual(parse_args().mode, mode)
            sys.argv = ["main.py", "--mode", "game", "--game-mode", "free", "--classifier", "neat"]
            args = parse_args()
            self.assertEqual(args.game_mode, "free")
            self.assertEqual(args.classifier, "neat")
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
