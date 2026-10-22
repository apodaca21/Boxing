"""Un golpe físico confirma una sola clase. No mide precisión en webcam."""

from __future__ import annotations

import unittest

from app.game.challenge import TargetChallenge
from app.game.free_round import FreeRound
from app.targets.placement import offset_for
from app.trainer.counting import PunchLedger, commit_punch_event
from app.trainer.heuristic import HeuristicClassifier
from app.trainer.results import ClassificationResult, PunchEvent
from app.trainer.sensitivity import profile_settings
from tests.test_game import _guard, _punch, _start, _uncertain
from tests.test_trainer import _features, _feed, _hook_cycle, _straight_cycle


def _event(
    label: int,
    name: str,
    arm: str,
    *,
    event_id: str,
    confirmed: bool = True,
) -> PunchEvent:
    return PunchEvent(
        label,
        name,
        arm,
        event_id=event_id,
        started_ms=0,
        ended_ms=100,
        confirmed=confirmed,
        source="test",
    )


class PunchLedgerTests(unittest.TestCase):
    def test_same_event_id_counts_once_even_if_class_changes(self) -> None:
        ledger = PunchLedger()
        first = _event(1, "Jab izquierdo", "left", event_id="left:10")
        second = _event(3, "Gancho izquierdo", "left", event_id="left:10")
        self.assertTrue(commit_punch_event(first, ledger))
        self.assertFalse(commit_punch_event(second, ledger))
        self.assertEqual(ledger.counts[1], 1)
        self.assertEqual(ledger.counts[3], 0)
        self.assertEqual(ledger.total, 1)
        self.assertEqual(ledger.total, ledger.category_sum)

    def test_unconfirmed_and_guard_do_not_count(self) -> None:
        ledger = PunchLedger()
        self.assertFalse(
            commit_punch_event(
                PunchEvent(1, "Jab izquierdo", "left", event_id="x", confirmed=False),
                ledger,
            )
        )
        self.assertFalse(
            commit_punch_event(PunchEvent(0, "Guardia / Neutral", None, event_id="g"), ledger)
        )
        self.assertEqual(ledger.total, 0)

    def test_ambiguous_event_does_not_count(self) -> None:
        ledger = PunchLedger()
        self.assertFalse(
            commit_punch_event(
                PunchEvent(None, "MOVIMIENTO NO CONFIRMADO", "left", event_id="amb"),
                ledger,
            )
        )
        self.assertEqual(ledger.total, 0)


class FreeRoundCommitTests(unittest.TestCase):
    def test_candidate_frames_do_not_score(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        candidate = ClassificationResult(
            source="heuristic",
            status="extendiendo",
            label_id=None,
            display_name="Jab izquierdo",
            candidate="Jab izquierdo",
            moving=True,
        )
        for index in range(6):
            game.tick(4000 + index * 30, candidate)
        game.tick(4300, _guard())
        view = game.snapshot()
        self.assertEqual(view.total, 0)
        self.assertEqual(sum(view.counts.values()), 0)

    def test_jab_then_hook_candidates_commit_the_final_event(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        game.tick(4000, ClassificationResult("heuristic", "extendiendo", None, "Jab izquierdo", moving=True))
        game.tick(4030, ClassificationResult("heuristic", "extendiendo", None, "Gancho izquierdo", moving=True))
        game.tick(
            4100,
            ClassificationResult(
                "heuristic",
                "completado",
                3,
                "Gancho izquierdo",
                events=(_event(3, "Gancho izquierdo", "left", event_id="left:4000"),),
            ),
        )
        view = game.snapshot()
        self.assertEqual(view.counts[3], 1)
        self.assertEqual(view.counts[1], 0)
        self.assertEqual(view.total, 1)

    def test_repeated_event_id_does_not_score_again(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        event = _event(2, "Cross derecho", "right", event_id="right:1")
        punch = ClassificationResult("heuristic", "completado", 2, "Cross derecho", events=(event,))
        game.tick(4000, punch)
        game.tick(4030, punch)
        self.assertEqual(game.snapshot().counts[2], 1)
        self.assertEqual(game.snapshot().total, 1)

    def test_ten_frames_of_the_same_punch_count_once(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        punch = _punch(1, "Jab izquierdo", "left")
        for index in range(10):
            game.tick(4000 + index * 30, punch)
        self.assertEqual(game.snapshot().total, 1)

    def test_two_real_jabs_and_jab_then_cross(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        game.tick(4000, _punch(1, "Jab izquierdo", "left"))
        game.tick(4300, _guard())
        game.tick(4600, _punch(1, "Jab izquierdo", "left"))
        game.tick(4900, _guard())
        game.tick(5200, _punch(2, "Cross derecho", "right"))
        view = game.snapshot()
        self.assertEqual(view.counts[1], 2)
        self.assertEqual(view.counts[2], 1)
        self.assertEqual(view.total, 3)
        self.assertEqual(view.total, sum(view.counts.values()))

    def test_uncertain_and_guard_do_not_increment(self) -> None:
        game = FreeRound("heuristic", duration_ms=90_000, countdown_ms=1000)
        _start(game)
        game.tick(4000, _uncertain())
        game.tick(4300, _guard())
        self.assertEqual(game.snapshot().total, 0)


class DetectorCycleTests(unittest.TestCase):
    def test_hook_that_passes_through_a_jab_candidate_counts_once(self) -> None:
        classifier = HeuristicClassifier(profile_settings("play"))
        frames = [_features(0), _features(40)]
        path = (
            (80, 1.05, -0.32, 100),
            (120, 1.18, -0.40, 108),
            (160, 1.22, -0.58, 112),
            (200, 1.20, -0.82, 108),
            (240, 1.18, -0.95, 102),
            (280, 1.10, -0.70, 100),
            (320, 1.00, -0.40, 95),
            (360, 0.92, -0.30, 90),
            (400, 0.90, -0.30, 90),
        )
        for timestamp, extension, x, elbow in path:
            frames.append(
                _features(
                    timestamp,
                    left_extension=extension,
                    left_x=x,
                    left_elbow=elbow,
                )
            )
        results = _feed(classifier, frames)
        labels = [event.label_id for result in results for event in result.events]
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0], 3)

    def test_completed_cross_and_right_hook_are_single_events(self) -> None:
        cross = _feed(
            HeuristicClassifier(profile_settings("play")),
            _straight_cycle(0, side="right"),
        )
        hook = _feed(
            HeuristicClassifier(profile_settings("play")),
            _hook_cycle(0, side="right"),
        )
        self.assertEqual([event.label_id for result in cross for event in result.events], [2])
        self.assertEqual([event.label_id for result in hook for event in result.events], [4])

    def test_weaker_other_arm_is_not_a_second_punch(self) -> None:
        classifier = HeuristicClassifier(profile_settings("play"))
        frames = []
        left = (0.90, 0.90, 1.20, 1.55, 1.90, 2.10, 1.70, 1.30, 1.00, 0.90)
        right = (0.90, 0.92, 0.96, 0.98, 0.97, 0.95, 0.93, 0.91, 0.90, 0.90)
        for index, (left_ext, right_ext) in enumerate(zip(left, right, strict=True)):
            frames.append(
                _features(
                    index * 40,
                    left_extension=left_ext,
                    left_elbow=90 + index * 8,
                    right_extension=right_ext,
                    right_elbow=90,
                )
            )
        results = _feed(classifier, frames)
        labels = [event.label_id for result in results for event in result.events]
        self.assertEqual(labels, [1])

    def test_guard_return_does_not_add_a_second_event(self) -> None:
        results = _feed(
            HeuristicClassifier(profile_settings("play")),
            _straight_cycle(0, side="left"),
        )
        self.assertEqual(sum(len(result.events) for result in results), 1)


class TargetSingleHitTests(unittest.TestCase):
    def test_same_event_id_cannot_score_two_targets(self) -> None:
        game = TargetChallenge("heuristic", difficulty="easy", duration_ms=30_000, countdown_ms=1000)
        game.handle_key(ord(" "))
        game.tick(0, tracking=True, event_present=False, event_label=None, event_key=None, wrist=None, uncertain=False)
        game.tick(1000, tracking=True, event_present=False, event_label=None, event_key=None, wrist=None, uncertain=False)
        label = game.label_id
        assert label is not None
        kwargs = {
            "tracking": True,
            "event_present": True,
            "event_label": label,
            "event_key": "once",
            "wrist": offset_for(label),
            "uncertain": False,
            "event_id": "left:99",
        }
        game.tick(1200, **kwargs)
        first_hits = game.hits
        game.tick(1400, tracking=True, event_present=False, event_label=None, event_key=None, wrist=None, uncertain=False)
        game.tick(1600, **kwargs)
        self.assertEqual(game.hits, first_hits)


if __name__ == "__main__":
    unittest.main()
