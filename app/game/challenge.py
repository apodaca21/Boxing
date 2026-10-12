"""Objetivos circulares. Un acierto exige evento, clase y muñeca dentro del círculo."""

from __future__ import annotations

from dataclasses import dataclass

from app.dataset_manager import MOVEMENTS
from app.targets.impact import judge_target
from app.targets.placement import offset_for, radius_for


DIFFICULTIES = {
    "easy": {
        "scale": 1.35,
        "limit_ms": 2800,
        "combos": ((1,), (2,), (3,), (4,)),
    },
    "normal": {
        "scale": 1.0,
        "limit_ms": 1800,
        "combos": ((1, 2), (2, 3), (1, 4), (3, 4)),
    },
    "hard": {
        "scale": 0.72,
        "limit_ms": 1100,
        "combos": ((1, 2, 3), (1, 1, 2), (2, 4, 3)),
    },
}

TARGET_STYLE = {
    1: ("JAB", (235, 150, 40)),
    2: ("CROSS", (40, 150, 245)),
    3: ("GANCHO IZQ", (200, 70, 180)),
    4: ("GANCHO DER", (50, 50, 220)),
}


@dataclass(frozen=True, slots=True)
class ChallengeSnapshot:
    phase: str
    label_id: int | None
    title: str
    progress: str
    score: int
    hits: int
    misses: int
    combo: int
    best_combo: int
    remaining_ms: int
    limit_ms: int
    message: str
    difficulty: str
    classifier: str
    radius_scale: float
    flash: bool


class TargetChallenge:
    def __init__(
        self,
        classifier_name: str,
        *,
        difficulty: str = "normal",
        duration_ms: int = 60_000,
        countdown_ms: int = 3_000,
    ) -> None:
        if difficulty not in DIFFICULTIES:
            raise ValueError("La dificultad debe ser easy, normal o hard.")
        self.classifier_name = classifier_name
        self.difficulty = difficulty
        self.duration_ms = duration_ms
        self.countdown_ms = countdown_ms
        preset = DIFFICULTIES[difficulty]
        self.scale = float(preset["scale"])
        self.limit_ms = int(preset["limit_ms"])
        self._combos = tuple(tuple(item) for item in preset["combos"])  # type: ignore[union-attr]
        self.phase = "idle"
        self.combo_index = 0
        self.step = 0
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.combo = 0
        self.best_combo = 0
        self.message = "ESPACIO PARA INICIAR"
        self._elapsed = 0
        self._attempt_ms = 0
        self._mark_ms: int | None = None
        self._consumed: str | None = None
        self._flash_until = -1

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return True
        if key == ord(" ") and self.phase == "idle":
            self.phase = "countdown"
            self._mark_ms = None
            self._elapsed = 0
            self.message = "3"
        elif key in (ord("r"), ord("R")) and self.phase == "summary":
            self.__init__(
                self.classifier_name,
                difficulty=self.difficulty,
                duration_ms=self.duration_ms,
                countdown_ms=self.countdown_ms,
            )
        return False

    @property
    def label_id(self) -> int | None:
        if self.phase != "active":
            return None
        return self._combos[self.combo_index][self.step]

    def tick(
        self,
        now_ms: int,
        *,
        tracking: bool,
        event_present: bool,
        event_label: int | None,
        event_key: str | None,
        wrist: tuple[float, float] | None,
        uncertain: bool,
    ) -> None:
        if self.phase in {"idle", "summary"}:
            return
        if self._mark_ms is None:
            self._mark_ms = now_ms
        delta = max(0, now_ms - self._mark_ms)
        self._mark_ms = now_ms
        if self.phase == "countdown":
            self._elapsed += delta
            left = self.countdown_ms - self._elapsed
            self.message = str(max(1, (left + 999) // 1000))
            if self._elapsed >= self.countdown_ms:
                self.phase = "active"
                self._elapsed = 0
                self._attempt_ms = 0
                self._mark_ms = now_ms
                self.message = self._title()
            return
        self._elapsed += delta
        if self._elapsed >= self.duration_ms:
            self.phase = "summary"
            self.message = "Desafio terminado"
            return
        if not tracking:
            self.message = "Sin seguimiento"
            return
        if not event_present:
            self._consumed = None
        self._attempt_ms += delta
        if uncertain and not event_present:
            self.message = "MOVIMIENTO NO CONFIRMADO"
            return
        label = self.label_id
        assert label is not None
        judgement = judge_target(
            requested=label,
            event_present=event_present,
            event_label=event_label,
            event_key=event_key,
            consumed_key=self._consumed,
            wrist=wrist,
            target=offset_for(label),
            radius=radius_for(label) * self.scale,
            tracking=True,
            elapsed_ms=self._attempt_ms,
            limit_ms=self.limit_ms,
        )
        if event_present and event_key is not None:
            self._consumed = event_key
        if judgement.status == "acierto":
            self._hit(now_ms)
        elif judgement.status == "equivocado":
            self.combo = 0
            self.misses += 1
            self.message = "GOLPE INCORRECTO"
        elif judgement.status == "lejos":
            self.message = "FUERA DEL OBJETIVO"
        elif judgement.status == "fallo":
            self.combo = 0
            self.misses += 1
            self.message = "TIEMPO"
            self._advance(now_ms)
        elif judgement.status == "no_identificado":
            self.message = "MOVIMIENTO NO CONFIRMADO"

    def _hit(self, now_ms: int) -> None:
        self.hits += 1
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo)
        self.score += 100 * self.combo
        self.message = "ACIERTO"
        self._flash_until = now_ms + 220
        self._advance(now_ms)

    def _advance(self, _now_ms: int) -> None:
        combo = self._combos[self.combo_index]
        self.step += 1
        self._attempt_ms = 0
        if self.step >= len(combo):
            self.step = 0
            self.combo_index = (self.combo_index + 1) % len(self._combos)

    def _title(self) -> str:
        label = self.label_id
        if label is None:
            return ""
        return TARGET_STYLE[label][0]

    def snapshot(self, now_ms: int = 0) -> ChallengeSnapshot:
        remaining = 0
        if self.phase == "active":
            remaining = max(0, self.duration_ms - self._elapsed)
        combo = self._combos[self.combo_index]
        return ChallengeSnapshot(
            phase=self.phase,
            label_id=self.label_id,
            title=self._title(),
            progress=f"{self.step + 1}/{len(combo)}" if self.phase == "active" else "—",
            score=self.score,
            hits=self.hits,
            misses=self.misses,
            combo=self.combo,
            best_combo=self.best_combo,
            remaining_ms=remaining,
            limit_ms=self.limit_ms,
            message=self.message,
            difficulty=self.difficulty,
            classifier=self.classifier_name,
            radius_scale=self.scale,
            flash=now_ms <= self._flash_until,
        )

    def record(self) -> dict[str, object]:
        view = self.snapshot()
        return {
            "game": "targets",
            "classifier": self.classifier_name,
            "difficulty": self.difficulty,
            "score": view.score,
            "score_is_not_neat_accuracy": True,
            "contains_training_samples": False,
            "hits": view.hits,
            "misses": view.misses,
            "best_combo": view.best_combo,
        }
