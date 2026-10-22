"""Ronda libre. Cuenta eventos completos y deja fuera guardia e incertidumbre."""

from __future__ import annotations

from dataclasses import dataclass

from app.dataset_manager import MOVEMENTS
from app.game.combo import ComboTracker
from app.trainer.counting import PunchLedger, rows_for
from app.trainer.results import ClassificationResult


@dataclass(frozen=True, slots=True)
class FreeSnapshot:
    phase: str
    remaining_ms: int
    elapsed_ms: int
    counts: dict[int, int]
    total: int
    last_punch: str
    combo: str
    longest_combo: int
    uncertain: int
    classifier: str
    message: str
    punches_per_minute: float | None
    named_counts: dict[str, int]
    rows: tuple[tuple[str, int, str], ...]


class FreeRound:
    def __init__(
        self,
        classifier_name: str,
        *,
        duration_ms: int = 90_000,
        countdown_ms: int = 3_000,
    ) -> None:
        if duration_ms <= 0 or countdown_ms <= 0:
            raise ValueError("La ronda y la cuenta regresiva deben durar más de cero.")
        self.classifier_name = classifier_name
        self.duration_ms = duration_ms
        self.countdown_ms = countdown_ms
        self.phase = "idle"
        self._ledger = PunchLedger()
        self.counts = self._ledger.counts
        self.named_counts = self._ledger.named_counts
        self.total = 0
        self.last_punch = "—"
        self.uncertain = 0
        self.message = "ESPACIO PARA INICIAR"
        self._combo = ComboTracker()
        self._mark_ms: int | None = None
        self._elapsed = 0
        self._uncertain_open = True

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return True
        if key == ord(" ") and self.phase == "idle":
            self.phase = "countdown"
            self._mark_ms = None
            self._elapsed = 0
            self.message = "3"
        elif key in (ord("r"), ord("R")) and self.phase == "summary":
            self.__init__(self.classifier_name, duration_ms=self.duration_ms, countdown_ms=self.countdown_ms)
        return False

    def tick(self, now_ms: int, result: ClassificationResult) -> None:
        if self.phase in {"idle", "summary"}:
            return
        if self._mark_ms is None:
            self._mark_ms = now_ms
        delta = max(0, now_ms - self._mark_ms)
        self._mark_ms = now_ms
        self._elapsed += delta
        if self.phase == "countdown":
            left = self.countdown_ms - self._elapsed
            self.message = str(max(1, (left + 999) // 1000))
            if self._elapsed >= self.countdown_ms:
                self.phase = "active"
                self._elapsed = 0
                self._mark_ms = now_ms
                self.message = "BOXEA"
            return
        if self._elapsed >= self.duration_ms:
            self.phase = "summary"
            self.message = "Ronda terminada"
            return
        self._observe(now_ms, result)

    def _observe(self, now_ms: int, result: ClassificationResult) -> None:
        if result.status == "incierto" or result.display_name == "MOVIMIENTO NO CONFIRMADO":
            if self._uncertain_open:
                self.uncertain += 1
                self._uncertain_open = False
            self.message = "MOVIMIENTO NO CONFIRMADO"
            self._combo.clear()
            self._ledger.release()
            return
        self._uncertain_open = True
        if not result.events:
            self._ledger.release()
            if result.label_id == 0:
                self.message = MOVEMENTS[0].display_name
            return
        for event in result.events:
            if not self._ledger.commit(event):
                continue
            self.counts = self._ledger.counts
            self.named_counts = self._ledger.named_counts
            self.total = self._ledger.total
            self.last_punch = self._ledger.last_punch
            self.message = event.display_name
            self._combo.add(now_ms, event.label_id, event.display_name)

    def snapshot(self) -> FreeSnapshot:
        remaining = 0
        elapsed = 0
        if self.phase == "active":
            elapsed = min(self._elapsed, self.duration_ms)
            remaining = self.duration_ms - elapsed
        elif self.phase == "summary":
            elapsed = self.duration_ms
        pace = None
        if elapsed >= 1000:
            pace = self.total * 60_000 / elapsed
        return FreeSnapshot(
            phase=self.phase,
            remaining_ms=remaining,
            elapsed_ms=elapsed,
            counts=dict(self.counts),
            total=self.total,
            last_punch=self.last_punch,
            combo=self._combo.text(),
            longest_combo=self._combo.longest,
            uncertain=self.uncertain,
            classifier=self.classifier_name,
            message=self.message,
            punches_per_minute=pace,
            named_counts=dict(self.named_counts),
            rows=rows_for(self.classifier_name, self.counts, self.named_counts),
        )

    def record(self) -> dict[str, object]:
        view = self.snapshot()
        return {
            "game": "free",
            "classifier": self.classifier_name,
            "score_is_not_neat_accuracy": True,
            "contains_training_samples": False,
            "elapsed_ms": view.elapsed_ms,
            "total": view.total,
            "counts": {str(label): count for label, count in view.counts.items()},
            "named_counts": dict(view.named_counts),
            "punch_rows": [
                {"name": name, "count": count, "side": side}
                for name, count, side in view.rows
            ],
            "longest_combo": view.longest_combo,
            "punches_per_minute": view.punches_per_minute,
            "uncertain": view.uncertain,
        }
