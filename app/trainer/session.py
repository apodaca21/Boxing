"""Sesión de entrenamiento libre: tiempo activo, conteos y ritmo."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.dataset_manager import MOVEMENTS
from app.trainer.counting import PunchLedger, event_identity, rows_for
from app.trainer.results import ClassificationResult


HIGHLIGHT_MS = 700


@dataclass(frozen=True, slots=True)
class LiveView:
    """Estado visual del entrenador. No incluye porcentajes de confianza."""

    running: bool
    recognition: str
    last_punch: str
    counts: dict[int, int]
    total: int
    punches_per_minute: float | None
    elapsed_ms: int
    source: str
    provisional: bool
    highlight_arm: str | None
    instructions: str
    rows: tuple[tuple[str, int, str], ...]
    candidate: str = "—"
    sensitivity: str = "NORMAL"
    latency_ms: float | None = None
    moving: bool = False
    debug: bool = False
    debug_lines: tuple[str, ...] = ()
    training_style: str = "libre"
    rejection: str | None = None


class TrainingSession:
    """El contador avanza solo con eventos completos y la sesión en marcha."""

    def __init__(self) -> None:
        self.running = True
        self._ledger = PunchLedger()
        self.counts = self._ledger.counts
        self.named_counts = self._ledger.named_counts
        self.punch_total = 0
        self.last_punch = "—"
        self.recognition = MOVEMENTS[0].display_name
        self.source = "heuristic"
        self.provisional = True
        self.elapsed_ms = 0
        self._previous_timestamp: int | None = None
        self._highlight_arm: str | None = None
        self._highlight_until_ms = -1
        self.candidate = "—"
        self.sensitivity = "NORMAL"
        self.latency_ms: float | None = None
        self.moving = False
        self.debug = False
        self.debug_lines: tuple[str, ...] = ()
        self.training_style = "libre"
        self.rejection: str | None = None

    def reset(self) -> None:
        self.running = True
        self._ledger.reset()
        self.counts = self._ledger.counts
        self.named_counts = self._ledger.named_counts
        self.punch_total = 0
        self.last_punch = "—"
        self.recognition = MOVEMENTS[0].display_name
        self.elapsed_ms = 0
        self._previous_timestamp = None
        self._highlight_arm = None
        self._highlight_until_ms = -1
        self.candidate = "—"
        self.moving = False
        self.rejection = None
        self.debug_lines = ()

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return True
        if key in (ord("p"), ord("P"), ord(" ")):
            self.running = not self.running
            self._previous_timestamp = None
        elif key in (ord("r"), ord("R")):
            self.reset()
        return False

    def advance(self, timestamp_ms: int, result: ClassificationResult) -> None:
        if self.running and self._previous_timestamp is not None:
            delta = timestamp_ms - self._previous_timestamp
            if 0 < delta <= 5000:
                self.elapsed_ms += delta
        self._previous_timestamp = timestamp_ms
        self.source = result.source
        self.provisional = result.source == "heuristic"
        self.candidate = result.candidate or "—"
        self.moving = result.moving
        self.rejection = result.rejection
        self.latency_ms = result.latency_ms
        if result.sensitivity:
            self.sensitivity = result.sensitivity.upper()
        self.debug_lines = result.debug_lines
        if result.events or not result.moving:
            self.recognition = result.display_name
        if not self.running:
            return
        if not result.events:
            self._ledger.release()
        for event in result.events:
            if event.event_id is None:
                key, arm = event_identity(event)
                event = replace(event, event_id=f"session:{timestamp_ms}:{arm}:{key}")
            if not self._ledger.commit(event):
                continue
            self.counts = self._ledger.counts
            self.named_counts = self._ledger.named_counts
            self.punch_total = self._ledger.total
            self.last_punch = self._ledger.last_punch
            self._highlight_arm = event.arm
            self._highlight_until_ms = timestamp_ms + HIGHLIGHT_MS

    def view(self, timestamp_ms: int) -> LiveView:
        total = self.punch_total
        pace = None
        if self.elapsed_ms >= 1000 and total >= 0:
            pace = total * 60_000 / self.elapsed_ms
        highlight = (
            self._highlight_arm
            if timestamp_ms <= self._highlight_until_ms
            else None
        )
        state = "EN MARCHA" if self.running else "EN PAUSA"
        return LiveView(
            running=self.running,
            recognition=self.recognition,
            last_punch=self.last_punch,
            counts=dict(self.counts),
            total=total,
            punches_per_minute=pace,
            elapsed_ms=self.elapsed_ms,
            source=self.source,
            provisional=self.provisional,
            highlight_arm=highlight,
            instructions=f"{state}   P pausa   R reinicia   Q sale",
            rows=self._rows(),
            candidate=self.candidate,
            sensitivity=self.sensitivity,
            latency_ms=self.latency_ms,
            moving=self.moving,
            debug=self.debug,
            debug_lines=self.debug_lines,
            training_style=self.training_style,
            rejection=self.rejection,
        )

    def _rows(self) -> tuple[tuple[str, int, str], ...]:
        return rows_for(self.source, self.counts, self.named_counts)
