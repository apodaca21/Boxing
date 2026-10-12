"""Secuencias de golpes confirmados. Un movimiento incierto corta la cadena."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComboStep:
    timestamp_ms: int
    label_id: int
    display_name: str


class ComboTracker:
    def __init__(self, window_ms: int = 1800) -> None:
        if window_ms <= 0:
            raise ValueError("La ventana de combinación debe ser positiva.")
        self.window_ms = window_ms
        self._steps: list[ComboStep] = []
        self.longest = 0

    def clear(self) -> None:
        self._steps.clear()

    def add(self, timestamp_ms: int, label_id: int, display_name: str) -> str:
        self._steps.append(ComboStep(timestamp_ms, label_id, display_name))
        self._steps = [
            step for step in self._steps if timestamp_ms - step.timestamp_ms <= self.window_ms
        ]
        self.longest = max(self.longest, len(self._steps))
        return self.text()

    def text(self) -> str:
        if not self._steps:
            return "—"
        return " > ".join(step.display_name for step in self._steps)
