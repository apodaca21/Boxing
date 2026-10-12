"""Registro local de eventos del entrenador, sin video ni imagen."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import PROJECT_ROOT


DEFAULT_DIAGNOSTIC_PATH = PROJECT_ROOT / "results" / "sparring_diagnostics.jsonl"


class DiagnosticLog:
    """Anota confirmaciones y rechazos. No guarda coordenadas ni fotogramas."""

    def __init__(self, path: Path = DEFAULT_DIAGNOSTIC_PATH) -> None:
        self.path = path
        self._last_rejection_ms = -10_000

    def write(self, record: dict[str, object]) -> None:
        kind = str(record.get("kind", ""))
        timestamp = int(record.get("timestamp_ms", 0))
        if kind == "rechazo" and timestamp - self._last_rejection_ms < 500:
            return
        if kind == "rechazo":
            self._last_rejection_ms = timestamp
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
