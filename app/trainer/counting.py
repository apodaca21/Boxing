"""Reglas comunes para identificar y mostrar golpes confirmados.

Los clasificadores no comparten las mismas etiquetas: el detector local usa
IDs 1-4 y el modelo externo usa nombres como ``jab`` o ``uppercut``. Un
candidato en movimiento no cuenta. Solo un evento confirmado, con identificador
único, incrementa una clase y el total.
"""

from __future__ import annotations

from app.dataset_manager import MOVEMENTS
from app.trainer.results import PunchEvent


EventIdentity = tuple[str, str | None]
CounterRow = tuple[str, int, str]
PUNCH_LABELS = (1, 2, 3, 4)


def event_identity(event: PunchEvent) -> EventIdentity:
    """Devuelve una identidad estable para bloquear duplicados de un evento."""

    key = event.counter_key
    if not key and event.label_id in MOVEMENTS and event.label_id != 0:
        key = MOVEMENTS[event.label_id].name
    if not key:
        key = event.display_name.strip().lower()
    return key, event.arm


def event_key(event: PunchEvent) -> str:
    """Identificador único del golpe físico, o una clave de respaldo."""

    if event.event_id:
        return event.event_id
    started = "" if event.started_ms is None else str(event.started_ms)
    key, arm = event_identity(event)
    return f"{event.source or 'event'}:{started}:{arm}:{key}"


def add_event(
    event: PunchEvent,
    counts: dict[int, int],
    named_counts: dict[str, int],
) -> bool:
    """Suma un evento al contador apropiado.

    Las cuatro clases del dataset conservan sus IDs. Las clases que solo
    existen en el modelo externo se guardan por nombre y no ocupan una clase
    del dataset por accidente.
    """

    if not event.confirmed:
        return False
    if event.label_id == 0:
        return False
    if event.label_id in counts:
        counts[event.label_id] += 1  # type: ignore[index]
        return True
    if event.label_id in PUNCH_LABELS:
        counts[event.label_id] = counts.get(event.label_id, 0) + 1
        return True
    key, _arm = event_identity(event)
    if not key:
        return False
    named_counts[key] = named_counts.get(key, 0) + 1
    return True


class PunchLedger:
    """Fuente única de conteo. Un event_id solo incrementa una clase."""

    def __init__(self) -> None:
        self.counts: dict[int, int] = {label: 0 for label in PUNCH_LABELS}
        self.named_counts: dict[str, int] = {}
        self.total = 0
        self.last_punch = "—"
        self._seen: set[str] = set()
        self._open: str | None = None

    def reset(self) -> None:
        self.__init__()

    def release(self) -> None:
        self._open = None

    def commit(self, event: PunchEvent) -> bool:
        return commit_punch_event(event, self)

    @property
    def category_sum(self) -> int:
        return sum(self.counts.values()) + sum(self.named_counts.values())


def commit_punch_event(event: PunchEvent, ledger: PunchLedger) -> bool:
    """Incrementa una sola clase y el total si el evento no se había visto."""

    if not event.confirmed:
        return False
    if event.label_id == 0:
        return False
    if event.label_id is None and not event.counter_key:
        return False
    if event.event_id:
        if event.event_id in ledger._seen:
            return False
    else:
        key = event_key(event)
        if key == ledger._open:
            return False
        ledger._open = key
    if not add_event(event, ledger.counts, ledger.named_counts):
        return False
    if event.event_id:
        ledger._seen.add(event.event_id)
    ledger.total += 1
    ledger.last_punch = event.display_name
    return True


def rows_for(
    source: str,
    counts: dict[int, int],
    named_counts: dict[str, int],
) -> tuple[CounterRow, ...]:
    """Construye las mismas filas para el entrenador y la ronda libre."""

    if source == "external":
        return (
            ("Jab", named_counts.get("jab", 0), "neutral"),
            ("Cross", named_counts.get("cross", 0), "neutral"),
            ("Gancho izquierdo", counts.get(3, 0), "left"),
            ("Gancho derecho", counts.get(4, 0), "right"),
            ("HOOK", named_counts.get("hook", 0), "neutral"),
            ("Uppercut", named_counts.get("uppercut", 0), "neutral"),
        )
    return tuple(
        (
            MOVEMENTS[label_id].display_name,
            counts.get(label_id, 0),
            "left" if label_id in (1, 3) else "right",
        )
        for label_id in PUNCH_LABELS
    )
