"""Resultado común de cualquier clasificador de movimiento."""

from __future__ import annotations

from dataclasses import dataclass

from app.dataset_manager import MOVEMENTS


STATUS_NEUTRAL = "neutral"
STATUS_STARTING = "iniciando"
STATUS_EXTENDING = "extendiendo"
STATUS_RETURNING = "regresando"
STATUS_COMPLETED = "completado"
STATUS_UNKNOWN = "unknown"
STATUS_NO_DETECTION = "no_detection"

PHASE_RANK = {
    STATUS_NEUTRAL: 0,
    STATUS_STARTING: 1,
    STATUS_EXTENDING: 2,
    STATUS_RETURNING: 3,
    STATUS_COMPLETED: 4,
}


@dataclass(frozen=True, slots=True)
class PunchEvent:
    """Golpe registrado una sola vez, al cerrar un ciclo temporal."""

    label_id: int | None
    display_name: str
    arm: str | None
    counter_key: str | None = None


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Salida intercambiable entre detector heurístico, NEAT y un modelo externo."""

    source: str
    status: str
    label_id: int | None
    display_name: str
    events: tuple[PunchEvent, ...] = ()
    active_arm: str | None = None
    candidate: str | None = None
    rejection: str | None = None
    latency_ms: float | None = None
    sensitivity: str | None = None
    moving: bool = False
    debug_lines: tuple[str, ...] = ()

    @property
    def counted(self) -> bool:
        return bool(self.events)


def neutral_result(source: str) -> ClassificationResult:
    movement = MOVEMENTS[0]
    return ClassificationResult(
        source=source,
        status=STATUS_NEUTRAL,
        label_id=movement.label_id,
        display_name=movement.display_name,
    )


def no_detection_result(source: str) -> ClassificationResult:
    return ClassificationResult(
        source=source,
        status=STATUS_NO_DETECTION,
        label_id=None,
        display_name="SIN DETECCION",
    )


def unknown_result(source: str, arm: str | None = None) -> ClassificationResult:
    return ClassificationResult(
        source=source,
        status=STATUS_UNKNOWN,
        label_id=None,
        display_name="MOVIMIENTO NO IDENTIFICADO",
        active_arm=arm,
    )
