"""Decide si un objetivo se completó. Cruzar el círculo no basta."""

from __future__ import annotations

from dataclasses import dataclass

from app.targets.placement import within_radius


@dataclass(frozen=True, slots=True)
class ImpactJudgement:
    status: str
    resolved: bool
    failure: bool
    pause: bool


def judge_target(
    *,
    requested: int,
    event_present: bool,
    event_label: int | None,
    event_key: str | None,
    consumed_key: str | None,
    wrist: tuple[float, float] | None,
    target: tuple[float, float],
    radius: float,
    tracking: bool,
    elapsed_ms: int,
    limit_ms: int,
) -> ImpactJudgement:
    """Combina el evento heurístico, la trayectoria y el tiempo."""

    if not tracking:
        return ImpactJudgement("sin_seguimiento", False, False, True)
    if event_present and event_key is not None and event_key == consumed_key:
        return ImpactJudgement("duplicado", False, False, False)
    if event_present:
        if requested == 0:
            return ImpactJudgement("guardia_rota", True, True, False)
        if event_label not in (1, 2, 3, 4):
            return ImpactJudgement("no_identificado", True, True, False)
        if event_label != requested:
            return ImpactJudgement("equivocado", True, True, False)
        if not within_radius(wrist, target, radius):
            return ImpactJudgement("lejos", True, True, False)
        return ImpactJudgement("acierto", True, False, False)
    if elapsed_ms >= limit_ms:
        if requested == 0:
            return ImpactJudgement("guardia", True, False, False)
        return ImpactJudgement("fallo", True, True, False)
    return ImpactJudgement("pendiente", False, False, False)
