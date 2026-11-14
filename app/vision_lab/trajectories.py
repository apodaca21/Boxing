"""Contratos ligeros para trayectorias de muñecas de la Fase 2."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WristPoint:
    timestamp_ns: int
    x: float
    y: float
    visibility: float
    hand: str
    camera_id: str = "front"
    tracking_valid: bool = True


@dataclass(slots=True)
class TrajectoryBuffer:
    """Buffer acotado; conserva puntos numéricos, no fotogramas."""

    max_points: int = 2400
    points: list[WristPoint] = field(default_factory=list)

    def append(self, point: WristPoint) -> None:
        if self.max_points < 1:
            raise ValueError("max_points debe ser positivo.")
        self.points.append(point)
        if len(self.points) > self.max_points:
            del self.points[: len(self.points) - self.max_points]

    def clear(self) -> None:
        self.points.clear()
