"""Objetivos anclados al torso y normalizados por la anchura de hombros."""

from __future__ import annotations

from dataclasses import dataclass
import math


MAX_REACH = 1.15
# x positivo: hacia el hombro derecho de la persona. y positivo: hacia las caderas.
OFFSETS: dict[int, tuple[float, float]] = {
    0: (0.0, -0.22),
    1: (-0.36, -0.68),
    2: (0.42, -0.64),
    3: (-0.96, -0.06),
    4: (0.96, -0.06),
}
RADII: dict[int, float] = {
    0: 0.0,
    1: 0.48,
    2: 0.48,
    3: 0.52,
    4: 0.52,
}


def clamp_offset(x: float, y: float) -> tuple[float, float]:
    """Mantiene el objetivo dentro del alcance de un brazo."""

    y = min(max(y, -0.90), 0.45)
    reach = math.hypot(x, y)
    if reach > MAX_REACH:
        scale = MAX_REACH / reach
        x *= scale
        y *= scale
    return x, y


def offset_for(label_id: int) -> tuple[float, float]:
    if label_id not in OFFSETS:
        raise ValueError("No hay un objetivo para esa clase.")
    return clamp_offset(*OFFSETS[label_id])


def radius_for(label_id: int) -> float:
    return RADII[label_id]


@dataclass
class PointSmoother:
    """Filtra saltos de un fotograma sin despegar el objetivo del cuerpo."""

    alpha: float = 0.2
    value: tuple[float, float] | None = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        if not 0 < self.alpha <= 1:
            raise ValueError("El suavizado debe estar entre 0 y 1.")
        if self.value is None:
            self.value = point
        else:
            self.value = (
                self.alpha * point[0] + (1.0 - self.alpha) * self.value[0],
                self.alpha * point[1] + (1.0 - self.alpha) * self.value[1],
            )
        return self.value

    def reset(self) -> None:
        self.value = None


def display_point(
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    mirror: bool,
) -> tuple[float, float]:
    """Pasa una coordenada normalizada de la cámara al fotograma que se dibuja."""

    shown_x = 1.0 - x if mirror else x
    return shown_x * (width - 1), y * (height - 1)


def _axes(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float, float, float, float, float, float] | None:
    vx = right[0] - left[0]
    vy = right[1] - left[1]
    scale = math.hypot(vx, vy)
    if scale < 1e-3:
        return None
    along_x = vx / scale
    along_y = vy / scale
    down_x = -along_y
    down_y = along_x
    center_x = (left[0] + right[0]) / 2.0
    center_y = (left[1] + right[1]) / 2.0
    return along_x, along_y, down_x, down_y, scale, center_x, center_y


def body_offset(
    point: tuple[float, float],
    left_shoulder: tuple[float, float],
    right_shoulder: tuple[float, float],
) -> tuple[float, float] | None:
    """Desplazamiento en anchos de hombro: lateral y hacia las caderas."""

    axes = _axes(left_shoulder, right_shoulder)
    if axes is None:
        return None
    along_x, along_y, down_x, down_y, scale, center_x, center_y = axes
    vx = point[0] - center_x
    vy = point[1] - center_y
    lateral = (vx * along_x + vy * along_y) / scale
    downward = (vx * down_x + vy * down_y) / scale
    if not math.isfinite(lateral) or not math.isfinite(downward):
        return None
    return lateral, downward


def project_target(
    left_shoulder: tuple[float, float],
    right_shoulder: tuple[float, float],
    offset: tuple[float, float],
    radius: float,
    width: int,
    height: int,
    *,
    margin_ratio: float = 0.06,
) -> tuple[int, int, int] | None:
    """Proyecta el objetivo y lo recorta para que no salga de la imagen."""

    axes = _axes(left_shoulder, right_shoulder)
    if axes is None:
        return None
    along_x, along_y, down_x, down_y, scale, center_x, center_y = axes
    lateral, downward = clamp_offset(*offset)
    px = center_x + lateral * scale * along_x + downward * scale * down_x
    py = center_y + lateral * scale * along_y + downward * scale * down_y
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    px = min(max(px, margin_x), width - 1 - margin_x)
    py = min(max(py, margin_y), height - 1 - margin_y)
    pixel_radius = max(12, int(round(radius * scale))) if radius > 0 else 0
    return int(round(px)), int(round(py)), pixel_radius


def within_radius(
    wrist: tuple[float, float] | None,
    target: tuple[float, float],
    radius: float,
) -> bool:
    if wrist is None or radius <= 0:
        return False
    return math.hypot(wrist[0] - target[0], wrist[1] - target[1]) <= radius
