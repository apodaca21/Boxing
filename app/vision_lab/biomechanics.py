"""Cálculos numéricos preparados para la Fase 2, en coordenadas de imagen."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from app.vision_lab.trajectories import WristPoint


@dataclass(frozen=True, slots=True)
class Kinematics:
    points: tuple[WristPoint, ...]
    positions: tuple[tuple[float, float] | None, ...]
    velocities_px_s: tuple[tuple[float, float] | None, ...]
    accelerations_px_s2: tuple[tuple[float, float] | None, ...]
    displacement_px: float
    path_length_px: float
    peak_speed_px_s: float
    peak_acceleration_px_s2: float
    direction_change_deg: float | None


def smooth_positions(points: list[WristPoint] | tuple[WristPoint, ...], alpha: float = 0.35) -> tuple[tuple[float, float] | None, ...]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha debe estar entre 0 y 1.")
    smoothed: list[tuple[float, float] | None] = []
    for point in points:
        if not point.tracking_valid or not np.isfinite((point.x, point.y)).all():
            smoothed.append(None)
            continue
        if not smoothed or smoothed[-1] is None:
            smoothed.append((point.x, point.y))
            continue
        previous = np.asarray(smoothed[-1])
        current = np.asarray((point.x, point.y))
        result = alpha * current + (1.0 - alpha) * previous
        smoothed.append((float(result[0]), float(result[1])))
    return tuple(smoothed)


def calculate_kinematics(points: list[WristPoint] | tuple[WristPoint, ...], *, smoothing_alpha: float | None = 0.35) -> Kinematics:
    """Usa dt real y deja inválidas las derivadas en saltos o puntos faltantes."""

    ordered = tuple(points)
    positions = smooth_positions(ordered, smoothing_alpha) if smoothing_alpha is not None else tuple(
        None if not point.tracking_valid or not np.isfinite((point.x, point.y)).all() else (point.x, point.y)
        for point in ordered
    )
    velocities: list[tuple[float, float] | None] = [None for _ in ordered]
    accelerations: list[tuple[float, float] | None] = [None for _ in ordered]
    for index in range(1, len(ordered)):
        dt = (ordered[index].timestamp_ns - ordered[index - 1].timestamp_ns) / 1_000_000_000.0
        if dt <= 0 or not ordered[index].tracking_valid or not ordered[index - 1].tracking_valid:
            continue
        if positions[index - 1] is None or positions[index] is None:
            continue
        first = np.asarray(positions[index - 1])
        second = np.asarray(positions[index])
        velocity = (second - first) / dt
        velocities[index] = (float(velocity[0]), float(velocity[1]))
        if index >= 2 and velocities[index - 1] is not None:
            previous_velocity = np.asarray(velocities[index - 1])
            acceleration = (velocity - previous_velocity) / dt
            accelerations[index] = (float(acceleration[0]), float(acceleration[1]))
    valid_positions = [np.asarray(position) for position in positions if position is not None]
    displacement = 0.0
    path_length = 0.0
    if len(valid_positions) >= 2:
        displacement = float(np.linalg.norm(valid_positions[-1] - valid_positions[0]))
        path_length = float(sum(np.linalg.norm(second - first) for first, second in zip(valid_positions, valid_positions[1:])))
    speeds = [float(np.linalg.norm(value)) for value in velocities if value is not None]
    accelerations_abs = [float(np.linalg.norm(value)) for value in accelerations if value is not None]
    directions: list[float] = []
    previous_velocity: np.ndarray | None = None
    for velocity in velocities:
        if velocity is None:
            previous_velocity = None
            continue
        current_velocity = np.asarray(velocity, dtype=float)
        if previous_velocity is not None:
            first_norm = float(np.linalg.norm(previous_velocity))
            second_norm = float(np.linalg.norm(current_velocity))
            if first_norm > 1e-9 and second_norm > 1e-9:
                cosine = float(np.clip(np.dot(previous_velocity, current_velocity) / (first_norm * second_norm), -1.0, 1.0))
                directions.append(math.degrees(math.acos(cosine)))
        previous_velocity = current_velocity
    return Kinematics(
        points=ordered,
        positions=positions,
        velocities_px_s=tuple(velocities),
        accelerations_px_s2=tuple(accelerations),
        displacement_px=displacement,
        path_length_px=path_length,
        peak_speed_px_s=max(speeds, default=0.0),
        peak_acceleration_px_s2=max(accelerations_abs, default=0.0),
        direction_change_deg=None if not directions else float(max(directions)),
    )


def calculate_elbow_angle(
    shoulder: tuple[float, float],
    elbow: tuple[float, float],
    wrist: tuple[float, float],
) -> float | None:
    """Ángulo hombro-codo-muñeca en grados; None para segmentos degenerados."""

    first = np.asarray(shoulder, dtype=float) - np.asarray(elbow, dtype=float)
    second = np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float)
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm <= 1e-12 or not np.isfinite(norm):
        return None
    cosine = float(np.clip(np.dot(first, second) / norm, -1.0, 1.0))
    return math.degrees(math.acos(cosine))
