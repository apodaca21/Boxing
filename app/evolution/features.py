"""Transforma cada ejecución completa en un vector temporal de tamaño fijo."""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.dataset_manager import MovementSample


STATIC_CHANNELS = (
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_wrist_distance",
    "right_shoulder_wrist_distance",
    "left_wrist_relative_x",
    "left_wrist_relative_y",
    "right_wrist_relative_x",
    "right_wrist_relative_y",
)
DYNAMIC_CHANNELS = (
    "left_wrist_speed",
    "right_wrist_speed",
    "left_extension_rate",
    "right_extension_rate",
)
MISSING_SENTINEL = -4.0
CLIP_LIMIT = 3.0
FEATURE_VERSION = 1


def vector_size(time_steps: int) -> int:
    if time_steps < 2:
        raise ValueError("El remuestreo temporal requiere al menos 2 puntos.")
    return time_steps * len(STATIC_CHANNELS) + (time_steps - 1) * len(DYNAMIC_CHANNELS)


def feature_names(time_steps: int) -> tuple[str, ...]:
    names = [
        f"t{step}_{channel}"
        for step in range(time_steps)
        for channel in STATIC_CHANNELS
    ]
    names.extend(
        f"i{interval}_{channel}"
        for interval in range(time_steps - 1)
        for channel in DYNAMIC_CHANNELS
    )
    return tuple(names)


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _static_series(sample: MovementSample) -> dict[str, list[tuple[float, float | None]]]:
    series = {channel: [] for channel in STATIC_CHANNELS}
    for frame in sample.frames:
        timestamp = float(frame.elapsed_ms)
        features = frame.features
        left = features.left_wrist_relative
        right = features.right_wrist_relative
        values = {
            "left_elbow_angle": features.left_elbow_angle,
            "right_elbow_angle": features.right_elbow_angle,
            "left_shoulder_wrist_distance": features.left_shoulder_wrist_distance,
            "right_shoulder_wrist_distance": features.right_shoulder_wrist_distance,
            "left_wrist_relative_x": None if left is None else left.x,
            "left_wrist_relative_y": None if left is None else left.y,
            "right_wrist_relative_x": None if right is None else right.x,
            "right_wrist_relative_y": None if right is None else right.y,
        }
        for channel, value in values.items():
            series[channel].append((timestamp, _finite(value)))
    return series


def _resample(
    points: list[tuple[float, float | None]],
    grid: list[float],
) -> list[float | None]:
    known: dict[float, float] = {}
    for timestamp, value in points:
        if value is not None:
            known[timestamp] = value
    ordered = sorted(known.items())
    resampled: list[float | None] = []
    for target in grid:
        if len(ordered) < 1 or target < ordered[0][0] or target > ordered[-1][0]:
            resampled.append(None)
            continue
        if target == ordered[0][0]:
            resampled.append(ordered[0][1])
            continue
        previous = ordered[0]
        chosen: float | None = None
        for current in ordered[1:]:
            if current[0] == target:
                chosen = current[1]
                break
            if previous[0] < target < current[0]:
                span = current[0] - previous[0]
                if span <= 0:
                    chosen = None
                else:
                    weight = (target - previous[0]) / span
                    chosen = previous[1] + weight * (current[1] - previous[1])
                break
            previous = current
        resampled.append(chosen if chosen is None or math.isfinite(chosen) else None)
    return resampled


def _rate(current: float | None, previous: float | None, dt_seconds: float) -> float | None:
    if current is None or previous is None or dt_seconds <= 0:
        return None
    value = (current - previous) / dt_seconds
    return value if math.isfinite(value) else None


def extract_raw_features(sample: MovementSample, time_steps: int = 8) -> tuple[float | None, ...]:
    """Remuestrea la secuencia completa sin sustituir ausencias por cero."""

    if sample.duration_ms < 0:
        raise ValueError("La duración de la muestra no puede ser negativa.")
    grid = [sample.duration_ms * step / (time_steps - 1) for step in range(time_steps)]
    resampled = {
        channel: _resample(points, grid)
        for channel, points in _static_series(sample).items()
    }
    values: list[float | None] = []
    for step in range(time_steps):
        for channel in STATIC_CHANNELS:
            values.append(resampled[channel][step])

    for interval in range(time_steps - 1):
        dt_seconds = (grid[interval + 1] - grid[interval]) / 1000.0
        left_speed = None
        right_speed = None
        left_x = resampled["left_wrist_relative_x"]
        left_y = resampled["left_wrist_relative_y"]
        right_x = resampled["right_wrist_relative_x"]
        right_y = resampled["right_wrist_relative_y"]
        if all(
            value is not None
            for value in (
                left_x[interval],
                left_x[interval + 1],
                left_y[interval],
                left_y[interval + 1],
            )
        ) and dt_seconds > 0:
            left_speed = math.hypot(
                left_x[interval + 1] - left_x[interval],  # type: ignore[operator]
                left_y[interval + 1] - left_y[interval],  # type: ignore[operator]
            ) / dt_seconds
        if all(
            value is not None
            for value in (
                right_x[interval],
                right_x[interval + 1],
                right_y[interval],
                right_y[interval + 1],
            )
        ) and dt_seconds > 0:
            right_speed = math.hypot(
                right_x[interval + 1] - right_x[interval],  # type: ignore[operator]
                right_y[interval + 1] - right_y[interval],  # type: ignore[operator]
            ) / dt_seconds
        values.extend(
            (
                _finite(left_speed),
                _finite(right_speed),
                _rate(
                    resampled["left_shoulder_wrist_distance"][interval + 1],
                    resampled["left_shoulder_wrist_distance"][interval],
                    dt_seconds,
                ),
                _rate(
                    resampled["right_shoulder_wrist_distance"][interval + 1],
                    resampled["right_shoulder_wrist_distance"][interval],
                    dt_seconds,
                ),
            )
        )
    expected = vector_size(time_steps)
    if len(values) != expected:
        raise RuntimeError(f"El vector temporal mide {len(values)}, no {expected}.")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    """Normalización ajustada exclusivamente con observaciones de entrenamiento."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    time_steps: int
    names: tuple[str, ...]
    missing_sentinel: float = MISSING_SENTINEL
    clip_limit: float = CLIP_LIMIT
    feature_version: int = FEATURE_VERSION

    @classmethod
    def fit(
        cls,
        rows: list[tuple[float | None, ...]],
        time_steps: int,
    ) -> FeatureScaler:
        if not rows:
            raise ValueError("No hay muestras de entrenamiento para normalizar.")
        width = vector_size(time_steps)
        if any(len(row) != width for row in rows):
            raise ValueError("Las muestras no tienen la dimensionalidad esperada.")
        means: list[float] = []
        deviations: list[float] = []
        for index in range(width):
            observed = [
                float(row[index])
                for row in rows
                if row[index] is not None and math.isfinite(row[index])
            ]
            if not observed:
                means.append(0.0)
                deviations.append(1.0)
                continue
            center = sum(observed) / len(observed)
            variance = sum((value - center) ** 2 for value in observed) / len(observed)
            scale = math.sqrt(variance)
            means.append(center)
            deviations.append(1.0 if scale < 1e-8 else scale)
        return cls(
            mean=tuple(means),
            std=tuple(deviations),
            time_steps=time_steps,
            names=feature_names(time_steps),
        )

    def transform(self, row: tuple[float | None, ...]) -> tuple[float, ...]:
        if len(row) != len(self.mean):
            raise ValueError("La muestra no coincide con el vector de características.")
        transformed: list[float] = []
        for index, value in enumerate(row):
            if value is None or not math.isfinite(value):
                transformed.append(self.missing_sentinel)
                continue
            scaled = (value - self.mean[index]) / self.std[index]
            transformed.append(min(self.clip_limit, max(-self.clip_limit, scaled)))
        return tuple(transformed)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_version": self.feature_version,
            "time_steps": self.time_steps,
            "vector_size": len(self.names),
            "names": list(self.names),
            "static_channels": list(STATIC_CHANNELS),
            "dynamic_channels": list(DYNAMIC_CHANNELS),
            "mean": list(self.mean),
            "std": list(self.std),
            "missing_sentinel": self.missing_sentinel,
            "clip_limit": self.clip_limit,
            "missing_policy": (
                "Los valores ausentes permanecen ausentes durante el remuestreo "
                "y se codifican después con un centinela fuera del rango normalizado."
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FeatureScaler:
        names = tuple(str(name) for name in payload["names"])  # type: ignore[index]
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),  # type: ignore[index]
            std=tuple(float(value) for value in payload["std"]),  # type: ignore[index]
            time_steps=int(payload["time_steps"]),
            names=names,
            missing_sentinel=float(payload["missing_sentinel"]),
            clip_limit=float(payload["clip_limit"]),
            feature_version=int(payload["feature_version"]),
        )
