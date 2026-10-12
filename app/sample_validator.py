"""Reglas de calidad para aceptar una ejecución completa."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from app.config import AppConfig
from app.dataset_manager import (
    SCHEMA_VERSION,
    SPLIT_UNIT,
    TRACKED_LANDMARK_NAMES,
    LandmarkRecord,
    MovementSample,
    movement_by_id,
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...]


def validate_sample(sample: MovementSample, config: AppConfig) -> ValidationResult:
    """Devuelve todos los motivos por los que una muestra no puede guardarse."""

    reasons: list[str] = []
    if sample.schema_version != SCHEMA_VERSION:
        reasons.append("La versión del esquema no es compatible.")
    if sample.split_unit != SPLIT_UNIT:
        reasons.append(
            "La muestra debe conservarse completa como unidad de división."
        )
    try:
        movement = movement_by_id(sample.label_id)
    except ValueError:
        movement = None
        reasons.append("La etiqueta del movimiento no es válida.")
    if movement is not None and sample.label_name != movement.name:
        reasons.append("El nombre de la etiqueta no coincide con su identificador.")
    if sample.stance != config.stance:
        reasons.append("La postura registrada no es orthodox.")
    if sample.contains_images or sample.contains_audio or sample.contains_video:
        reasons.append("La muestra no puede contener imágenes, audio ni video.")
    if (
        sample.camera.actual_width is None
        or sample.camera.actual_height is None
        or sample.camera.actual_width <= 0
        or sample.camera.actual_height <= 0
    ):
        reasons.append("Falta la resolución real de la cámara.")
    if _contains_invalid_number(asdict(sample)):
        reasons.append("La muestra contiene valores numéricos no válidos.")

    frames = sample.frames
    if sample.frame_count != len(frames):
        reasons.append("El número de fotogramas no coincide con la secuencia.")
    if not frames:
        reasons.append("La captura no contiene fotogramas.")
        return ValidationResult(False, tuple(reasons))
    if len(frames) < config.min_sample_frames:
        reasons.append(
            f"La captura solo contiene {len(frames)} fotogramas; "
            f"se requieren al menos {config.min_sample_frames}."
        )
    if any(set(frame.landmarks) != set(TRACKED_LANDMARK_NAMES) for frame in frames):
        reasons.append("La estructura de articulaciones está incompleta.")

    timestamps = [frame.timestamp_ms for frame in frames]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        reasons.append("Los timestamps no son estrictamente crecientes.")
    measured_duration = timestamps[-1] - timestamps[0]
    if sample.duration_ms != measured_duration:
        reasons.append("La duración no coincide con los timestamps.")
    if not (
        config.min_sample_duration_ms
        <= measured_duration
        <= config.max_sample_duration_ms
    ):
        reasons.append(
            "La duración real está fuera del rango permitido de 1.5 a 2.8 segundos."
        )
    if any(
        frame.elapsed_ms != frame.timestamp_ms - timestamps[0] for frame in frames
    ):
        reasons.append("Los tiempos relativos no coinciden con los timestamps.")
    if measured_duration > 0:
        expected_fps = (len(frames) - 1) * 1000 / measured_duration
        if (
            sample.approximate_fps is None
            or not math.isclose(
                sample.approximate_fps,
                expected_fps,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
        ):
            reasons.append("El FPS aproximado no coincide con los timestamps.")

    detected_ratio = sum(frame.pose_detected for frame in frames) / len(frames)
    if detected_ratio < config.min_detected_ratio:
        reasons.append("No se detectó una persona en suficientes fotogramas.")
    reliable_ratio = (
        sum(_frame_is_reliable(frame.landmarks, config) for frame in frames)
        / len(frames)
    )
    if reliable_ratio < config.min_reliable_ratio:
        reasons.append("Los brazos o la cadera no fueron suficientemente visibles.")

    return ValidationResult(not reasons, tuple(reasons))


def _frame_is_reliable(
    landmarks: dict[str, LandmarkRecord | None],
    config: AppConfig,
) -> bool:
    return all(
        _record_is_reliable(landmarks.get(name), config)
        for name in TRACKED_LANDMARK_NAMES
    )


def _record_is_reliable(
    record: LandmarkRecord | None,
    config: AppConfig,
) -> bool:
    if record is None:
        return False
    values = (record.x, record.y, record.z, record.visibility, record.presence)
    if any(value is None or not math.isfinite(value) for value in values):
        return False
    return (
        record.visibility >= config.min_landmark_visibility
        and record.presence >= config.min_landmark_presence
    )


def _contains_invalid_number(value: object) -> bool:
    if value is None or isinstance(value, str):
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, dict):
        return any(_contains_invalid_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_invalid_number(item) for item in value)
    return False
