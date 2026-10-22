"""Mediciones de visibilidad para comprobar una sesión real antes de calibrar.

Este módulo no intenta inferir automáticamente el ángulo de la cámara. La
orientación se registra como una elección explícita de la persona (frontal o
30--45 grados); así no se confunde orientación de cámara con lateralidad
anatómica.
"""

from __future__ import annotations

from app.config import AppConfig
from app.pose_tracker import LandmarkId, PoseObservation


REQUIRED_LANDMARKS = (
    LandmarkId.LEFT_SHOULDER,
    LandmarkId.RIGHT_SHOULDER,
    LandmarkId.LEFT_ELBOW,
    LandmarkId.RIGHT_ELBOW,
    LandmarkId.LEFT_WRIST,
    LandmarkId.RIGHT_WRIST,
    LandmarkId.LEFT_HIP,
    LandmarkId.RIGHT_HIP,
)
LEFT_ARM_LANDMARKS = (
    LandmarkId.LEFT_SHOULDER,
    LandmarkId.LEFT_ELBOW,
    LandmarkId.LEFT_WRIST,
)
RIGHT_ARM_LANDMARKS = (
    LandmarkId.RIGHT_SHOULDER,
    LandmarkId.RIGHT_ELBOW,
    LandmarkId.RIGHT_WRIST,
)
TORSO_LANDMARKS = (
    LandmarkId.LEFT_SHOULDER,
    LandmarkId.RIGHT_SHOULDER,
    LandmarkId.LEFT_HIP,
    LandmarkId.RIGHT_HIP,
)


def _reliable(pose: PoseObservation, landmark_id: LandmarkId, config: AppConfig) -> bool:
    landmark = pose.landmark(landmark_id)
    return landmark is not None and landmark.is_reliable(
        config.min_landmark_visibility,
        config.min_landmark_presence,
    )


def _ratio(pose: PoseObservation, ids: tuple[LandmarkId, ...], config: AppConfig) -> float:
    return sum(_reliable(pose, item, config) for item in ids) / len(ids)


def tracking_snapshot(
    pose: PoseObservation,
    config: AppConfig,
    *,
    fps: float | None = None,
    orientation: str = "no_especificada",
    left_elbow_angle: float | None = None,
    right_elbow_angle: float | None = None,
) -> dict[str, object]:
    """Devuelve una fila auditable de calidad, sin rellenar landmarks ausentes."""

    flags = {
        f"landmark_{landmark_id.name.lower()}": _reliable(pose, landmark_id, config)
        for landmark_id in REQUIRED_LANDMARKS
    }
    left_quality = _ratio(pose, LEFT_ARM_LANDMARKS, config)
    right_quality = _ratio(pose, RIGHT_ARM_LANDMARKS, config)
    torso_quality = _ratio(pose, TORSO_LANDMARKS, config)
    body_visible = bool(pose.detected and torso_quality == 1.0)
    return {
        "timestamp_ms": pose.timestamp_ms,
        "orientation": orientation,
        "pose_detected": pose.detected,
        "body_visible": body_visible,
        "required_landmarks_quality": sum(flags.values()) / len(flags),
        "left_arm_tracking_quality": left_quality,
        "right_arm_tracking_quality": right_quality,
        "torso_tracking_quality": torso_quality,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "fps": fps,
        **flags,
    }


def orientation_label(value: str | None) -> str:
    """Normaliza nombres que se guardan en reportes y calibraciones."""

    key = (value or "").strip().lower()
    aliases = {
        "front": "frontal",
        "frontal": "frontal",
        "f": "frontal",
        "angle": "angulo_30_45",
        "angled": "angulo_30_45",
        "angled_30_45": "angulo_30_45",
        "angulo": "angulo_30_45",
        "a": "angulo_30_45",
        "no_especificada": "no_especificada",
    }
    return aliases.get(key, "no_especificada")
