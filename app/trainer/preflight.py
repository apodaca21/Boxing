"""Mediciones de visibilidad para comprobar una sesión real antes de calibrar.

Este módulo no intenta inferir automáticamente el ángulo de la cámara. La
orientación se registra como una elección explícita de la persona (frontal o
30--45 grados); así no se confunde orientación de cámara con lateralidad
anatómica.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig
from app.pose_tracker import LandmarkId, PoseObservation


FRONT_REQUIRED_STABLE_FRAMES = 8
FRONT_MIN_ARM_QUALITY = 1.0
FRONT_MIN_TORSO_QUALITY = 1.0
FRONT_SHOULDER_MIN_WIDTH = 0.10
FRONT_SHOULDER_MAX_WIDTH = 0.78
FRONT_CENTER_MARGIN = 0.08


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
# Para contar golpes no hace falta ver las piernas ni las caderas. Los
# hombros son la referencia suficiente para normalizar la trayectoria de cada
# muñeca, y el clasificador ya tiene un fallback si las caderas no aparecen.
FRONT_TORSO_LANDMARKS = (
    LandmarkId.LEFT_SHOULDER,
    LandmarkId.RIGHT_SHOULDER,
)


def _reliable(pose: PoseObservation, landmark_id: LandmarkId, config: AppConfig) -> bool:
    landmark = pose.landmark(landmark_id)
    return landmark is not None and landmark.is_reliable(
        config.min_landmark_visibility,
        config.min_landmark_presence,
    )


def _ratio(pose: PoseObservation, ids: tuple[LandmarkId, ...], config: AppConfig) -> float:
    return sum(_reliable(pose, item, config) for item in ids) / len(ids)


def arm_tracking_ok(pose: PoseObservation, side: str, config: AppConfig) -> bool:
    """True si hombro+codo+muñeca del brazo se ven lo bastante para separar I/D.

    Las muñecas toleran menos visibilidad que el resto porque en un golpe rápido
    MediaPipe las baja sin perder del todo el brazo.
    """

    ids = LEFT_ARM_LANDMARKS if side == "left" else RIGHT_ARM_LANDMARKS
    for landmark_id in ids:
        landmark = pose.landmark(landmark_id)
        if landmark is None or not landmark.is_finite:
            return False
        if landmark_id in (LandmarkId.LEFT_WRIST, LandmarkId.RIGHT_WRIST):
            if landmark.visibility < 0.32 or landmark.presence < 0.40:
                return False
        elif not landmark.is_reliable(
            config.min_landmark_visibility,
            config.min_landmark_presence,
        ):
            return False
    return True


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


@dataclass(frozen=True, slots=True)
class FrontReadiness:
    """Estado de la webcam frontal para medir ángulos con seguridad."""

    detected: bool = False
    arm_quality: float = 0.0
    torso_quality: float = 0.0
    framed: bool = False
    stable_frames: int = 0
    ready: bool = False
    status: str = "FRONTAL BUSCANDO PERSONA"


class FrontSetupTracker:
    """Exige torso, ambos brazos y encuadre estable antes de contar golpes."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._stable_frames = 0
        self._previous_center: tuple[float, float] | None = None
        self._previous_scale: float | None = None
        self.readiness = FrontReadiness()

    def reset(self) -> None:
        self._stable_frames = 0
        self._previous_center = None
        self._previous_scale = None
        self.readiness = FrontReadiness()

    def observe(self, pose: PoseObservation) -> FrontReadiness:
        if not pose.detected:
            self._stable_frames = 0
            self._previous_center = None
            self._previous_scale = None
            self.readiness = FrontReadiness(status="FRONTAL BUSCANDO PERSONA")
            return self.readiness

        arm_quality = (
            _ratio(pose, LEFT_ARM_LANDMARKS, self._config)
            + _ratio(pose, RIGHT_ARM_LANDMARKS, self._config)
        ) / 2.0
        torso_quality = _ratio(pose, FRONT_TORSO_LANDMARKS, self._config)
        framed, frame_status = self._frame_ok(pose)
        stable = self._update_stability(pose)

        if arm_quality < FRONT_MIN_ARM_QUALITY:
            status = f"FRONTAL MUESTRA BRAZOS {arm_quality:.0%}"
            ready = False
        elif torso_quality < FRONT_MIN_TORSO_QUALITY:
            status = f"FRONTAL MUESTRA TORSO {torso_quality:.0%}"
            ready = False
        elif not framed:
            status = frame_status
            ready = False
        elif self._stable_frames < FRONT_REQUIRED_STABLE_FRAMES:
            status = "FRONTAL ESTABILIZANDO ENCUADRE"
            ready = False
        else:
            status = "FRONTAL LISTA PARA ANGULOS"
            ready = True

        if not ready and not stable and self._stable_frames > 0:
            # Si el encuadre saltó antes de estar listo, reinicia el contador.
            pass

        self.readiness = FrontReadiness(
            detected=True,
            arm_quality=arm_quality,
            torso_quality=torso_quality,
            framed=framed,
            stable_frames=self._stable_frames,
            ready=ready,
            status=status,
        )
        return self.readiness

    def _frame_ok(self, pose: PoseObservation) -> tuple[bool, str]:
        left = pose.landmark(LandmarkId.LEFT_SHOULDER)
        right = pose.landmark(LandmarkId.RIGHT_SHOULDER)
        if (
            left is None
            or right is None
            or not left.is_reliable(
                self._config.min_landmark_visibility,
                self._config.min_landmark_presence,
            )
            or not right.is_reliable(
                self._config.min_landmark_visibility,
                self._config.min_landmark_presence,
            )
        ):
            return False, "FRONTAL MUESTRA HOMBROS"

        width = abs(right.x - left.x)
        center_x = (left.x + right.x) / 2.0
        if width < FRONT_SHOULDER_MIN_WIDTH:
            return False, "FRONTAL ACERCATE (pecho y brazos)"
        if width > FRONT_SHOULDER_MAX_WIDTH:
            return False, "FRONTAL ALEJATE: munecas salen del cuadro"
        if center_x < FRONT_CENTER_MARGIN or center_x > 1.0 - FRONT_CENTER_MARGIN:
            return False, "FRONTAL CENTRATE EN EL ENCUADRE"
        return True, "FRONTAL ENCUADRE OK"

    def _update_stability(self, pose: PoseObservation) -> bool:
        left = pose.landmark(LandmarkId.LEFT_SHOULDER)
        right = pose.landmark(LandmarkId.RIGHT_SHOULDER)
        if left is None or right is None:
            self._stable_frames = 0
            self._previous_center = None
            self._previous_scale = None
            return False

        center = ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0)
        scale = max(abs(right.x - left.x), 0.05)
        stable = True
        if self._previous_center is not None and self._previous_scale is not None:
            movement = (
                (center[0] - self._previous_center[0]) ** 2
                + (center[1] - self._previous_center[1]) ** 2
            ) ** 0.5
            scale_change = abs(scale - self._previous_scale) / self._previous_scale
            stable = movement / scale <= 0.08 and scale_change <= 0.12
        if stable:
            self._stable_frames += 1
        else:
            self._stable_frames = 1
        self._previous_center = center
        self._previous_scale = scale
        return stable
