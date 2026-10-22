"""Detector biomecánico provisional con un ciclo temporal por brazo.

Los umbrales son una primera hipótesis de trabajo y necesitan calibración con
personas y cámaras reales. En el perfil normal un golpe solo se registra
cuando el brazo sale de la guardia y regresa. En sparring, con confirmación
anticipada, basta un pico seguido de frenado. Mantener el brazo extendido no
acumula conteos.

Cada brazo tiene su propio estado y su propio cooldown. Las coordenadas de la
pose se expresan respecto al torso para que caminar o girar no parezcan un golpe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

from app.dataset_manager import MOVEMENTS
from app.geometry import BiomechanicalFeatures, RelativePoint
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from app.trainer.results import (
    PHASE_RANK,
    STATUS_COMPLETED,
    STATUS_EXTENDING,
    STATUS_NEUTRAL,
    STATUS_RETURNING,
    STATUS_STARTING,
    ClassificationResult,
    PunchEvent,
    neutral_result,
    no_detection_result,
    unknown_result,
)


@dataclass(frozen=True, slots=True)
class HeuristicSettings:
    """Parámetros iniciales. No representan una calibración validada.

    ``straight_elbow_gain`` y ``hook_elbow_gain_max`` son restricciones
    temporales, no umbrales de una postura aislada: un recto debe abrir el
    codo respecto a su propia guardia y un gancho debe conservarlo flexionado.
    Esto evita convertir cualquier desplazamiento lateral en gancho.
    """

    smooth_alpha: float = 0.5
    baseline_alpha: float = 0.15
    start_extension: float = 0.16
    commit_extension: float = 0.30
    straight_gain: float = 0.40
    straight_elbow: float = 110.0
    straight_elbow_gain: float = 25.0
    hook_lateral: float = 0.48
    hook_elbow_max: float = 145.0
    hook_elbow_gain_max: float = 20.0
    lateral_start: float = 0.22
    lateral_commit: float = 0.36
    retract_hysteresis: float = 0.14
    lateral_hysteresis: float = 0.12
    return_margin: float = 0.20
    return_lateral: float = 0.22
    min_wrist_speed: float = 0.25
    min_extension_rate: float = 0.80
    min_cycle_ms: int = 80
    max_cycle_ms: int = 1500
    start_timeout_ms: int = 400
    cooldown_ms: int = 280
    max_missing_frames: int = 6
    allow_early_confirm: bool = False
    confirm_on_slowdown: bool = False
    profile_name: str = "normal"


@dataclass(frozen=True, slots=True)
class _ArmObservation:
    extension: float
    elbow: float
    wrist: RelativePoint
    depth: float = 0.0


@dataclass
class _ArmDecision:
    phase: str
    event: PunchEvent | None = None
    unknown: bool = False
    tracking: bool = True
    candidate: str | None = None
    rejection: str | None = None
    extension: float | None = None
    speed: float = 0.0
    lateral: float = 0.0
    event_start_ms: int | None = None
    event_end_ms: int | None = None
    cooldown_until_ms: int | None = None
    classification_reason: str | None = None
    elbow: float | None = None
    peak_extension: float = 0.0
    peak_lateral: float = 0.0
    event_id: str | None = None


class _ArmTracker:
    def __init__(self, side: str, settings: HeuristicSettings) -> None:
        self.side = side
        self.settings = settings
        self.phase = STATUS_NEUTRAL
        self.smooth_extension: float | None = None
        self.smooth_elbow: float | None = None
        self.smooth_x: float | None = None
        self.smooth_y: float | None = None
        self.baseline_extension: float | None = None
        self.baseline_x: float | None = None
        self.frozen_extension = 0.0
        self.frozen_x = 0.0
        self.peak_extension = 0.0
        self.initial_elbow: float | None = None
        self.peak_elbow = 0.0
        self.peak_elbow_gain = 0.0
        self.peak_lateral = 0.0
        self.started_ms: int | None = None
        self.cooldown_until_ms = -1
        self.missing_frames = 0
        self.smooth_z: float | None = None
        self.previous_ms: int | None = None
        self.previous_extension: float | None = None
        self.previous_x: float | None = None
        self.previous_y: float | None = None
        self.previous_z: float | None = None
        self.last_speed = 0.0
        self.peak_speed = 0.0
        self.last_lateral = 0.0
        self.last_rejection: str | None = None
        self.last_classification_reason: str | None = None
        self.event_id: str | None = None
        self.emitted = False

    def reset(self) -> None:
        self.__init__(self.side, self.settings)

    def update(self, timestamp_ms: int, observation: _ArmObservation | None) -> _ArmDecision:
        if observation is None:
            self.missing_frames += 1
            if self.missing_frames > self.settings.max_missing_frames:
                unknown = self.phase != STATUS_NEUTRAL
                self.last_rejection = "seguimiento perdido" if unknown else None
                self.last_classification_reason = "seguimiento perdido" if unknown else None
                self._return_to_guard(timestamp_ms, cooldown=False)
                return self._pack(
                    STATUS_NEUTRAL,
                    unknown=unknown,
                    tracking=False,
                    rejection=self.last_rejection,
                )
            return self._pack(self.phase, tracking=self.phase != STATUS_NEUTRAL)

        self.missing_frames = 0
        self._smooth(observation)
        speed, extension_rate = self._rates(timestamp_ms)
        self.previous_ms = timestamp_ms
        self.previous_extension = self.smooth_extension
        self.previous_x = self.smooth_x
        self.previous_y = self.smooth_y
        self.previous_z = self.smooth_z

        assert self.smooth_extension is not None
        assert self.smooth_elbow is not None
        assert self.smooth_x is not None
        if self.baseline_extension is None or self.baseline_x is None:
            self.baseline_extension = self.smooth_extension
            self.baseline_x = self.smooth_x
            return self._pack(STATUS_NEUTRAL)

        lateral = abs(self.smooth_x - self.baseline_x)
        self.last_lateral = lateral
        extension_gain = self.smooth_extension - self.baseline_extension

        if self.phase == STATUS_NEUTRAL:
            started = self._movement_started(
                extension_gain,
                lateral,
                speed,
                extension_rate,
            )
            if started and timestamp_ms < self.cooldown_until_ms:
                self.last_rejection = "cooldown"
                return self._pack(STATUS_NEUTRAL, rejection="cooldown")
            if started:
                self.last_rejection = None
                self.last_classification_reason = None
                self.phase = STATUS_STARTING
                self.started_ms = timestamp_ms
                self.event_id = f"{self.side}:{timestamp_ms}"
                self.emitted = False
                self.frozen_extension = self.baseline_extension
                self.frozen_x = self.baseline_x
                self.peak_extension = self.smooth_extension
                self.initial_elbow = self.smooth_elbow
                self.peak_elbow = self.smooth_elbow
                self.peak_elbow_gain = 0.0
                self.peak_lateral = abs(self.smooth_x - self.frozen_x)
                self.peak_speed = speed
            elif self._near_guard(extension_gain, lateral):
                self._update_baseline()
            return self._pack(self.phase)

        previous_peak_extension = self.peak_extension
        previous_peak_lateral = self.peak_lateral
        self.peak_extension = max(self.peak_extension, self.smooth_extension)
        self.peak_elbow = max(self.peak_elbow, self.smooth_elbow)
        if self.initial_elbow is not None:
            self.peak_elbow_gain = max(
                self.peak_elbow_gain,
                self.smooth_elbow - self.initial_elbow,
            )
        self.peak_lateral = max(self.peak_lateral, abs(self.smooth_x - self.frozen_x))
        self.peak_speed = max(self.peak_speed, speed)
        still_growing = (
            self.peak_extension > previous_peak_extension + 1e-4
            or self.peak_lateral > previous_peak_lateral + 1e-4
        )
        elapsed = 0 if self.started_ms is None else timestamp_ms - self.started_ms
        if elapsed > self.settings.max_cycle_ms:
            self.last_rejection = "ciclo excedido"
            self.last_classification_reason = "ciclo excedido"
            self._return_to_guard(timestamp_ms, cooldown=False)
            return self._pack(STATUS_NEUTRAL, unknown=True, rejection="ciclo excedido")

        if self.phase == STATUS_STARTING:
            committed = (
                self.smooth_extension >= self.frozen_extension + self.settings.commit_extension
                or self.peak_lateral >= self.settings.lateral_commit
            )
            if committed:
                self.phase = STATUS_EXTENDING
            elif elapsed > self.settings.start_timeout_ms:
                self.last_rejection = "sin compromiso"
                self.last_classification_reason = "activacion insuficiente"
                self._return_to_guard(timestamp_ms, cooldown=False)
            return self._pack(self.phase, rejection=self.last_rejection)

        retracting = (
            self.smooth_extension <= self.peak_extension - self.settings.retract_hysteresis
            or (
                self.peak_lateral >= self.settings.hook_lateral
                and abs(self.smooth_x - self.frozen_x)
                <= self.peak_lateral - self.settings.lateral_hysteresis
            )
        )
        committed = (
            self.peak_extension >= self.frozen_extension + self.settings.commit_extension
            or self.peak_lateral >= self.settings.lateral_commit
        )
        had_motion = self.peak_speed >= self.settings.min_wrist_speed
        slowing = had_motion and speed <= max(0.05, self.settings.min_wrist_speed * 0.85)
        ready_to_close = (
            elapsed >= self.settings.min_cycle_ms
            and committed
            and (retracting or (self.settings.confirm_on_slowdown and slowing))
            and not still_growing
        )
        if self.settings.allow_early_confirm and ready_to_close:
            completed = self._complete_cycle(timestamp_ms)
            if completed is not None:
                return completed

        if self.phase == STATUS_EXTENDING:
            if retracting:
                self.phase = STATUS_RETURNING
            return self._pack(self.phase)

        gain = max(0.0, self.peak_extension - self.frozen_extension)
        returned = (
            self.smooth_extension
            <= self.frozen_extension + max(self.settings.return_margin, 0.45 * gain)
            and abs(self.smooth_x - self.frozen_x)
            <= max(self.settings.return_lateral, 0.45 * self.peak_lateral)
        )
        if returned and elapsed >= self.settings.min_cycle_ms and not still_growing:
            completed = self._complete_cycle(timestamp_ms)
            if completed is not None:
                return completed
            self.last_rejection = "tipo no distinguible"
            return self._pack(
                STATUS_NEUTRAL,
                unknown=True,
                rejection="tipo no distinguible",
            )
        return self._pack(self.phase)

    def _smooth(self, observation: _ArmObservation) -> None:
        alpha = self.settings.smooth_alpha
        self.smooth_extension = _blend(self.smooth_extension, observation.extension, alpha)
        self.smooth_elbow = _blend(self.smooth_elbow, observation.elbow, alpha)
        self.smooth_x = _blend(self.smooth_x, observation.wrist.x, alpha)
        self.smooth_y = _blend(self.smooth_y, observation.wrist.y, alpha)
        self.smooth_z = _blend(self.smooth_z, observation.depth, alpha)

    def _rates(self, timestamp_ms: int) -> tuple[float, float]:
        if (
            self.previous_ms is None
            or self.previous_x is None
            or self.previous_y is None
            or self.previous_extension is None
            or self.smooth_x is None
            or self.smooth_y is None
            or self.smooth_extension is None
        ):
            return 0.0, 0.0
        dt_seconds = (timestamp_ms - self.previous_ms) / 1000.0
        if dt_seconds <= 0.0:
            return 0.0, 0.0
        depth_delta = 0.0 if self.smooth_z is None or self.previous_z is None else self.smooth_z - self.previous_z
        speed = (
            (self.smooth_x - self.previous_x) ** 2
            + (self.smooth_y - self.previous_y) ** 2
            + depth_delta ** 2
        ) ** 0.5 / dt_seconds
        self.last_speed = speed
        extension_rate = (self.smooth_extension - self.previous_extension) / dt_seconds
        return speed, extension_rate

    def _movement_started(
        self,
        extension_gain: float,
        lateral: float,
        speed: float,
        extension_rate: float,
    ) -> bool:
        displaced = (
            extension_gain >= self.settings.start_extension
            or lateral >= self.settings.lateral_start
        )
        energetic = (
            speed >= self.settings.min_wrist_speed
            or extension_rate >= self.settings.min_extension_rate
        )
        return displaced and energetic

    def _near_guard(self, extension_gain: float, lateral: float) -> bool:
        return (
            extension_gain < self.settings.start_extension * 0.5
            and lateral < self.settings.lateral_start * 0.5
        )

    def _update_baseline(self) -> None:
        assert self.smooth_extension is not None and self.smooth_x is not None
        assert self.baseline_extension is not None and self.baseline_x is not None
        alpha = self.settings.baseline_alpha
        self.baseline_extension = _blend(self.baseline_extension, self.smooth_extension, alpha)
        self.baseline_x = _blend(self.baseline_x, self.smooth_x, alpha)

    def _snap_baseline(self) -> None:
        """La guardia de una combinación es la postura al frenar, no la inicial."""

        if self.smooth_extension is None or self.smooth_x is None:
            return
        self.baseline_extension = self.smooth_extension
        self.baseline_x = self.smooth_x

    def _complete_cycle(self, timestamp_ms: int) -> _ArmDecision | None:
        """Cierra el ciclo una sola vez, con la clase de toda la trayectoria."""

        if self.emitted:
            return self._pack(STATUS_RETURNING)
        event = self._classify()
        if event is None:
            self._return_to_guard(timestamp_ms, cooldown=False)
            return None
        event = replace(event, ended_ms=timestamp_ms)
        self.emitted = True
        event_start_ms = self.started_ms
        self._snap_baseline()
        self._return_to_guard(timestamp_ms, cooldown=True)
        return self._pack(
            STATUS_COMPLETED,
            event=event,
            event_start_ms=event_start_ms,
            event_end_ms=timestamp_ms,
        )

    def _pack(
        self,
        phase: str,
        *,
        event: PunchEvent | None = None,
        unknown: bool = False,
        tracking: bool = True,
        rejection: str | None = None,
        event_start_ms: int | None = None,
        event_end_ms: int | None = None,
    ) -> _ArmDecision:
        candidate = None
        if event is None and phase not in (STATUS_NEUTRAL, STATUS_COMPLETED):
            tentative = self._classify() if self.started_ms is not None else None
            candidate = tentative.display_name if tentative is not None else "MOVIMIENTO"
        return _ArmDecision(
            phase,
            event=event,
            unknown=unknown,
            tracking=tracking,
            candidate=candidate,
            rejection=rejection,
            extension=self.smooth_extension,
            speed=self.last_speed,
            lateral=self.last_lateral,
            event_start_ms=event_start_ms,
            event_end_ms=event_end_ms,
            cooldown_until_ms=self.cooldown_until_ms,
            classification_reason=self.last_classification_reason,
            elbow=self.smooth_elbow,
            peak_extension=self.peak_extension,
            peak_lateral=self.peak_lateral,
            event_id=self.event_id,
        )

    def _return_to_guard(self, timestamp_ms: int, *, cooldown: bool = True) -> None:
        self.phase = STATUS_NEUTRAL
        self.started_ms = None
        self.event_id = None
        self.emitted = False
        self.initial_elbow = None
        self.peak_extension = 0.0
        self.peak_elbow = 0.0
        self.peak_elbow_gain = 0.0
        self.peak_lateral = 0.0
        self.peak_speed = 0.0
        if cooldown:
            self.cooldown_until_ms = timestamp_ms + self.settings.cooldown_ms

    def _classify(self) -> PunchEvent | None:
        extension_gain = self.peak_extension - self.frozen_extension
        elbow_gain = self.peak_elbow_gain
        wide = self.peak_lateral >= self.settings.hook_lateral
        self.last_classification_reason = None

        # La evidencia de recto tiene prioridad sobre la anchura aparente. En
        # una cámara frontal el movimiento hacia profundidad puede proyectarse
        # lateralmente; la apertura progresiva del codo y la extensión radial
        # son la evidencia más estable disponible para un recto.
        straight = (
            extension_gain >= self.settings.straight_gain
            and (
                self.peak_elbow >= self.settings.straight_elbow
                or elbow_gain >= self.settings.straight_elbow_gain
            )
        )
        hook = (
            wide
            and self.peak_elbow <= self.settings.hook_elbow_max
            and elbow_gain <= self.settings.hook_elbow_gain_max
        )
        if straight and hook:
            # Un gancho real acumula anchura; no lo conviertas en jab porque
            # el codo también se abrió durante el mismo ciclo.
            if self.peak_lateral >= max(extension_gain, self.settings.hook_lateral):
                straight = False
            else:
                hook = False
        if straight:
            self.last_classification_reason = (
                "recto: extension y apertura progresiva del codo"
            )
            label_id = 1 if self.side == "left" else 2
        elif hook:
            self.last_classification_reason = (
                "gancho: trayectoria lateral con codo flexionado"
            )
            label_id = 3 if self.side == "left" else 4
        elif self.settings.confirm_on_slowdown and (
            extension_gain >= self.settings.start_extension
            or self.peak_lateral >= self.settings.lateral_start
        ):
            if self.peak_lateral >= max(extension_gain, self.settings.lateral_start):
                self.last_classification_reason = "candidato lateral sin evidencia completa"
                label_id = 3 if self.side == "left" else 4
            else:
                self.last_classification_reason = "candidato frontal sin evidencia completa"
                label_id = 1 if self.side == "left" else 2
        else:
            self.last_classification_reason = "geometria ambigua o insuficiente"
            return None
        movement = MOVEMENTS[label_id]
        return PunchEvent(
            movement.label_id,
            movement.display_name,
            self.side,
            event_id=self.event_id,
            started_ms=self.started_ms,
            confirmed=True,
            source="heuristic",
        )


def _blend(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return value
    return (1.0 - alpha) * previous + alpha * value


def _arm_observation(
    features: BiomechanicalFeatures,
    side: str,
    pose: PoseObservation | None = None,
) -> _ArmObservation | None:
    spatial = None if pose is None else _observation_from_pose(pose, side)
    if spatial is not None:
        return spatial
    if side == "left":
        extension = features.left_shoulder_wrist_distance
        elbow = features.left_elbow_angle
        wrist = features.left_wrist_relative
    else:
        extension = features.right_shoulder_wrist_distance
        elbow = features.right_elbow_angle
        wrist = features.right_wrist_relative
    if extension is None or elbow is None or wrist is None:
        return None
    return _ArmObservation(extension, elbow, wrist)


def _observation_from_pose(
    pose: PoseObservation,
    side: str,
) -> _ArmObservation | None:
    """Extrae geometría sin rellenar landmarks ni profundidad.

    MediaPipe entrega coordenadas de imagen y coordenadas mundiales. Las
    mundiales solo se usan cuando hombros, codo y muñeca del brazo están
    presentes y superan los niveles de visibilidad/presencia usados por el
    propio extractor 2D. Si no, se conserva la geometría 2D y se descarta z;
    una z de baja calidad no debe fabricar un recto hacia cámara.
    """

    side_elbow = (
        LandmarkId.LEFT_ELBOW if side == "left" else LandmarkId.RIGHT_ELBOW
    )
    side_wrist = (
        LandmarkId.LEFT_WRIST if side == "left" else LandmarkId.RIGHT_WRIST
    )
    required = (
        LandmarkId.LEFT_SHOULDER,
        LandmarkId.RIGHT_SHOULDER,
        side_elbow,
        side_wrist,
    )
    source: tuple[LandmarkObservation, ...] | None = None
    use_world = False
    if _source_is_reliable(pose.world_landmarks, required):
        source = pose.world_landmarks
        use_world = True
    elif _source_is_reliable(pose.landmarks, required):
        source = pose.landmarks
    if source is None:
        return None
    left_shoulder = source[int(LandmarkId.LEFT_SHOULDER)]
    right_shoulder = source[int(LandmarkId.RIGHT_SHOULDER)]
    if side == "left":
        shoulder = left_shoulder
        elbow = source[int(LandmarkId.LEFT_ELBOW)]
        wrist = source[int(LandmarkId.LEFT_WRIST)]
    else:
        shoulder = right_shoulder
        elbow = source[int(LandmarkId.RIGHT_ELBOW)]
        wrist = source[int(LandmarkId.RIGHT_WRIST)]
    if not all(_finite_point(point) for point in (left_shoulder, right_shoulder, shoulder, elbow, wrist)):
        return None

    local = _torso_coordinates(pose, source, wrist, use_depth=use_world)
    if local is None:
        return None
    local_x, local_y, local_z, scale = local
    angle = _angle3(shoulder, elbow, wrist) if use_world else _angle2(shoulder, elbow, wrist)
    if angle is None:
        return None
    return _ArmObservation(
        extension=_distance(shoulder, wrist, use_depth=use_world) / scale,
        elbow=angle,
        wrist=RelativePoint(local_x, local_y),
        depth=local_z,
    )


def _torso_coordinates(
    pose: PoseObservation,
    source: tuple[LandmarkObservation, ...],
    wrist: LandmarkObservation,
    *,
    use_depth: bool,
) -> tuple[float, float, float, float] | None:
    """Muñeca en el plano del torso, en anchos de hombro.

    El origen es el centro de los hombros. El eje lateral une los hombros y el
    eje vertical apunta hacia las caderas cuando existen. Trasladar o girar
    todo el cuerpo no cambia estas coordenadas.
    """

    left_shoulder = source[int(LandmarkId.LEFT_SHOULDER)]
    right_shoulder = source[int(LandmarkId.RIGHT_SHOULDER)]
    axis_x = _subtract(right_shoulder, left_shoulder, use_depth=use_depth)
    scale = _norm(axis_x)
    if scale <= 1e-4:
        return None
    axis_x = _scale(axis_x, 1.0 / scale)
    center = _midpoint(left_shoulder, right_shoulder, use_depth=use_depth)
    down = _body_down(pose, source, center, axis_x, use_depth=use_depth)
    down = _reject(down, axis_x)
    down_norm = _norm(down)
    if down_norm <= 1e-4:
        return None
    axis_y = _scale(down, 1.0 / down_norm)
    axis_z = _cross(axis_x, axis_y)
    wrist_point = _coordinates(wrist, use_depth=use_depth)
    offset = tuple(wrist_point[index] - center[index] for index in range(3))
    return (
        _dot(offset, axis_x) / scale,
        _dot(offset, axis_y) / scale,
        _dot(offset, axis_z) / scale,
        scale,
    )


def _body_down(
    pose: PoseObservation,
    source: tuple[LandmarkObservation, ...],
    center: tuple[float, float, float],
    axis_x: tuple[float, float, float],
    *,
    use_depth: bool,
) -> tuple[float, float, float]:
    hip_index = int(LandmarkId.LEFT_HIP)
    if len(source) > int(LandmarkId.RIGHT_HIP):
        left_hip = source[hip_index]
        right_hip = source[int(LandmarkId.RIGHT_HIP)]
        if (
            left_hip.is_reliable(_POSE_MIN_VISIBILITY, _POSE_MIN_PRESENCE)
            and right_hip.is_reliable(_POSE_MIN_VISIBILITY, _POSE_MIN_PRESENCE)
        ):
            hip_center = _midpoint(left_hip, right_hip, use_depth=use_depth)
            return (
                hip_center[0] - center[0],
                hip_center[1] - center[1],
                hip_center[2] - center[2],
            )
    del pose
    fallback = (0.0, 1.0, 0.0)
    return _reject(fallback, axis_x) if _norm(_reject(fallback, axis_x)) > 1e-4 else (0.0, 0.0, 1.0)


_POSE_MIN_VISIBILITY = 0.5
_POSE_MIN_PRESENCE = 0.5


def _source_is_reliable(
    source: tuple[LandmarkObservation, ...],
    required: tuple[LandmarkId, ...],
) -> bool:
    return bool(source) and all(
        index < len(source)
        and source[int(index)].is_reliable(_POSE_MIN_VISIBILITY, _POSE_MIN_PRESENCE)
        for index in required
    )


def _coordinates(point: LandmarkObservation, *, use_depth: bool) -> tuple[float, float, float]:
    return (point.x, point.y, point.z if use_depth else 0.0)


def _subtract(
    first: LandmarkObservation,
    second: LandmarkObservation,
    *,
    use_depth: bool = True,
) -> tuple[float, float, float]:
    first_coordinates = _coordinates(first, use_depth=use_depth)
    second_coordinates = _coordinates(second, use_depth=use_depth)
    return tuple(first_coordinates[index] - second_coordinates[index] for index in range(3))


def _midpoint(
    first: LandmarkObservation,
    second: LandmarkObservation,
    *,
    use_depth: bool = True,
) -> tuple[float, float, float]:
    first_coordinates = _coordinates(first, use_depth=use_depth)
    second_coordinates = _coordinates(second, use_depth=use_depth)
    return tuple(
        (first_coordinates[index] + second_coordinates[index]) / 2.0
        for index in range(3)
    )


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _scale(vector: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _reject(
    vector: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> tuple[float, float, float]:
    projected = _dot(vector, axis)
    return (
        vector[0] - projected * axis[0],
        vector[1] - projected * axis[1],
        vector[2] - projected * axis[2],
    )


def _finite_point(point: LandmarkObservation) -> bool:
    return point.is_finite


def _distance(
    first: LandmarkObservation,
    second: LandmarkObservation,
    *,
    use_depth: bool,
) -> float:
    return math.dist(
        _coordinates(first, use_depth=use_depth),
        _coordinates(second, use_depth=use_depth),
    )


def _angle2(
    first: LandmarkObservation,
    vertex: LandmarkObservation,
    third: LandmarkObservation,
) -> float | None:
    vector_a = (first.x - vertex.x, first.y - vertex.y)
    vector_b = (third.x - vertex.x, third.y - vertex.y)
    norm_a = math.hypot(*vector_a)
    norm_b = math.hypot(*vector_b)
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return None
    cosine = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    return angle if math.isfinite(angle) else None


def _angle3(
    first: LandmarkObservation,
    vertex: LandmarkObservation,
    third: LandmarkObservation,
) -> float | None:
    vector_a = (first.x - vertex.x, first.y - vertex.y, first.z - vertex.z)
    vector_b = (third.x - vertex.x, third.y - vertex.y, third.z - vertex.z)
    norm_a = math.sqrt(vector_a[0] ** 2 + vector_a[1] ** 2 + vector_a[2] ** 2)
    norm_b = math.sqrt(vector_b[0] ** 2 + vector_b[1] ** 2 + vector_b[2] ** 2)
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return None
    cosine = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1] + vector_a[2] * vector_b[2]) / (
        norm_a * norm_b
    )
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


class HeuristicClassifier:
    """Clasificador provisional. No es un modelo NEAT ni un modelo externo."""

    name = "heuristic"
    provisional = True

    def __init__(self, settings: HeuristicSettings | None = None) -> None:
        self.settings = settings or HeuristicSettings()
        self._left = _ArmTracker("left", self.settings)
        self._right = _ArmTracker("right", self.settings)
        self.telemetry = {
            "left_extension": None,
            "right_extension": None,
            "left_elbow_angle": None,
            "right_elbow_angle": None,
            "left_speed": 0.0,
            "right_speed": 0.0,
            "left_lateral": 0.0,
            "right_lateral": 0.0,
            "left_phase": STATUS_NEUTRAL,
            "right_phase": STATUS_NEUTRAL,
            "left_event_start_ms": None,
            "left_event_end_ms": None,
            "right_event_start_ms": None,
            "right_event_end_ms": None,
            "left_cooldown_until_ms": -1,
            "right_cooldown_until_ms": -1,
            "left_rejection": None,
            "right_rejection": None,
        }

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()

    def apply_settings(self, settings: HeuristicSettings) -> None:
        self.settings = settings
        self._left.settings = settings
        self._right.settings = settings

    def set_sensitivity(self, name: str) -> None:
        from app.trainer.sensitivity import profile_settings

        self.apply_settings(profile_settings(name))

    def update(
        self,
        features: BiomechanicalFeatures,
        *,
        pose_detected: bool,
        pose: PoseObservation | None = None,
        frame: object | None = None,
    ) -> ClassificationResult:
        started = time.perf_counter()
        if not pose_detected:
            pose = None
        left = self._left.update(
            features.timestamp_ms,
            None if not pose_detected else _arm_observation(features, "left", pose),
        )
        right = self._right.update(
            features.timestamp_ms,
            None if not pose_detected else _arm_observation(features, "right", pose),
        )
        self.telemetry = {
            "left_extension": left.extension,
            "right_extension": right.extension,
            "left_elbow_angle": left.elbow,
            "right_elbow_angle": right.elbow,
            "left_lateral": left.lateral,
            "right_lateral": right.lateral,
            "left_speed": left.speed,
            "right_speed": right.speed,
            "left_phase": left.phase,
            "right_phase": right.phase,
            "left_event_start_ms": left.event_start_ms,
            "left_event_end_ms": left.event_end_ms,
            "right_event_start_ms": right.event_start_ms,
            "right_event_end_ms": right.event_end_ms,
            "left_cooldown_until_ms": left.cooldown_until_ms,
            "right_cooldown_until_ms": right.cooldown_until_ms,
            "left_rejection": left.rejection,
            "right_rejection": right.rejection,
        }
        latency_ms = (time.perf_counter() - started) * 1000.0
        rejection = left.rejection or right.rejection
        candidate = _preferred_candidate(left, right)
        moving = _is_moving(left, right)
        debug_lines = (
            f"I {left.phase} ext {_fmt(left.extension)} vel {left.speed:.2f} lat {left.lateral:.2f}",
            f"D {right.phase} ext {_fmt(right.extension)} vel {right.speed:.2f} lat {right.lateral:.2f}",
            f"rechazo {rejection or '-'}",
        )
        if not left.tracking and not right.tracking:
            return replace(
                no_detection_result(self.name),
                rejection=rejection,
                latency_ms=latency_ms,
                sensitivity=self.settings.profile_name,
                debug_lines=debug_lines,
                diagnostics=_decision_diagnostics(left, right),
            )

        events = _select_events(left, right)
        if events:
            latest = events[-1]
            return ClassificationResult(
                source=self.name,
                status=STATUS_COMPLETED,
                label_id=latest.label_id,
                display_name=latest.display_name,
                events=events,
                active_arm=latest.arm,
                latency_ms=latency_ms,
                sensitivity=self.settings.profile_name,
                debug_lines=debug_lines,
                diagnostics=_decision_diagnostics(left, right),
            )
        if left.unknown or right.unknown:
            arm = "left" if left.unknown else "right"
            return replace(
                unknown_result(self.name, arm),
                candidate=candidate,
                rejection=rejection,
                latency_ms=latency_ms,
                sensitivity=self.settings.profile_name,
                debug_lines=debug_lines,
                diagnostics=_decision_diagnostics(left, right),
            )

        phase, arm = _dominant_phase(left, right)
        if phase == STATUS_NEUTRAL:
            return replace(
                neutral_result(self.name),
                candidate=candidate,
                rejection=rejection,
                latency_ms=latency_ms,
                sensitivity=self.settings.profile_name,
                moving=moving,
                debug_lines=debug_lines,
                diagnostics=_decision_diagnostics(left, right),
            )
        return ClassificationResult(
            source=self.name,
            status=phase,
            label_id=None,
            display_name=candidate or phase.upper(),
            active_arm=arm,
            candidate=candidate,
            rejection=rejection,
            latency_ms=latency_ms,
            sensitivity=self.settings.profile_name,
            moving=True,
            debug_lines=debug_lines,
            diagnostics=_decision_diagnostics(left, right),
        )


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _decision_diagnostics(left: _ArmDecision, right: _ArmDecision) -> dict[str, object]:
    """Expone la causa terminal por brazo sin convertirla en una etiqueta."""

    return {
        "left_phase": left.phase,
        "right_phase": right.phase,
        "left_rejection": left.rejection,
        "right_rejection": right.rejection,
        "left_event_start_ms": left.event_start_ms,
        "left_event_end_ms": left.event_end_ms,
        "right_event_start_ms": right.event_start_ms,
        "right_event_end_ms": right.event_end_ms,
        "left_cooldown_until_ms": left.cooldown_until_ms,
        "right_cooldown_until_ms": right.cooldown_until_ms,
        "left_tracking": left.tracking,
        "right_tracking": right.tracking,
        "left_movement_detected": left.phase != STATUS_NEUTRAL,
        "right_movement_detected": right.phase != STATUS_NEUTRAL,
        "left_candidate": left.candidate,
        "right_candidate": right.candidate,
        "left_classification_reason": left.classification_reason,
        "right_classification_reason": right.classification_reason,
        "left_speed": left.speed,
        "right_speed": right.speed,
        "left_lateral": left.lateral,
        "right_lateral": right.lateral,
        "left_extension": left.extension,
        "right_extension": right.extension,
        "left_elbow_angle": left.elbow,
        "right_elbow_angle": right.elbow,
    }


def _is_moving(left: _ArmDecision, right: _ArmDecision) -> bool:
    return left.phase != STATUS_NEUTRAL or right.phase != STATUS_NEUTRAL


def _motion_mass(decision: _ArmDecision) -> float:
    return max(0.0, decision.peak_extension) + max(0.0, decision.peak_lateral)


def _select_events(left: _ArmDecision, right: _ArmDecision) -> tuple[PunchEvent, ...]:
    """Un fotograma no entrega dos golpes si uno es compensación de guardia."""

    if left.event is None and right.event is None:
        return ()
    if left.event is None:
        return (right.event,)
    if right.event is None:
        return (left.event,)
    left_mass = _motion_mass(left)
    right_mass = _motion_mass(right)
    if left_mass >= max(0.08, right_mass * 2.0) and right_mass < left_mass:
        return (left.event,)
    if right_mass >= max(0.08, left_mass * 2.0) and left_mass < right_mass:
        return (right.event,)
    return (left.event, right.event)


def _preferred_candidate(left: _ArmDecision, right: _ArmDecision) -> str | None:
    if PHASE_RANK[right.phase] > PHASE_RANK[left.phase]:
        return right.candidate or left.candidate
    return left.candidate or right.candidate


def _dominant_phase(left: _ArmDecision, right: _ArmDecision) -> tuple[str, str | None]:
    if PHASE_RANK[right.phase] > PHASE_RANK[left.phase]:
        return right.phase, "right"
    if PHASE_RANK[left.phase] > PHASE_RANK[right.phase]:
        return left.phase, "left"
    if left.phase == STATUS_NEUTRAL:
        return STATUS_NEUTRAL, None
    return left.phase, "left"
