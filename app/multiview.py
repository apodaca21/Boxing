"""Evidencia auxiliar de varias cámaras sin asumir una calibración inexistente."""

from __future__ import annotations

from dataclasses import dataclass, replace
from app.camera import CameraRig, RigFrame
from app.config import AppConfig
from app.pose_tracker import LandmarkId, PoseObservation, PoseTracker
from app.trainer.results import ClassificationResult


@dataclass(frozen=True, slots=True)
class ViewQuality:
    role: str
    index: int
    detected: bool
    arm_quality: float


def _arm_quality(pose: PoseObservation) -> float:
    required = (
        LandmarkId.LEFT_SHOULDER,
        LandmarkId.RIGHT_SHOULDER,
        LandmarkId.LEFT_WRIST,
        LandmarkId.RIGHT_WRIST,
    )
    if not pose.detected:
        return 0.0
    reliable = sum(
        1
        for landmark_id in required
        if (landmark := pose.landmark(landmark_id)) is not None
        and landmark.is_finite
        and landmark.visibility >= 0.5
        and landmark.presence >= 0.5
    )
    return reliable / len(required)


class _DepthMotionCue:
    """Señal temporal de desplazamiento observada por una vista lateral.

    No convierte coordenadas de una cámara en profundidad métrica. Solo mide
    si la muñeca se separó lateralmente de su hombro en la vista auxiliar,
    suficiente para una confirmación temporal después de una calibración física
    razonable de la cámara.
    """

    def __init__(self) -> None:
        self._baseline: dict[str, float | None] = {"left": None, "right": None}
        self._signal: dict[str, tuple[int, float]] = {"left": (-1, 0.0), "right": (-1, 0.0)}

    def observe(self, timestamp_ms: int, pose: PoseObservation) -> None:
        for arm, shoulder_id, wrist_id in (
            ("left", LandmarkId.LEFT_SHOULDER, LandmarkId.LEFT_WRIST),
            ("right", LandmarkId.RIGHT_SHOULDER, LandmarkId.RIGHT_WRIST),
        ):
            shoulder = pose.landmark(shoulder_id)
            wrist = pose.landmark(wrist_id)
            if (
                shoulder is None
                or wrist is None
                or not shoulder.is_finite
                or not wrist.is_finite
                or shoulder.visibility < 0.5
                or wrist.visibility < 0.5
            ):
                continue
            relative_x = wrist.x - shoulder.x
            baseline = self._baseline[arm]
            if baseline is None:
                self._baseline[arm] = relative_x
                continue
            signal = abs(relative_x - baseline)
            previous_time, previous_signal = self._signal[arm]
            # Decae lentamente para tolerar el desfase entre cámaras, pero no
            # permite que una señal antigua confirme un golpe nuevo.
            if previous_time < 0 or timestamp_ms - previous_time > 450:
                previous_signal = 0.0
            self._signal[arm] = (timestamp_ms, max(signal, previous_signal * 0.82))
            if signal < 0.04:
                self._baseline[arm] = baseline * 0.97 + relative_x * 0.03

    def confirms(self, arm: str | None, timestamp_ms: int) -> bool:
        arms = (arm,) if arm in {"left", "right"} else ("left", "right")
        return any(
            timestamp_ms - observed_ms <= 350 and signal >= 0.08
            for selected in arms
            for observed_ms, signal in (self._signal[selected],)
            if observed_ms >= 0
        )


class MultiViewMonitor:
    """Ejecuta pose en auxiliares y filtra eventos solo en modo strict."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._trackers: dict[str, PoseTracker] = {}
        self._depth = _DepthMotionCue()
        self.poses: dict[str, PoseObservation] = {}
        self.quality: tuple[ViewQuality, ...] = ()

    def open(self, rig: CameraRig) -> None:
        for role in rig.role_indices:
            # Las vistas izquierda/derecha se capturan para futuras fusiones,
            # pero no gastan inferencia adicional hasta que se elige depth.
            if role != "depth":
                continue
            self._trackers[role] = PoseTracker(
                model_path=self._config.model_path,
                min_detection_confidence=self._config.min_pose_detection_confidence,
                min_presence_confidence=self._config.min_pose_presence_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )

    def observe(self, frame: RigFrame, rig: CameraRig) -> dict[str, PoseObservation]:
        self.poses = {}
        rows: list[ViewQuality] = []
        for role, tracker in self._trackers.items():
            packet = frame.by_role.get(role)
            if packet is None:
                continue
            pose = tracker.detect(packet.frame, packet.timestamp_ms)
            self.poses[role] = pose
            rows.append(
                ViewQuality(
                    role=role,
                    index=rig.role_indices[role],
                    detected=pose.detected,
                    arm_quality=_arm_quality(pose),
                )
            )
            if role == "depth":
                self._depth.observe(packet.timestamp_ms, pose)
        self.quality = tuple(rows)
        return dict(self.poses)

    def filter_result(self, result: ClassificationResult, timestamp_ms: int) -> ClassificationResult:
        if self._config.depth_gate != "strict" or not result.events:
            return result
        if not self._trackers or "depth" not in self._trackers:
            return replace(
                result,
                events=(),
                status="incierto",
                display_name="MOVIMIENTO NO CONFIRMADO",
                rejection="no hay cámara depth disponible",
                diagnostics={**result.diagnostics, "depth_gate": "missing"},
            )
        unconfirmed = tuple(
            event
            for event in result.events
            if not self._depth.confirms(event.arm, timestamp_ms)
        )
        if not unconfirmed:
            return replace(
                result,
                diagnostics={**result.diagnostics, "depth_gate": "confirmed"},
            )
        return replace(
            result,
            events=(),
            status="incierto",
            label_id=None,
            display_name="MOVIMIENTO NO CONFIRMADO",
            rejection="sin confirmación de la cámara depth",
            diagnostics={
                **result.diagnostics,
                "depth_gate": "rejected",
                "depth_unconfirmed_arms": tuple(event.arm for event in unconfirmed),
            },
        )

    def summary(self) -> str:
        if not self.quality:
            return ""
        return "  ".join(
            f"{row.role} {'OK' if row.detected else 'sin pose'} {row.arm_quality:.2f}"
            for row in self.quality
        )

    def close(self) -> None:
        for tracker in self._trackers.values():
            tracker.close()
        self._trackers.clear()
        self.poses.clear()
        self.quality = ()

