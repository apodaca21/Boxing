"""Evidencia auxiliar de varias cámaras sin asumir una calibración métrica."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from app.camera import CameraRig, RigFrame
from app.config import AppConfig
from app.dataset_manager import MOVEMENTS
from app.geometry import extract_biomechanical_features
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation, PoseTracker
from app.trainer.heuristic import HeuristicClassifier
from app.trainer.results import STATUS_COMPLETED, ClassificationResult, PunchEvent
from app.trainer.sensitivity import profile_settings


@dataclass(frozen=True, slots=True)
class ViewQuality:
    role: str
    index: int
    detected: bool
    arm_quality: float


@dataclass(frozen=True, slots=True)
class DepthSignals:
    lateral: float
    reach: float
    age_ms: int
    approach: float = 0.0


@dataclass(frozen=True, slots=True)
class DepthReadiness:
    """Estado visual para saber si la vista auxiliar ya está lista."""

    camera_ok: bool = False
    detected: bool = False
    focus_score: float = 0.0
    brightness_ok: bool = False
    arm_quality: float = 0.0
    stable_frames: int = 0
    ready: bool = False
    status: str = "DEPTH SIN CÁMARA"


DEPTH_MIN_SHARPNESS = 45.0
DEPTH_MIN_ARM_QUALITY = 0.75
DEPTH_REQUIRED_STABLE_FRAMES = 8
DEPTH_EVENT_DEDUPE_MS = 400
DEPTH_HOLD_SHARPNESS = 25.0
DEPTH_HOLD_ARM_QUALITY = 0.50


def _focus_measure(frame: NDArray[np.uint8]) -> float:
    """Mide nitidez con la varianza del Laplaciano, sin guardar imágenes."""

    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim < 2:
        return 0.0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except cv2.error:
        return 0.0


def _focus_score(sharpness: float) -> float:
    """Convierte nitidez en un valor 0..1 solo para el indicador visual."""

    return float(np.clip((sharpness - 15.0) / 135.0, 0.0, 1.0))


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


def _arm_depth_offset(
    pose: PoseObservation,
    shoulder_id: LandmarkId,
    wrist_id: LandmarkId,
    shoulder: LandmarkObservation,
    wrist: LandmarkObservation,
) -> float:
    """Usa z de world landmarks si existen; si no, el z de la imagen."""

    world = pose.world_landmarks
    index_s = int(shoulder_id)
    index_w = int(wrist_id)
    if world and index_s < len(world) and index_w < len(world):
        world_s = world[index_s]
        world_w = world[index_w]
        if world_s.is_finite and world_w.is_finite:
            return world_w.z - world_s.z
    return wrist.z - shoulder.z


class _DepthMotionCue:
    """Señal temporal de una webcam lateral que apoya a la frontal.

    No convierte coordenadas en profundidad métrica. Mide alcance (hombro-muñeca)
    y desplazamiento lateral en la vista auxiliar. Un recto suele crecer en
    alcance; un gancho suele acumular más lateral con poco alcance.
    """

    def __init__(self) -> None:
        self._baseline_x: dict[str, float | None] = {"left": None, "right": None}
        self._baseline_reach: dict[str, float | None] = {"left": None, "right": None}
        self._baseline_z: dict[str, float | None] = {"left": None, "right": None}
        self._signal: dict[str, tuple[int, float, float, float]] = {
            "left": (-1, 0.0, 0.0, 0.0),
            "right": (-1, 0.0, 0.0, 0.0),
        }

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
            reach = float(np.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y))
            relative_z = _arm_depth_offset(pose, shoulder_id, wrist_id, shoulder, wrist)
            baseline_x = self._baseline_x[arm]
            baseline_reach = self._baseline_reach[arm]
            baseline_z = self._baseline_z[arm]
            if baseline_x is None or baseline_reach is None or baseline_z is None:
                self._baseline_x[arm] = relative_x
                self._baseline_reach[arm] = reach
                self._baseline_z[arm] = relative_z
                continue
            lateral = abs(relative_x - baseline_x)
            reach_gain = max(0.0, reach - baseline_reach)
            approach = abs(relative_z - baseline_z)
            previous_time, previous_lateral, previous_reach, previous_approach = self._signal[arm]
            if previous_time < 0 or timestamp_ms - previous_time > 450:
                previous_lateral = 0.0
                previous_reach = 0.0
                previous_approach = 0.0
            self._signal[arm] = (
                timestamp_ms,
                max(lateral, previous_lateral * 0.82),
                max(reach_gain, previous_reach * 0.82),
                max(approach, previous_approach * 0.82),
            )
            if lateral < 0.04 and reach_gain < 0.03 and approach < 0.03:
                self._baseline_x[arm] = baseline_x * 0.97 + relative_x * 0.03
                self._baseline_reach[arm] = baseline_reach * 0.97 + reach * 0.03
                self._baseline_z[arm] = baseline_z * 0.97 + relative_z * 0.03

    def signals(self, arm: str | None, timestamp_ms: int) -> DepthSignals:
        arms = (arm,) if arm in {"left", "right"} else ("left", "right")
        best = DepthSignals(0.0, 0.0, 10_000)
        for selected in arms:
            observed_ms, lateral, reach, approach = self._signal[selected]
            if observed_ms < 0:
                continue
            age = timestamp_ms - observed_ms
            if age < best.age_ms:
                best = DepthSignals(lateral, reach, age, approach)
        return best

    def confirms(self, arm: str | None, timestamp_ms: int) -> bool:
        cue = self.signals(arm, timestamp_ms)
        return cue.age_ms <= 350 and max(cue.lateral, cue.reach, cue.approach) >= 0.08

    def suggests_straight(self, arm: str | None, timestamp_ms: int) -> bool:
        cue = self.signals(arm, timestamp_ms)
        return (
            cue.age_ms <= 350
            and cue.reach >= 0.10
            and cue.reach >= cue.lateral * 0.90
            and cue.reach >= cue.approach * 0.85
        )

    def suggests_hook(self, arm: str | None, timestamp_ms: int) -> bool:
        cue = self.signals(arm, timestamp_ms)
        if cue.age_ms > 350:
            return False
        lateral_hook = cue.lateral >= 0.08 and cue.lateral > cue.reach * 1.15
        depth_hook = cue.approach >= 0.08 and cue.approach > cue.reach * 1.10
        return lateral_hook or depth_hook


def _relabel_event(event: PunchEvent, label_id: int) -> PunchEvent:
    movement = MOVEMENTS[label_id]
    return replace(
        event,
        label_id=movement.label_id,
        display_name=movement.display_name,
        arm="left" if label_id in (1, 3) else "right",
        counter_key=None,
    )


class MultiViewMonitor:
    """Ejecuta pose en la auxiliar depth y apoya o filtra la clasificación frontal."""

    def __init__(self, config: AppConfig, *, depth_sensitivity: str = "high") -> None:
        self._config = config
        self._depth_sensitivity = depth_sensitivity
        self._trackers: dict[str, PoseTracker] = {}
        self._depth_classifier: HeuristicClassifier | None = None
        self._depth_result: ClassificationResult | None = None
        self._depth = _DepthMotionCue()
        self._previous_depth_center: tuple[float, float] | None = None
        self._previous_depth_scale: float | None = None
        self._stable_depth_frames = 0
        self._recent_by_arm: dict[str, tuple[int, str | None, int | None]] = {}
        self.poses: dict[str, PoseObservation] = {}
        self.quality: tuple[ViewQuality, ...] = ()
        self.preview_bgr: NDArray[np.uint8] | None = None
        self.readiness = DepthReadiness()
        self.status_line = ""
        self.role_line = ""

    def set_sensitivity(self, name: str) -> None:
        self._depth_sensitivity = name
        if self._depth_classifier is not None:
            self._depth_classifier.set_sensitivity(name)

    def open(self, rig: CameraRig) -> None:
        self.role_line = "  ".join(
            f"{role}={index}" for role, index in rig.role_indices.items()
        )
        for role in rig.role_indices:
            if role != "depth":
                continue
            self._trackers[role] = PoseTracker(
                model_path=self._config.model_path,
                min_detection_confidence=self._config.min_pose_detection_confidence,
                min_presence_confidence=self._config.min_pose_presence_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            self._depth_classifier = HeuristicClassifier(
                replace(
                    profile_settings(self._depth_sensitivity),
                    arm_only=self._config.arm_only,
                )
            )

    def observe(self, frame: RigFrame, rig: CameraRig) -> dict[str, PoseObservation]:
        self.poses = {}
        rows: list[ViewQuality] = []
        self._depth_result = None
        depth_packet = frame.by_role.get("depth")
        self.preview_bgr = None if depth_packet is None else np.array(depth_packet.frame, copy=True)
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
                if self._depth_classifier is not None:
                    features = extract_biomechanical_features(
                        pose,
                        min_visibility=self._config.min_landmark_visibility,
                        min_presence=self._config.min_landmark_presence,
                        image_size=(packet.width, packet.height),
                    )
                    self._depth_result = self._depth_classifier.update(
                        features,
                        pose_detected=pose.detected,
                        pose=pose,
                        frame=packet.frame,
                    )
        self.quality = tuple(rows)
        depth_pose = self.poses.get("depth")
        self.readiness = self._depth_readiness(depth_packet, depth_pose)
        self.status_line = self.summary()
        return dict(self.poses)

    def _depth_readiness(
        self,
        packet: object,
        pose: PoseObservation | None,
    ) -> DepthReadiness:
        """Evalúa la cámara depth para orientar el encuadre en tiempo real."""

        frame = getattr(packet, "frame", None)
        camera_ok = (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.size > 0
            and frame.shape[2] >= 3
        )
        if not camera_ok or pose is None:
            self._stable_depth_frames = 0
            self._previous_depth_center = None
            self._previous_depth_scale = None
            return DepthReadiness(
                camera_ok=camera_ok,
                status="DEPTH SIN SEÑAL" if not camera_ok else "DEPTH BUSCANDO OBJETIVO",
            )

        sharpness = _focus_measure(frame)
        focus_score = _focus_score(sharpness)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        brightness_ok = 18.0 <= mean_brightness <= 238.0
        arm_quality = _arm_quality(pose)
        self._update_depth_stability(pose)

        hold_ready = bool(
            (
                self.readiness.ready
                or self._stable_depth_frames >= DEPTH_REQUIRED_STABLE_FRAMES
            )
            and pose.detected
            and brightness_ok
            and sharpness >= DEPTH_HOLD_SHARPNESS
            and arm_quality >= DEPTH_HOLD_ARM_QUALITY
        )
        if hold_ready:
            status = f"DEPTH LISTA  FOCO {focus_score:.0%}"
            ready = True
        elif not pose.detected:
            status = "DEPTH BUSCANDO OBJETIVO"
            ready = False
        elif sharpness < DEPTH_MIN_SHARPNESS:
            status = f"DEPTH AJUSTA ENFOQUE {focus_score:.0%}"
            ready = False
        elif not brightness_ok:
            status = "DEPTH AJUSTA LUZ"
            ready = False
        elif arm_quality < DEPTH_MIN_ARM_QUALITY:
            status = f"DEPTH MUESTRA BRAZOS {arm_quality:.0%}"
            ready = False
        elif self._stable_depth_frames < DEPTH_REQUIRED_STABLE_FRAMES:
            status = "DEPTH ESTABILIZANDO OBJETIVO"
            ready = False
        else:
            status = f"DEPTH LISTA  FOCO {focus_score:.0%}"
            ready = True
        return DepthReadiness(
            camera_ok=True,
            detected=pose.detected,
            focus_score=focus_score,
            brightness_ok=brightness_ok,
            arm_quality=arm_quality,
            stable_frames=self._stable_depth_frames,
            ready=ready,
            status=status,
        )

    def _update_depth_stability(self, pose: PoseObservation) -> bool:
        """Cuenta fotogramas en los que el objetivo no salta de encuadre."""

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
            self._stable_depth_frames = 0
            self._previous_depth_center = None
            self._previous_depth_scale = None
            return False

        center = ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0)
        scale = max(float(np.hypot(right.x - left.x, right.y - left.y)), 0.05)
        stable = True
        if self._previous_depth_center is not None and self._previous_depth_scale is not None:
            movement = float(np.hypot(
                center[0] - self._previous_depth_center[0],
                center[1] - self._previous_depth_center[1],
            ))
            scale_change = abs(scale - self._previous_depth_scale) / self._previous_depth_scale
            stable = movement / scale <= 0.08 and scale_change <= 0.12
        if stable:
            self._stable_depth_frames += 1
        elif (
            self.readiness.ready
            or self._stable_depth_frames >= DEPTH_REQUIRED_STABLE_FRAMES
        ):
            # El golpe mueve hombros: no se pierde LISTA si el objetivo sigue visible.
            pass
        else:
            self._stable_depth_frames = 1
        self._previous_depth_center = center
        self._previous_depth_scale = scale
        return stable

    def filter_result(self, result: ClassificationResult, timestamp_ms: int) -> ClassificationResult:
        if self._config.depth_gate == "off":
            return result

        fused = self._fuse_depth_event(result, timestamp_ms)
        assisted = self._assist(fused, timestamp_ms)
        if self._config.depth_gate != "strict" or not assisted.events:
            return assisted
        if self._depth_result is None and (
            not self._trackers or "depth" not in self._trackers
        ):
            return replace(
                assisted,
                events=(),
                status="incierto",
                display_name="MOVIMIENTO NO CONFIRMADO",
                rejection="no hay cámara depth disponible",
                diagnostics={**assisted.diagnostics, "depth_gate": "missing"},
            )
        depth_confirmed_arms = {
            event.arm for event in (self._depth_result.events if self._depth_result else ())
        }
        unconfirmed = tuple(
            event
            for event in assisted.events
            if event.source != "depth"
            and event.arm not in depth_confirmed_arms
            and not self._depth.confirms(event.arm, timestamp_ms)
        )
        if not unconfirmed:
            return replace(
                assisted,
                diagnostics={**assisted.diagnostics, "depth_gate": "confirmed"},
            )
        return replace(
            assisted,
            events=(),
            status="incierto",
            label_id=None,
            display_name="MOVIMIENTO NO CONFIRMADO",
            rejection="sin confirmación de la cámara depth",
            diagnostics={
                **assisted.diagnostics,
                "depth_gate": "rejected",
                "depth_unconfirmed_arms": tuple(event.arm for event in unconfirmed),
            },
        )

    def _fuse_depth_event(
        self,
        result: ClassificationResult,
        timestamp_ms: int,
    ) -> ClassificationResult:
        """Une frontal y depth: una clase por brazo, sin duplicar entre fotogramas."""

        depth_result = self._depth_result
        depth_events = tuple(
            replace(
                event,
                source="depth",
                event_id=f"depth:{event.event_id or timestamp_ms}:{event.arm}",
            )
            for event in (depth_result.events if depth_result is not None else ())
        )
        by_arm: dict[str, dict[str, PunchEvent]] = {}
        unlabeled: list[PunchEvent] = []
        for event in result.events:
            if event.arm in {"left", "right"}:
                by_arm.setdefault(event.arm, {})["front"] = event
            else:
                unlabeled.append(event)
        for event in depth_events:
            if event.arm in {"left", "right"}:
                by_arm.setdefault(event.arm, {})["depth"] = event

        side_conflict = False
        front_candidates = [event for event in result.events if event.arm in {"left", "right"}]
        depth_candidates = [event for event in depth_events if event.arm in {"left", "right"}]
        if len(front_candidates) == 1 and len(depth_candidates) == 1:
            front_event = front_candidates[0]
            depth_event = depth_candidates[0]
            front_time = front_event.ended_ms if front_event.ended_ms is not None else timestamp_ms
            depth_time = depth_event.ended_ms if depth_event.ended_ms is not None else timestamp_ms
            side_conflict = (
                front_event.arm != depth_event.arm
                and abs(front_time - depth_time) <= DEPTH_EVENT_DEDUPE_MS
            )
            if side_conflict:
                by_arm[depth_event.arm].pop("depth", None)

        merged = list(unlabeled)
        reasons: list[str] = []
        added = 0
        for arm, pair in by_arm.items():
            chosen, reason = self._choose_event(
                pair.get("front"),
                pair.get("depth"),
                timestamp_ms,
            )
            if chosen is None:
                if reason == "depth_unready":
                    reasons.append(reason)
                continue
            if self._is_recent_duplicate(arm, chosen, timestamp_ms):
                reasons.append("suppressed_duplicate")
                continue
            merged.append(chosen)
            if side_conflict and reason == "front_only":
                reason = "front_side_conflict"
            reasons.append(reason)
            if reason == "added":
                added += 1
            ended = chosen.ended_ms if chosen.ended_ms is not None else timestamp_ms
            self._recent_by_arm[arm] = (ended, chosen.event_id, chosen.label_id)

        fusion = "+".join(dict.fromkeys(reasons)) if reasons else "none"
        diagnostics = {
            **result.diagnostics,
            "depth_fusion": fusion,
            "depth_events": added,
        }
        if not merged:
            return replace(result, events=(), diagnostics=diagnostics)
        if merged == list(result.events) and fusion in {
            "none",
            "front_only",
            "same_event",
            "front_class_kept",
            "front_side_conflict",
        }:
            return replace(result, diagnostics=diagnostics)

        latest = merged[-1]
        debug = result.debug_lines
        if added:
            debug = debug + tuple(
                f"depth detecto {event.display_name}" for event in depth_events
            )
        if "front_class_kept" in fusion:
            debug = debug + ("frontal conservo jab/gancho",)
        candidate = result.candidate
        if depth_result is not None and added:
            candidate = depth_result.candidate or candidate
        return replace(
            result,
            status=STATUS_COMPLETED,
            label_id=latest.label_id,
            display_name=latest.display_name,
            events=tuple(merged),
            active_arm=latest.arm,
            candidate=candidate,
            rejection=None,
            diagnostics=diagnostics,
            debug_lines=debug,
        )

    def _choose_event(
        self,
        front: PunchEvent | None,
        depth: PunchEvent | None,
        timestamp_ms: int,
    ) -> tuple[PunchEvent | None, str]:
        if front is None and depth is None:
            return None, "none"
        if front is None:
            # La Steren sola inventaba ganchos con pose lateral ruidosa. Solo
            # se acepta un golpe exclusivo de depth si ya está lista o el cue
            # de movimiento confirma el brazo.
            if depth is None:
                return None, "none"
            if self.readiness.ready or self._depth.confirms(depth.arm, timestamp_ms):
                return depth, "added"
            return None, "depth_unready"
        if depth is None:
            return front, "front_only"
        # La vista frontal sigue siendo la autoridad anatómica. Una cámara
        # lateral/RGB puede invertir la lectura izquierda-derecha por su
        # encuadre o espejado; no permitimos que esa contradicción cambie el
        # lado del evento ya detectado en la frontal.
        if front.arm in {"left", "right"} and depth.arm in {"left", "right"} and front.arm != depth.arm:
            return front, "front_side_conflict"
        if front.label_id == depth.label_id or depth.label_id is None:
            return front, "same_event"
        # Tampoco dejamos que el clasificador depth pise jab<->gancho: en la
        # última sesión convertía rectos largos en ganchos. El cue de movimiento
        # (_assist) sí puede corregir la clase cuando la geometría lateral lo
        # respalda.
        return front, "front_class_kept"

    def _is_recent_duplicate(
        self,
        arm: str,
        event: PunchEvent,
        timestamp_ms: int,
    ) -> bool:
        recent = self._recent_by_arm.get(arm)
        if recent is None:
            return False
        recent_ms, recent_id, _label = recent
        if event.event_id is not None and event.event_id == recent_id:
            return True
        event_time = event.ended_ms if event.ended_ms is not None else timestamp_ms
        return 0 <= event_time - recent_ms <= DEPTH_EVENT_DEDUPE_MS

    @staticmethod
    def _same_physical_event(first: PunchEvent, second: PunchEvent) -> bool:
        """Compara eventos por brazo y cercanía temporal, no por etiqueta."""

        if first.arm is None or second.arm is None or first.arm != second.arm:
            return False
        if first.event_id is not None and first.event_id == second.event_id:
            return True
        first_time = first.ended_ms if first.ended_ms is not None else first.started_ms
        second_time = second.ended_ms if second.ended_ms is not None else second.started_ms
        if first_time is not None and second_time is not None:
            return abs(first_time - second_time) <= DEPTH_EVENT_DEDUPE_MS
        return first.label_id == second.label_id

    def _assist(self, result: ClassificationResult, timestamp_ms: int) -> ClassificationResult:
        cue = self._depth.signals(result.active_arm, timestamp_ms)
        diagnostics = {
            **result.diagnostics,
            "depth_gate": self._config.depth_gate,
            "depth_lateral": round(cue.lateral, 3),
            "depth_reach": round(cue.reach, 3),
            "depth_approach": round(cue.approach, 3),
            "depth_age_ms": cue.age_ms,
            "depth_status": self.status_line or "sin depth",
        }
        debug = result.debug_lines + (
            f"depth lat {cue.lateral:.2f} alc {cue.reach:.2f} z {cue.approach:.2f}",
            self.status_line or "depth inactiva",
        )
        if not result.events:
            return replace(result, diagnostics=diagnostics, debug_lines=debug)

        # En el modo por brazo la cámara auxiliar solo aporta respaldo de
        # movimiento. No debe volver a inventar una clase jab/gancho.
        if self._config.arm_only:
            return replace(result, diagnostics=diagnostics, debug_lines=debug)

        rewritten: list[PunchEvent] = []
        changed = False
        for event in result.events:
            label = event.label_id
            if label is None:
                rewritten.append(event)
                continue
            arm = event.arm
            if label in (1, 2) and self._depth.suggests_hook(arm, timestamp_ms) and not self._depth.suggests_straight(arm, timestamp_ms):
                new_label = 3 if arm == "left" or label == 1 else 4
                rewritten.append(_relabel_event(event, new_label))
                changed = True
                continue
            if label in (3, 4) and self._depth.suggests_straight(arm, timestamp_ms) and not self._depth.suggests_hook(arm, timestamp_ms):
                new_label = 1 if arm == "left" or label == 3 else 2
                rewritten.append(_relabel_event(event, new_label))
                changed = True
                continue
            rewritten.append(event)

        if not changed:
            return replace(result, diagnostics=diagnostics, debug_lines=debug)
        latest = rewritten[-1]
        return replace(
            result,
            events=tuple(rewritten),
            label_id=latest.label_id,
            display_name=latest.display_name,
            active_arm=latest.arm,
            diagnostics={**diagnostics, "depth_assist": "relabeled"},
            debug_lines=debug + ("depth corrigio clase",),
        )

    def summary(self) -> str:
        if not self.quality:
            if self._config.camera_setup == "depth":
                return "DEPTH sin camara"
            return ""
        summary = "  ".join(
            f"{row.role}{' OK' if row.detected else ' --'} q={row.arm_quality:.2f}"
            for row in self.quality
        )
        if any(row.role == "depth" for row in self.quality):
            return f"{summary}  |  {self.readiness.status}"
        return summary

    def close(self) -> None:
        for tracker in self._trackers.values():
            tracker.close()
        self._trackers.clear()
        self._depth_classifier = None
        self._depth_result = None
        self.poses.clear()
        self.quality = ()
        self.preview_bgr = None
        self.readiness = DepthReadiness()
        self._previous_depth_center = None
        self._previous_depth_scale = None
        self._stable_depth_frames = 0
        self._recent_by_arm.clear()
        self._depth = _DepthMotionCue()
        self.status_line = ""
        self.role_line = ""
