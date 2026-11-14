"""Grabador de sesiones numéricas para las Fases 2 y 3.

No conserva frames. Guarda landmarks, eventos, métricas y acumuladores en un
directorio temporal que se mueve atómicamente al finalizar.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from app.camera import RigFrame
from app.pose_tracker import LandmarkId, PoseObservation
from app.trainer.results import ClassificationResult, PunchEvent
from app.vision_lab.biomechanics import calculate_elbow_angle, calculate_kinematics
from app.vision_lab.heatmaps import (
    HeatmapAccumulator,
    add_heatmap_legend,
    accumulate_motion,
    enhance_reference_frame,
    heatmap_to_bgr,
    overlay_heatmap_on_frame,
)
from app.vision_lab.trajectories import WristPoint


@dataclass(frozen=True, slots=True)
class SessionReport:
    session_id: str
    directory: Path
    summary: dict[str, Any]
    exports: dict[str, str]


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]


def _json_float(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


class SessionRecorder:
    """Captura trayectorias de la frontal y actividad de la Steren."""

    def __init__(
        self,
        results_dir: Path,
        *,
        classifier: str,
        session_kind: str = "live",
        smoothing_alpha: float = 0.35,
    ) -> None:
        self.results_dir = Path(results_dir).expanduser().resolve()
        self.classifier = classifier
        self.session_kind = session_kind
        self.smoothing_alpha = smoothing_alpha
        self._metadata: dict[str, Any] = {}
        self.reset()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def has_observations(self) -> bool:
        return self._frame_count > 0 or bool(self._events)

    def reset(self) -> None:
        self._session_id = _now_id()
        self._frame_count = 0
        self._tracking_frames = 0
        self._valid_wrist_samples = 0
        self._active_duration_s = 0.0
        self._movement_duration_s = 0.0
        self._first_timestamp_ns: int | None = None
        self._last_active_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None
        self._events: list[dict[str, Any]] = []
        self._event_ids: set[str] = set()
        self._trajectory_points: dict[str, list[WristPoint]] = {"left": [], "right": []}
        self._trajectory_rows: list[dict[str, Any]] = []
        self._elbow_samples: dict[str, list[tuple[int, float]]] = {"left": [], "right": []}
        self._left_heatmap: HeatmapAccumulator | None = None
        self._right_heatmap: HeatmapAccumulator | None = None
        self._external_heatmap: np.ndarray | None = None
        self._main_shape: tuple[int, int] | None = None
        self._external_frames = 0
        self._external_sync_valid = 0
        self._external_confidence: list[float] = []
        self._primary_reference_frame: np.ndarray | None = None
        self._external_reference_frame: np.ndarray | None = None
        self._metadata = {
            "classifier": self.classifier,
            "session_kind": self.session_kind,
            "contains_raw_video": False,
            "contains_reference_screenshots": True,
            "reference_screenshots_enhanced": True,
            "heatmap_visualization": "captura de webcam con actividad superpuesta y escala adaptativa",
            "external_camera_is_rgb_only": True,
            "three_dimensional_metric_data": False,
            "coordinate_space": "camera_native_normalized",
            "frame_contract": {
                "frame_id": "camera-local integer when available",
                "captured_at_monotonic_ns": True,
                "camera_id": "source camera identifier",
                "rotation_deg": 0,
                "display_mirrored": "render-only flag; anatomical labels are unchanged",
                "valid": "tracking_valid per landmark",
            },
            "heatmap_policy": "only intentional motion or validated punch events; guard presence stays in CSV",
            "units": {
                "trajectory": "px",
                "velocity": "px/s",
                "acceleration": "px/s²",
                "angle": "degrees",
                "normalized_reach": "shoulder_widths",
            },
        }

    def update_metadata(self, **values: Any) -> None:
        self._metadata.update(values)

    def record_frame(
        self,
        *,
        timestamp_ns: int,
        frame_id: int | None = None,
        pose: PoseObservation,
        result: ClassificationResult,
        frame_size: tuple[int, int],
        primary_frame: np.ndarray | None = None,
        active: bool,
        rig_frame: RigFrame | None = None,
        external_motion_mask: np.ndarray | None = None,
        external_motion_score: float = 0.0,
        external_segmentation_confidence: float = 0.0,
        external_sync_valid: bool = False,
        external_role: str | None = None,
    ) -> None:
        timestamp_ns = int(timestamp_ns or pose.timestamp_ms * 1_000_000)
        width, height = frame_size
        if self._first_timestamp_ns is None:
            self._first_timestamp_ns = timestamp_ns
        if active and self._primary_reference_frame is None and primary_frame is not None:
            self._primary_reference_frame = np.array(primary_frame, copy=True)
        if active and self._external_reference_frame is None and rig_frame is not None and external_role:
            external_packet = rig_frame.by_role.get(external_role)
            if external_packet is not None:
                self._external_reference_frame = np.array(external_packet.frame, copy=True)
        delta_s = 0.0
        if active and self._last_active_timestamp_ns is not None:
            delta_ns = timestamp_ns - self._last_active_timestamp_ns
            if 0 < delta_ns <= 5_000_000_000:
                delta_s = delta_ns / 1_000_000_000.0
                self._active_duration_s += delta_s
        self._last_timestamp_ns = timestamp_ns
        self._last_active_timestamp_ns = timestamp_ns if active else None
        self._frame_count += 1
        if pose.detected:
            self._tracking_frames += 1
        if external_motion_mask is not None:
            self._external_frames += 1
            if external_sync_valid:
                self._external_sync_valid += 1
            self._external_confidence.append(float(external_segmentation_confidence))
            if (
                active
                and external_sync_valid
                and delta_s > 0
                and np.count_nonzero(external_motion_mask)
            ):
                self._movement_duration_s += delta_s
            if self._external_heatmap is None and external_sync_valid:
                self._external_heatmap = np.zeros(external_motion_mask.shape, dtype=np.float32)
            if (
                active
                and external_sync_valid
                and self._external_heatmap.shape == external_motion_mask.shape
            ):
                accumulate_motion(self._external_heatmap, external_motion_mask, delta_s)

        if active:
            primary_camera_id = (
                "front"
                if rig_frame is None
                else rig_frame.primary.camera_id or "front"
            )
            self._record_landmarks(
                timestamp_ns,
                frame_id,
                pose,
                width,
                height,
                camera_id=primary_camera_id,
                active_hands=self._active_hands(result),
            )
            self._record_events(
                result.events,
                timestamp_ns,
                external_role,
                external_sync_valid,
                camera_id=(
                    "front"
                    if rig_frame is None
                    else rig_frame.primary.camera_id or "front"
                ),
            )

    @staticmethod
    def _active_hands(result: ClassificationResult) -> set[str]:
        hands = {
            event.arm
            for event in result.events
            if event.arm in {"left", "right"}
        }
        if result.moving and result.active_arm in {"left", "right"}:
            hands.add(result.active_arm)
        return hands

    def _record_landmarks(
        self,
        timestamp_ns: int,
        frame_id: int | None,
        pose: PoseObservation,
        width: int,
        height: int,
        *,
        camera_id: str,
        active_hands: set[str],
    ) -> None:
        if self._main_shape is None:
            self._main_shape = (height, width)
        if self._left_heatmap is None and self._main_shape is not None and pose.detected:
            self._left_heatmap = HeatmapAccumulator(self._main_shape)
            self._right_heatmap = HeatmapAccumulator(self._main_shape)
        shoulders = (
            pose.landmark(LandmarkId.LEFT_SHOULDER),
            pose.landmark(LandmarkId.RIGHT_SHOULDER),
        )
        shoulder_scale_px: float | None = None
        if all(point is not None and point.is_finite for point in shoulders):
            assert shoulders[0] is not None and shoulders[1] is not None
            shoulder_scale_px = float(np.hypot(
                (shoulders[1].x - shoulders[0].x) * width,
                (shoulders[1].y - shoulders[0].y) * height,
            ))
            if shoulder_scale_px <= 1e-6:
                shoulder_scale_px = None
        for hand, shoulder_id, elbow_id, wrist_id, heatmap in (
            ("left", LandmarkId.LEFT_SHOULDER, LandmarkId.LEFT_ELBOW, LandmarkId.LEFT_WRIST, self._left_heatmap),
            ("right", LandmarkId.RIGHT_SHOULDER, LandmarkId.RIGHT_ELBOW, LandmarkId.RIGHT_WRIST, self._right_heatmap),
        ):
            shoulder = pose.landmark(shoulder_id)
            elbow = pose.landmark(elbow_id)
            wrist = pose.landmark(wrist_id)
            valid = bool(
                wrist is not None
                and wrist.is_finite
                and 0.0 <= wrist.x <= 1.0
                and 0.0 <= wrist.y <= 1.0
                and wrist.visibility >= 0.32
                and wrist.presence >= 0.40
            )
            x_px = None if not valid or wrist is None else float(wrist.x * width)
            y_px = None if not valid or wrist is None else float(wrist.y * height)
            point = WristPoint(
                timestamp_ns=timestamp_ns,
                x=float("nan") if x_px is None else x_px,
                y=float("nan") if y_px is None else y_px,
                visibility=0.0 if wrist is None else float(wrist.visibility),
                hand=hand,
                camera_id=camera_id,
                tracking_valid=valid,
            )
            self._trajectory_points[hand].append(point)
            if valid:
                self._valid_wrist_samples += 1
                if heatmap is not None and wrist is not None and hand in active_hands:
                    # Sólo se deposita movimiento intencional/eventos; la
                    # guardia quieta queda disponible en CSV, no contamina el
                    # mapa de actividad.
                    radius = max(3, int(round(width * 0.03)))
                    heatmap.add_point(wrist.x, wrist.y, weight=1.0, radius=radius)
                if shoulder is not None and elbow is not None and wrist is not None:
                    angle = calculate_elbow_angle(
                        (shoulder.x * width, shoulder.y * height),
                        (elbow.x * width, elbow.y * height),
                        (wrist.x * width, wrist.y * height),
                    )
                    if angle is not None:
                        self._elbow_samples[hand].append((timestamp_ns, angle))
            self._trajectory_rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "timestamp_ms": timestamp_ns // 1_000_000,
                    "hand": hand,
                    "camera_id": camera_id,
                    "source_width": width,
                    "source_height": height,
                    "rotation_deg": 0,
                    "mirrored_for_display": bool(self._metadata.get("mirror_view", False)),
                    "presence": None if wrist is None else float(wrist.presence),
                    "x_px": x_px,
                    "y_px": y_px,
                    "x_normalized": None if not valid or wrist is None else float(wrist.x),
                    "y_normalized": None if not valid or wrist is None else float(wrist.y),
                    "visibility": None if wrist is None else float(wrist.visibility),
                    "tracking_valid": valid,
                    "shoulder_width_px": shoulder_scale_px,
                }
            )

    def _record_events(
        self,
        events: tuple[PunchEvent, ...],
        timestamp_ns: int,
        external_role: str | None,
        external_sync_valid: bool,
        camera_id: str,
    ) -> None:
        for index, event in enumerate(events):
            event_id = event.event_id or f"{timestamp_ns}:{event.label_id}:{event.arm}:{index}"
            if event_id in self._event_ids:
                continue
            self._event_ids.add(event_id)
            self._events.append(
                {
                    "event_id": event_id,
                    "label_id": event.label_id,
                    "type": event.display_name,
                    "hand": event.arm,
                    "started_ms": event.started_ms,
                    "ended_ms": event.ended_ms or timestamp_ns // 1_000_000,
                    "confidence": 1.0 if event.confirmed else 0.0,
                    "tracking_valid": True,
                    "sync_valid": None if external_role is None else bool(external_sync_valid),
                    "source_camera": camera_id,
                    "external_camera_role": external_role,
                }
            )

    def _event_metrics(self, event: dict[str, Any]) -> dict[str, Any]:
        hand = event.get("hand")
        points = self._trajectory_points.get(hand, []) if hand in self._trajectory_points else []
        start_ns = int(event["started_ms"] or 0) * 1_000_000
        end_ns = int(event["ended_ms"] or 0) * 1_000_000
        selected = [point for point in points if start_ns <= point.timestamp_ns <= end_ns]
        if len(selected) < 2:
            selected = points
        kinematics = calculate_kinematics(selected, smoothing_alpha=self.smoothing_alpha) if len(selected) >= 2 else None
        elbow = self._elbow_samples.get(hand, []) if hand in self._elbow_samples else []
        elbow_values = [value for timestamp, value in elbow if start_ns <= timestamp <= end_ns]
        if not elbow_values:
            elbow_values = [value for _, value in elbow]
        shoulder_scales = [
            float(row["shoulder_width_px"])
            for row in self._trajectory_rows
            if row["hand"] == hand
            and start_ns <= int(row["timestamp_ns"]) <= end_ns
            and row["shoulder_width_px"]
        ]
        scale = float(np.median(shoulder_scales)) if shoulder_scales else None
        return {
            "duration_ms": max(0, int(event["ended_ms"] or 0) - int(event["started_ms"] or 0)),
            "reach_normalized": None if kinematics is None or scale is None else _json_float(kinematics.displacement_px / scale),
            "path_length_px": None if kinematics is None else _json_float(kinematics.path_length_px),
            "peak_speed_px_s": None if kinematics is None else _json_float(kinematics.peak_speed_px_s),
            "peak_speed_normalized": None if kinematics is None or scale is None else _json_float(kinematics.peak_speed_px_s / scale),
            "peak_acceleration_px_s2": None if kinematics is None else _json_float(kinematics.peak_acceleration_px_s2),
            "peak_acceleration_normalized": None if kinematics is None or scale is None else _json_float(kinematics.peak_acceleration_px_s2 / scale),
            "elbow_min_deg": None if not elbow_values else min(elbow_values),
            "elbow_max_deg": None if not elbow_values else max(elbow_values),
            "direction_change_deg": None if kinematics is None else kinematics.direction_change_deg,
            "tracking_valid": bool(selected) and all(point.tracking_valid for point in selected),
            "data_status": "ok" if kinematics is not None else "Datos insuficientes",
        }

    def finalize(self, *, extra_metadata: dict[str, Any] | None = None) -> SessionReport:
        parent = self.results_dir / "sessions"
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / self._session_id
        temporary = Path(tempfile.mkdtemp(prefix=f".{self._session_id}-", dir=str(parent)))
        try:
            metrics = []
            for event in self._events:
                enriched = dict(event)
                enriched.update(self._event_metrics(event))
                metrics.append(enriched)
            by_type: dict[str, int] = {}
            by_hand: dict[str, int] = {}
            for event in self._events:
                by_type[event["type"]] = by_type.get(event["type"], 0) + 1
                hand = event.get("hand") or "unknown"
                by_hand[hand] = by_hand.get(hand, 0) + 1
            tracking_quality = self._tracking_frames / self._frame_count if self._frame_count else 0.0
            wrist_quality = self._valid_wrist_samples / max(1, self._frame_count * 2)
            sync_quality = self._external_sync_valid / self._external_frames if self._external_frames else None
            summary: dict[str, Any] = {
                "session_id": self._session_id,
                "session_kind": self.session_kind,
                "duration_active_s": round(self._active_duration_s, 4),
                "movement_time_s": round(self._movement_duration_s, 4),
                "total_punches": len(self._events),
                "by_type": by_type,
                "by_hand": by_hand,
                "tracking_quality": round(tracking_quality, 4),
                "wrist_tracking_quality": round(wrist_quality, 4),
                "external_frames": self._external_frames,
                "external_sync_quality": None if sync_quality is None else round(sync_quality, 4),
                "external_segmentation_confidence": None if not self._external_confidence else round(float(np.mean(self._external_confidence)), 4),
                "external_activity_available": self._external_heatmap is not None,
                "three_dimensional_metric_data": False,
                "status": "Datos insuficientes" if not self._events and not self._trajectory_rows else "ok",
            }
            metadata = {**self._metadata, "created_at_utc": datetime.now(timezone.utc).isoformat(), "summary": summary}
            if extra_metadata:
                metadata.update(extra_metadata)
            self._write_json(temporary / "metadata.json", metadata)
            self._write_csv(temporary / "punch_events.csv", metrics)
            self._write_csv(temporary / "wrist_trajectories.csv", self._trajectory_rows)
            self._write_json(temporary / "biomechanics.json", {"events": metrics, "hands": self._hand_metrics()})
            exports_dir = temporary / "exports"
            exports_dir.mkdir()
            exports: dict[str, str] = {}
            if self._external_heatmap is not None:
                np.save(temporary / "heatmap_external_activity.npy", self._external_heatmap)
                path = exports_dir / "heatmap_activity.png"
                activity = self._external_heatmap
                if self._external_reference_frame is not None:
                    activity_image = overlay_heatmap_on_frame(
                        enhance_reference_frame(self._external_reference_frame),
                        activity,
                        mirror=bool(self._metadata.get("mirror_view", False)),
                    )
                else:
                    activity_image = heatmap_to_bgr(activity)
                activity_image = add_heatmap_legend(
                    activity_image,
                    maximum=float(np.max(activity)),
                    label="actividad Steren (s)",
                )
                cv2.imwrite(str(path), activity_image)
                exports["activity"] = str(path)
            if self._left_heatmap is not None and self._right_heatmap is not None:
                left = self._left_heatmap.values
                right = self._right_heatmap.values
                assert left is not None and right is not None
                shared_maximum = max(float(np.max(left)), float(np.max(right)))
                np.save(temporary / "heatmap_left.npy", left)
                np.save(temporary / "heatmap_right.npy", right)
                for name, values in (("left", left), ("right", right)):
                    path = exports_dir / f"heatmap_{name}.png"
                    if self._primary_reference_frame is not None:
                        image = overlay_heatmap_on_frame(
                            enhance_reference_frame(self._primary_reference_frame),
                            values,
                            maximum=shared_maximum,
                            mirror=bool(self._metadata.get("mirror_view", False)),
                        )
                    else:
                        image = heatmap_to_bgr(values, maximum=shared_maximum)
                    image = add_heatmap_legend(
                        image,
                        maximum=shared_maximum,
                        label=("muñeca izquierda" if name == "left" else "muñeca derecha"),
                    )
                    cv2.imwrite(str(path), image)
                    exports[name] = str(path)
                if self._primary_reference_frame is not None:
                    reference_path = exports_dir / "reference_macbook.png"
                    reference = enhance_reference_frame(self._primary_reference_frame)
                    reference = np.fliplr(reference) if self._metadata.get("mirror_view", False) else reference
                    cv2.imwrite(str(reference_path), reference)
                    exports["reference_macbook"] = str(reference_path)
            if self._external_reference_frame is not None:
                reference_path = exports_dir / "reference_steren.png"
                reference = enhance_reference_frame(self._external_reference_frame)
                reference = np.fliplr(reference) if self._metadata.get("mirror_view", False) else reference
                cv2.imwrite(str(reference_path), reference)
                exports["reference_steren"] = str(reference_path)
            trajectory_path = exports_dir / "trajectory_arms.png"
            cv2.imwrite(str(trajectory_path), self._trajectory_image())
            exports["trajectories"] = str(trajectory_path)
            summary["exports"] = exports
            self._write_json(exports_dir / "session_summary.json", summary)
            os.replace(str(temporary), str(target))
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return SessionReport(self._session_id, target, summary, {key: str(target / Path(value).relative_to(temporary)) for key, value in exports.items()})

    def _hand_metrics(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for hand, points in self._trajectory_points.items():
            if len(points) < 2:
                output[hand] = {"status": "Datos insuficientes"}
                continue
            kinematics = calculate_kinematics(points, smoothing_alpha=self.smoothing_alpha)
            output[hand] = {
                "status": "ok",
                "displacement_px": _json_float(kinematics.displacement_px),
                "path_length_px": _json_float(kinematics.path_length_px),
                "peak_speed_px_s": _json_float(kinematics.peak_speed_px_s),
                "peak_acceleration_px_s2": _json_float(kinematics.peak_acceleration_px_s2),
                "direction_change_deg": kinematics.direction_change_deg,
            }
        return output

    def _trajectory_image(self) -> np.ndarray:
        image = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (479, 359), (70, 75, 85), 1)
        cv2.putText(image, "Trayectorias de muñeca (imagen 2D)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 250), 1, cv2.LINE_AA)
        colors = {"left": (220, 210, 50), "right": (50, 150, 245)}
        for hand, color in colors.items():
            points = [
                (int(float(row["x_normalized"]) * 478), int(float(row["y_normalized"]) * 338 + 20))
                for row in self._trajectory_rows
                if row["hand"] == hand
                and row["tracking_valid"]
                and row["x_normalized"] is not None
                and row["y_normalized"] is not None
            ]
            if len(points) >= 2:
                cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            if points:
                cv2.circle(image, points[0], 5, color, -1, cv2.LINE_AA)
                cv2.circle(image, points[-1], 5, (245, 248, 252), -1, cv2.LINE_AA)
            cv2.putText(image, "Izquierda" if hand == "left" else "Derecha", (20 if hand == "left" else 360, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return image

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
