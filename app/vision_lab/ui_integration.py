"""Adaptador de la Fase 1 para observar la Steren sin tocar el clasificador."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from app.camera import RigFrame
from app.vision_lab.background_model import BackgroundModel
from app.vision_lab.motion_detector import MotionDetector, MotionResult
from app.vision_lab.morphology import dilate
from app.vision_lab.segmentation import (
    ForegroundSegmenter,
    SegmentationResult,
    segment_foreground_result,
)
from app.vision_lab.synchronization import SyncStatus, nearest_timestamp


VISION_MODES = ("RGB", "FONDO", "SILUETA", "MOVIMIENTO", "VALIDO", "SUPERPOSICION")


def _resize_for_processing(frame: NDArray[np.uint8], width: int) -> NDArray[np.uint8]:
    """Reduce resolución con área para evitar aliasing de la máscara."""

    height, source_width = frame.shape[:2]
    if source_width <= width:
        return np.array(frame, copy=True)
    target_height = max(1, int(round(height * width / source_width)))
    return cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True, slots=True)
class VisionLabTelemetry:
    external_role: str | None = None
    calibrated: bool = False
    calibration_progress: float = 0.0
    external_motion_score: float = 0.0
    segmentation_confidence: float = 0.0
    sync_valid: bool = False
    sync_delta_ms: float | None = None
    frame_valid: bool = False
    frame_id: int | None = None
    camera_id: str | None = None
    source_width: int | None = None
    source_height: int | None = None
    rotation_deg: int = 0
    display_mirrored: bool = False
    status: str = "STEREN SIN CÁMARA"


class ExternalVisionLab:
    """Procesa la webcam RGB externa a baja resolución y sólo devuelve telemetría."""

    def __init__(
        self,
        *,
        process_width: int = 320,
        calibration_frames: int = 36,
        threshold: float = 25.0,
        motion_threshold: float = 18.0,
        calibration_tolerance_ms: float = 100.0,
    ) -> None:
        if process_width < 32:
            raise ValueError("process_width debe ser al menos 32.")
        if calibration_frames < 20 or calibration_frames > 40:
            raise ValueError("calibration_frames debe estar entre 20 y 40.")
        self.process_width = process_width
        self.calibration_target = calibration_frames
        self.calibration_tolerance_ms = calibration_tolerance_ms
        # La escena de calibración se conserva durante la sesión. Si en el
        # futuro se activa adaptación, sólo debe hacerse con píxeles confiables
        # explícitos; nunca absorber al usuario por defecto.
        self.background = BackgroundModel(freeze_background=True)
        self.segmenter = ForegroundSegmenter(threshold=threshold)
        self.motion = MotionDetector(threshold=motion_threshold)
        self.mode = "RGB"
        self.roi: tuple[float, float, float, float] | None = None
        self.preview_bgr: NDArray[np.uint8] | None = None
        self.telemetry = VisionLabTelemetry()
        self._calibrating = False
        self._calibration_buffer: list[NDArray[np.uint8]] = []
        self._last_segmentation: SegmentationResult | None = None
        self._last_motion: MotionResult | None = None

    @property
    def status_line(self) -> str:
        return self.telemetry.status

    @property
    def motion_mask(self) -> NDArray[np.bool_] | None:
        return None if self._last_motion is None else self._last_motion.motion_mask

    @property
    def foreground_mask(self) -> NDArray[np.bool_] | None:
        return None if self._last_segmentation is None else self._last_segmentation.foreground_mask

    @property
    def activity_mask(self) -> NDArray[np.bool_] | None:
        """Movimiento que además pertenece a la silueta detectada.

        La máscara cruda sigue disponible para la vista MOVIMIENTO. Para los
        mapas se excluyen cambios de luz y fondo; se dilata la silueta unos
        píxeles para no perder los bordes de los guantes.
        """

        motion = self.motion_mask
        foreground = self.foreground_mask
        if motion is None or foreground is None:
            return None
        if self._last_segmentation is not None and self._last_segmentation.confidence <= 0.05:
            return np.zeros_like(motion)
        if self._last_motion is not None and self._last_motion.global_change:
            return np.zeros_like(motion)
        return motion & dilate(foreground, iterations=3)

    def request_recalibration(self) -> None:
        """Inicia una nueva mediana; la siguiente escena debe estar sin persona."""

        self.background.reset()
        self.segmenter.reset_tracking()
        self.motion.reset()
        self._calibration_buffer = []
        self._calibrating = True
        self.preview_bgr = None
        self._last_segmentation = None
        self._last_motion = None
        self.telemetry = VisionLabTelemetry(
            status=f"STEREN CALIBRANDO 0/{self.calibration_target}"
        )

    def set_threshold(self, threshold: float) -> None:
        self.segmenter.threshold = float(np.clip(threshold, 1.0, 120.0))

    def set_roi(self, roi: tuple[float, float, float, float] | None) -> None:
        if roi is not None:
            x0, y0, x1, y1 = roi
            if not 0.0 <= x0 < x1 <= 1.0 or not 0.0 <= y0 < y1 <= 1.0:
                raise ValueError("ROI inválida.")
        self.roi = roi
        self.segmenter.roi = roi

    def cycle_mode(self) -> str:
        self.mode = VISION_MODES[(VISION_MODES.index(self.mode) + 1) % len(VISION_MODES)]
        return self.mode

    def observe(self, rig_frame: RigFrame) -> VisionLabTelemetry:
        role, packet = self._external_packet(rig_frame)
        if packet is None:
            self.preview_bgr = None
            self._last_segmentation = None
            self._last_motion = None
            self.telemetry = VisionLabTelemetry(status="STEREN SIN CÁMARA")
            return self.telemetry

        frame = _resize_for_processing(packet.frame, self.process_width)
        if self._calibrating:
            self._calibration_buffer.append(frame)
            progress = len(self._calibration_buffer) / float(self.calibration_target)
            if len(self._calibration_buffer) >= self.calibration_target:
                self.background.calibrate(self._calibration_buffer)
                self._calibration_buffer = []
                self._calibrating = False
                self.segmenter.reset_tracking()
                self.motion.reset()
            else:
                self.preview_bgr = self._annotate(frame, None, None, f"CALIBRANDO FONDO {len(self._calibration_buffer)}/{self.calibration_target}")
                self.telemetry = VisionLabTelemetry(
                    external_role=role,
                    calibration_progress=progress,
                    sync_valid=self._sync(rig_frame, packet).valid,
                    sync_delta_ms=self._sync(rig_frame, packet).delta_ms,
                    frame_valid=True,
                    frame_id=getattr(packet, "frame_index", None),
                    camera_id=getattr(packet, "camera_id", None) or role,
                    source_width=getattr(packet, "width", None),
                    source_height=getattr(packet, "height", None),
                    rotation_deg=int(getattr(packet, "rotation_deg", 0)),
                    display_mirrored=bool(getattr(packet, "display_mirrored", False)),
                    status=f"STEREN CALIBRANDO {len(self._calibration_buffer)}/{self.calibration_target}",
                )
                return self.telemetry

        segmentation = (
            segment_foreground_result(
                frame,
                self.background.background_rgb,
                threshold=self.segmenter.threshold,
                roi=self.segmenter.roi,
                min_area=self.segmenter.min_area,
                previous_centroid=self.segmenter.previous_centroid,
                noise_floor=self.background.noise_floor,
            )
            if self.background.calibrated and self.background.background_rgb is not None
            else None
        )
        if segmentation is not None:
            self.segmenter.previous_centroid = (
                None if segmentation.component is None else segmentation.component.centroid
            )
        motion = self.motion.process(frame, packet.timestamp_ns or packet.timestamp_ms * 1_000_000)
        self._last_segmentation = segmentation
        self._last_motion = motion
        if segmentation is not None:
            reliable_background = ~segmentation.foreground_mask & ~motion.motion_mask
            self.background.update(frame, reliable_background)
        sync = self._sync(rig_frame, packet)
        self.preview_bgr = self._render_preview(frame, segmentation, motion)
        status = "STEREN CALIBRADO" if segmentation is not None else "STEREN: pulsa B para calibrar"
        if motion.global_change:
            status = "STEREN: cambio de luz; pulsa B para recalibrar"
        elif segmentation is not None and segmentation.confidence <= 0.05:
            status = "STEREN: silueta incierta"
        self.telemetry = VisionLabTelemetry(
            external_role=role,
            calibrated=self.background.calibrated,
            external_motion_score=motion.score,
            segmentation_confidence=0.0 if segmentation is None else segmentation.confidence,
            sync_valid=sync.valid,
            sync_delta_ms=sync.delta_ms,
            frame_valid=True,
            frame_id=getattr(packet, "frame_index", None),
            camera_id=getattr(packet, "camera_id", None) or role,
            source_width=getattr(packet, "width", None),
            source_height=getattr(packet, "height", None),
            rotation_deg=int(getattr(packet, "rotation_deg", 0)),
            display_mirrored=bool(getattr(packet, "display_mirrored", False)),
            status=f"{status} | {self.mode} | UMB {self.segmenter.threshold:.0f}",
        )
        return self.telemetry

    def handle_key(self, key: int) -> bool:
        """Controles de la vista: V modo, B fondo, +/- umbral, I ROI."""

        if key in (ord("v"), ord("V")):
            self.cycle_mode()
            return True
        if key in (ord("b"), ord("B")):
            self.request_recalibration()
            return True
        if key in (ord("+"), ord("=")):
            self.set_threshold(self.segmenter.threshold + 2.0)
            return True
        if key in (ord("-"), ord("_")):
            self.set_threshold(self.segmenter.threshold - 2.0)
            return True
        if key in (ord("i"), ord("I")):
            self.set_roi(None if self.roi is not None else (0.10, 0.05, 0.90, 0.98))
            return True
        return False

    def _external_packet(self, rig_frame: RigFrame) -> tuple[str | None, object | None]:
        for role in ("depth", "left", "right"):
            packet = rig_frame.by_role.get(role)
            if packet is not None:
                return role, packet
        return None, None

    def _sync(self, rig_frame: RigFrame, packet: object) -> SyncStatus:
        primary = rig_frame.primary
        auxiliary_ns = getattr(packet, "timestamp_ns", 0) or getattr(packet, "timestamp_ms", 0) * 1_000_000
        primary_ns = primary.timestamp_ns or primary.timestamp_ms * 1_000_000
        return nearest_timestamp(primary_ns, auxiliary_ns, tolerance_ms=self.calibration_tolerance_ms)

    def _render_preview(
        self,
        frame: NDArray[np.uint8],
        segmentation: SegmentationResult | None,
        motion: MotionResult,
    ) -> NDArray[np.uint8]:
        if self.mode == "FONDO":
            output = (
                np.array(self.background.background_rgb, copy=True)
                if self.background.background_rgb is not None
                else np.zeros_like(frame)
            )
        elif self.mode == "MOVIMIENTO":
            output = np.zeros_like(frame)
            output[motion.motion_mask] = (50, 190, 255)
        elif self.mode == "VALIDO":
            output = np.zeros_like(frame)
            valid = self.activity_mask
            if valid is not None:
                output[valid] = (40, 220, 120)
        elif segmentation is None:
            output = np.array(frame, copy=True)
        elif self.mode == "SILUETA":
            output = np.zeros_like(frame)
            output[segmentation.foreground_mask] = (224, 238, 250)
        elif self.mode == "SUPERPOSICION":
            output = np.array(frame, copy=True)
            foreground = segmentation.foreground_mask
            moving = self.activity_mask
            output[foreground] = (0.55 * output[foreground] + 0.45 * np.asarray((60, 220, 90))).astype(np.uint8)
            if moving is not None:
                output[moving] = (0.40 * output[moving] + 0.60 * np.asarray((40, 120, 255))).astype(np.uint8)
        else:
            output = np.array(frame, copy=True)
        return self._annotate(output, segmentation, motion, f"STEREN RGB  {self.mode}")

    def _annotate(
        self,
        image: NDArray[np.uint8],
        segmentation: SegmentationResult | None,
        motion: MotionResult | None,
        title: str,
    ) -> NDArray[np.uint8]:
        output = np.array(image, copy=True)
        height, width = output.shape[:2]
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            cv2.rectangle(output, (int(x0 * width), int(y0 * height)), (int(x1 * width) - 1, int(y1 * height) - 1), (255, 210, 50), 1)
        cv2.putText(output, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 247, 250), 1, cv2.LINE_AA)
        detail = "B recalibra  V vista  +/- umbral  I ROI"
        if segmentation is not None and motion is not None:
            detail = f"FG {segmentation.confidence:.0%}  MOV {motion.score:.1%}  " + detail
        cv2.putText(output, detail[:100], (8, max(34, height - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 230, 235), 1, cv2.LINE_AA)
        return output

    def close(self) -> None:
        self.preview_bgr = None
        self._calibration_buffer.clear()
