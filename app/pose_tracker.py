"""Seguimiento corporal con MediaPipe Tasks Pose Landmarker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray


class LandmarkId(IntEnum):
    """Índices oficiales del modelo de pose de MediaPipe."""

    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


@dataclass(frozen=True, slots=True)
class LandmarkObservation:
    """Una articulación con coordenadas y calidad de estimación."""

    x: float
    y: float
    z: float
    visibility: float
    presence: float

    @property
    def is_finite(self) -> bool:
        return all(
            math.isfinite(value)
            for value in (self.x, self.y, self.z, self.visibility, self.presence)
        )

    def is_reliable(
        self,
        min_visibility: float,
        min_presence: float,
    ) -> bool:
        return (
            self.is_finite
            and self.visibility >= min_visibility
            and self.presence >= min_presence
        )


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """Resultado desacoplado de MediaPipe para un instante de video."""

    timestamp_ms: int
    landmarks: tuple[LandmarkObservation, ...] = ()
    world_landmarks: tuple[LandmarkObservation, ...] = ()

    @property
    def detected(self) -> bool:
        return bool(self.landmarks)

    def landmark(self, landmark_id: LandmarkId | int) -> LandmarkObservation | None:
        index = int(landmark_id)
        if index < 0 or index >= len(self.landmarks):
            return None
        return self.landmarks[index]


class PoseTracker:
    """Adaptador síncrono sobre PoseLandmarker en modo VIDEO."""

    def __init__(
        self,
        model_path: Path,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"El modelo Pose Landmarker no existe o está vacío: '{model_path}'."
            )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    @staticmethod
    def _convert_landmarks(raw_landmarks: object) -> tuple[LandmarkObservation, ...]:
        return tuple(
            LandmarkObservation(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=float(landmark.visibility),
                presence=float(landmark.presence),
            )
            for landmark in raw_landmarks
        )

    def detect(
        self,
        frame_bgr: NDArray[np.uint8],
        timestamp_ms: int,
    ) -> PoseObservation:
        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError(
                "PoseLandmarker VIDEO requiere timestamps estrictamente crecientes."
            )
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("El fotograma de entrada está vacío o no es un ndarray.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        if not result.pose_landmarks:
            return PoseObservation(timestamp_ms=timestamp_ms)

        normalized = self._convert_landmarks(result.pose_landmarks[0])
        world = (
            self._convert_landmarks(result.pose_world_landmarks[0])
            if result.pose_world_landmarks
            else ()
        )
        return PoseObservation(
            timestamp_ms=timestamp_ms,
            landmarks=normalized,
            world_landmarks=world,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseTracker:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
