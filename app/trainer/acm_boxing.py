"""LSTM y preprocesamiento publicados por ACM40960/Boxing.

Copyright (c) 2025 Jayaganeshan Thanga Kumar, Kabilesh Sekar.

El repositorio original se distribuye con licencia MIT. Este módulo no copia
la aplicación: reproduce la red y la normalización necesarias para cargar su
state_dict. El texto completo de la licencia está en
``models/external/LICENSE``.

La entrada es una secuencia de 25 fotogramas por 16 valores: ocho keypoints
YOLO (x, y), centrados en los hombros y divididos por la distancia entre
hombros. No acepta el vector NEAT de 92 características.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from app.dataset_manager import MOVEMENTS
from app.trainer.results import (
    STATUS_COMPLETED,
    STATUS_NEUTRAL,
    ClassificationResult,
    PunchEvent,
    neutral_result,
    no_detection_result,
)


KEYPOINT_MAP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

ACTION_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
PUNCH_ACTIONS = ("jab", "cross", "hook", "uppercut")
BLOCK_ACTIONS = ("forearm_block", "high_guard", "parry", "negative")
SEQUENCE_LENGTH = 25
INPUT_SIZE = len(ACTION_KEYPOINTS) * 2
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
YOLO_KEYPOINTS = 17


@dataclass(frozen=True, slots=True)
class ExternalSettings:
    """Umbrales tomados del analizador original y ajustables."""

    sequence_length: int = SEQUENCE_LENGTH
    confidence_threshold: float = 0.7
    keypoint_confidence: float = 0.5
    min_punch_speed: float = 2.0
    # El analizador original esperaba 60 fotogramas y por eso una combinación
    # desaparecía. 20 fotogramas separan eventos sin pedir una secuencia nueva.
    recovery_frames: int = 20
    max_missing_frames: int = 8
    arm_ratio: float = 1.4
    min_arm_pixels: float = 8.0


def normalize_keypoints(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    *,
    keypoint_confidence: float = 0.5,
) -> np.ndarray | None:
    """Reproduce la normalización original. Un punto poco fiable queda en cero."""

    points = np.asarray(keypoints, dtype=np.float64)
    scores = np.asarray(confidences, dtype=np.float64)
    if points.shape[0] < YOLO_KEYPOINTS or scores.shape[0] < YOLO_KEYPOINTS:
        return None
    left_shoulder = KEYPOINT_MAP["left_shoulder"]
    right_shoulder = KEYPOINT_MAP["right_shoulder"]
    if scores[left_shoulder] < keypoint_confidence or scores[right_shoulder] < keypoint_confidence:
        return None
    scale = float(np.linalg.norm(points[left_shoulder] - points[right_shoulder]))
    if scale < 1e-4:
        return None
    indices = [KEYPOINT_MAP[name] for name in ACTION_KEYPOINTS]
    selected = points[indices]
    selected_scores = scores[indices]
    center = (points[left_shoulder] + points[right_shoulder]) / 2.0
    normalized = np.zeros_like(selected)
    valid = selected_scores > keypoint_confidence
    normalized[valid] = (selected[valid] - center) / scale
    flat = normalized.reshape(-1)
    if flat.shape != (INPUT_SIZE,):
        return None
    return flat.astype(np.float32)


def punch_speed(
    frames: list[tuple[np.ndarray, np.ndarray]],
    *,
    keypoint_confidence: float = 0.5,
) -> float:
    """Velocidad en píxeles, igual que el cálculo original sobre las últimas muestras."""

    recent = frames[-5:]
    if len(recent) < 2:
        return 0.0
    speeds: list[float] = []
    for name in ("left_wrist", "right_wrist"):
        index = KEYPOINT_MAP[name]
        positions = [
            frame[0][index]
            for frame in recent
            if frame[1][index] > keypoint_confidence
        ]
        if len(positions) < 2:
            continue
        distance = sum(
            float(np.linalg.norm(positions[cursor + 1] - positions[cursor]))
            for cursor in range(len(positions) - 1)
        )
        speeds.append(distance / len(positions))
    return max(speeds) if speeds else 0.0


def dominant_arm(
    frames: list[tuple[np.ndarray, np.ndarray]],
    *,
    keypoint_confidence: float = 0.5,
    arm_ratio: float = 1.4,
    min_arm_pixels: float = 8.0,
) -> str | None:
    """Brazo con un desplazamiento claramente mayor. None si la evidencia no alcanza."""

    if len(frames) < 2:
        return None
    start_points, start_scores = frames[0]
    end_points, end_scores = frames[-1]

    def travel(name: str) -> float | None:
        index = KEYPOINT_MAP[name]
        if start_scores[index] <= keypoint_confidence or end_scores[index] <= keypoint_confidence:
            return None
        return float(np.linalg.norm(end_points[index] - start_points[index]))

    left = travel("left_wrist")
    right = travel("right_wrist")
    if left is None or right is None:
        return None
    if left >= right * arm_ratio and left >= min_arm_pixels:
        return "left"
    if right >= left * arm_ratio and right >= min_arm_pixels:
        return "right"
    return None


def punch_event(action: str, arm: str | None) -> PunchEvent:
    """Traduce una clase del LSTM sin asignar jab, cross ni uppercut a un lado."""

    if action == "jab":
        return PunchEvent(None, "Jab", arm, "jab")
    if action == "cross":
        return PunchEvent(None, "Cross", arm, "cross")
    if action == "uppercut":
        return PunchEvent(None, "Uppercut", arm, "uppercut")
    if action == "hook" and arm == "left":
        movement = MOVEMENTS[3]
        return PunchEvent(movement.label_id, movement.display_name, arm, None)
    if action == "hook" and arm == "right":
        movement = MOVEMENTS[4]
        return PunchEvent(movement.label_id, movement.display_name, arm, None)
    return PunchEvent(None, "HOOK", None, "hook")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponential = np.exp(shifted)
    total = float(np.sum(exponential))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(logits.shape, 1.0 / logits.size)
    return exponential / total


class SequenceRecognizer:
    """Confirma un golpe una sola vez y espera antes del siguiente evento."""

    def __init__(
        self,
        predict_punch,
        *,
        predict_block=None,
        settings: ExternalSettings | None = None,
    ) -> None:
        self._predict_punch = predict_punch
        self._predict_block = predict_block
        self.settings = settings or ExternalSettings()
        self._frames: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=self.settings.sequence_length
        )
        self._recovery = 0
        self._missing = 0
        self._last_name = MOVEMENTS[0].display_name

    def reset(self) -> None:
        self._frames.clear()
        self._recovery = 0
        self._missing = 0
        self._last_name = MOVEMENTS[0].display_name

    def lost(self) -> ClassificationResult:
        self._missing += 1
        if self._missing > self.settings.max_missing_frames:
            self._frames.clear()
            self._recovery = 0
        return no_detection_result("external")

    def push(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray,
    ) -> ClassificationResult:
        normalized = normalize_keypoints(
            keypoints,
            confidences,
            keypoint_confidence=self.settings.keypoint_confidence,
        )
        if normalized is None:
            return self.lost()
        self._missing = 0
        self._frames.append(
            (
                np.asarray(keypoints, dtype=np.float64),
                np.asarray(confidences, dtype=np.float64),
                normalized,
            )
        )
        if self._recovery > 0:
            self._recovery -= 1
            raw_frames = [(frame[0], frame[1]) for frame in self._frames]
            speed = punch_speed(
                raw_frames,
                keypoint_confidence=self.settings.keypoint_confidence,
            )
            return ClassificationResult(
                source="external",
                status=STATUS_NEUTRAL,
                label_id=None,
                display_name=self._last_name,
                rejection="cooldown" if speed >= self.settings.min_punch_speed else None,
            )
        if len(self._frames) < self.settings.sequence_length:
            return neutral_result("external")
        raw_frames = [(frame[0], frame[1]) for frame in self._frames]
        if punch_speed(raw_frames, keypoint_confidence=self.settings.keypoint_confidence) < self.settings.min_punch_speed:
            return neutral_result("external")
        sequence = np.stack([frame[2] for frame in self._frames]).astype(np.float32)
        if sequence.shape != (self.settings.sequence_length, INPUT_SIZE):
            return no_detection_result("external")
        try:
            logits = np.asarray(self._predict_punch(sequence), dtype=np.float64).reshape(-1)
        except Exception:
            return ClassificationResult(
                source="external",
                status="no_detection",
                label_id=None,
                display_name="ERROR DE MODELO",
            )
        if logits.shape != (len(PUNCH_ACTIONS),) or not np.all(np.isfinite(logits)):
            return ClassificationResult(
                source="external",
                status="no_detection",
                label_id=None,
                display_name="ERROR DE MODELO",
            )
        scores = _softmax(logits)
        choice = int(np.argmax(scores))
        if float(scores[choice]) < self.settings.confidence_threshold:
            return self._block_or_neutral(sequence)
        action = PUNCH_ACTIONS[choice]
        arm = dominant_arm(
            raw_frames,
            keypoint_confidence=self.settings.keypoint_confidence,
            arm_ratio=self.settings.arm_ratio,
            min_arm_pixels=self.settings.min_arm_pixels,
        )
        event = punch_event(action, arm)
        self._last_name = event.display_name
        self._recovery = self.settings.recovery_frames
        return ClassificationResult(
            source="external",
            status=STATUS_COMPLETED,
            label_id=event.label_id,
            display_name=event.display_name,
            events=(event,),
            active_arm=event.arm,
        )

    def _block_or_neutral(self, sequence: np.ndarray) -> ClassificationResult:
        if self._predict_block is None:
            return neutral_result("external")
        try:
            logits = np.asarray(self._predict_block(sequence), dtype=np.float64).reshape(-1)
        except Exception:
            return neutral_result("external")
        if logits.shape != (len(BLOCK_ACTIONS),) or not np.all(np.isfinite(logits)):
            return neutral_result("external")
        scores = _softmax(logits)
        choice = int(np.argmax(scores))
        if float(scores[choice]) < self.settings.confidence_threshold:
            return neutral_result("external")
        action = BLOCK_ACTIONS[choice]
        if action == "high_guard":
            return neutral_result("external")
        if action == "negative":
            return neutral_result("external")
        display = "Bloqueo de antebrazo" if action == "forearm_block" else "Parada"
        self._last_name = display
        self._recovery = self.settings.recovery_frames
        return ClassificationResult(
            source="external",
            status=STATUS_NEUTRAL,
            label_id=None,
            display_name=display,
        )
