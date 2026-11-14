"""Detección de cambios locales entre fotogramas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.vision_lab.morphology import closing, opening, keep_components


@dataclass(frozen=True, slots=True)
class MotionResult:
    motion_mask: NDArray[np.bool_]
    score: float
    changed_pixels: int
    dt_s: float | None
    global_change: bool = False
    quality: float = 1.0


class MotionDetector:
    """Señal de movimiento separada de la silueta de primer plano."""

    def __init__(self, *, threshold: float = 18.0, min_area: int = 8) -> None:
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self._previous: NDArray[np.uint8] | None = None
        self._previous_timestamp_ns: int | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_timestamp_ns = None

    def process(self, frame: NDArray[np.uint8], timestamp_ns: int | None = None) -> MotionResult:
        current = np.asarray(frame, dtype=np.uint8)
        if current.ndim != 3 or current.shape[2] < 3:
            raise ValueError("frame debe ser una imagen HxWx3.")
        if self._previous is None:
            self._previous = np.array(current, copy=True)
            self._previous_timestamp_ns = timestamp_ns
            return MotionResult(np.zeros(current.shape[:2], dtype=bool), 0.0, 0, None)
        # Compensación robusta de exposición: si toda la imagen cambia por
        # una variación de luz, la mediana por canal se elimina y no termina
        # convertida en una enorme "zona caliente" del fondo.
        delta = current.astype(np.float32) - self._previous.astype(np.float32)
        bias = np.median(delta.reshape(-1, delta.shape[2]), axis=0)
        global_change = float(np.mean(np.abs(bias))) >= max(12.0, self.threshold * 0.75)
        difference = np.mean(np.abs(delta - bias), axis=2)
        mask = difference >= self.threshold
        mask = opening(mask, 1)
        mask = closing(mask, 1)
        mask = keep_components(mask, min_area=max(1, self.min_area))
        dt_s = None
        if timestamp_ns is not None and self._previous_timestamp_ns is not None:
            delta = timestamp_ns - self._previous_timestamp_ns
            if delta > 0:
                dt_s = delta / 1_000_000_000.0
        self._previous = np.array(current, copy=True)
        self._previous_timestamp_ns = timestamp_ns
        changed = int(np.count_nonzero(mask))
        return MotionResult(
            mask,
            changed / float(mask.size),
            changed,
            dt_s,
            global_change=global_change,
            quality=0.0 if global_change else 1.0,
        )
