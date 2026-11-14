"""Segmentación de primer plano por diferencia contra el fondo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.vision_lab.morphology import (
    ConnectedComponent,
    closing,
    connected_components,
    keep_components,
    opening,
)


BoolArray = NDArray[np.bool_]


def _validate_frame_pair(frame: NDArray[np.uint8], background: NDArray[np.uint8]) -> None:
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame debe ser una imagen HxWx3.")
    if background.shape != frame.shape:
        raise ValueError("frame y background deben tener el mismo tamaño.")


def difference_metric(
    frame: NDArray[np.uint8],
    background: NDArray[np.uint8],
    *,
    metric: str = "mean",
) -> NDArray[np.float32]:
    """Diferencia absoluta por píxel; ``mean`` y ``max`` son métricas explicables."""

    _validate_frame_pair(frame, background)
    if metric not in {"mean", "max"}:
        raise ValueError("metric debe ser mean o max.")
    difference = np.abs(frame.astype(np.float32) - background.astype(np.float32))
    if metric == "max":
        return np.max(difference[:, :, :3], axis=2)
    return np.mean(difference[:, :, :3], axis=2)


def _roi_mask(shape: tuple[int, int], roi: tuple[float, float, float, float] | None) -> BoolArray:
    height, width = shape
    if roi is None:
        return np.ones(shape, dtype=bool)
    x0, y0, x1, y1 = roi
    if not 0.0 <= x0 < x1 <= 1.0 or not 0.0 <= y0 < y1 <= 1.0:
        raise ValueError("ROI debe estar en coordenadas normalizadas y ser no vacía.")
    result = np.zeros(shape, dtype=bool)
    left = int(np.floor(x0 * width))
    top = int(np.floor(y0 * height))
    right = max(left + 1, int(np.ceil(x1 * width)))
    bottom = max(top + 1, int(np.ceil(y1 * height)))
    result[top:min(bottom, height), left:min(right, width)] = True
    return result


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    foreground_mask: BoolArray
    raw_mask: BoolArray
    difference: NDArray[np.float32]
    component: ConnectedComponent | None
    confidence: float


def _choose_component(
    mask: BoolArray,
    components: tuple[ConnectedComponent, ...],
    previous_centroid: tuple[float, float] | None,
) -> ConnectedComponent | None:
    if not components:
        return None
    height, width = mask.shape
    image_area = float(height * width)
    def score(component: ConnectedComponent) -> float:
        area_score = min(component.area / max(image_area * 0.70, 1.0), 1.0)
        x, y = component.centroid
        center_score = 1.0 - min(float(np.hypot(x - width / 2.0, y - height / 2.0)) / max(width, height), 1.0)
        continuity = 0.0
        if previous_centroid is not None:
            continuity = 1.0 - min(float(np.hypot(x - previous_centroid[0], y - previous_centroid[1])) / max(width, height), 1.0)
        return 0.55 * area_score + 0.25 * center_score + 0.20 * continuity
    return max(components, key=score)


def segment_foreground_result(
    frame: NDArray[np.uint8],
    background: NDArray[np.uint8],
    threshold: float = 25.0,
    roi: tuple[float, float, float, float] | None = None,
    *,
    metric: str = "mean",
    min_area: int = 24,
    opening_iterations: int = 1,
    closing_iterations: int = 1,
    previous_centroid: tuple[float, float] | None = None,
    noise_floor: NDArray[np.float32] | None = None,
    noise_scale: float = 2.0,
) -> SegmentationResult:
    if threshold < 0:
        raise ValueError("threshold no puede ser negativo.")
    difference = difference_metric(frame, background, metric=metric)
    roi_mask = _roi_mask(difference.shape, roi)
    if noise_scale < 0:
        raise ValueError("noise_scale no puede ser negativo.")
    if noise_floor is not None:
        noise = np.asarray(noise_floor, dtype=np.float32)
        if noise.shape != difference.shape:
            raise ValueError("noise_floor debe coincidir con el tamaño del fotograma.")
        effective_threshold = float(threshold) + float(noise_scale) * noise
    else:
        effective_threshold = float(threshold)
    raw = (difference >= effective_threshold) & roi_mask
    cleaned = raw
    if opening_iterations:
        cleaned = opening(cleaned, opening_iterations)
    if closing_iterations:
        cleaned = closing(cleaned, closing_iterations)
    components = connected_components(cleaned, min_area=max(1, min_area))
    selected = _choose_component(cleaned, components, previous_centroid)
    selected_mask = np.zeros_like(cleaned)
    if selected is not None:
        for y, x in selected.pixels:
            selected_mask[y, x] = True
    area_ratio = 0.0 if selected is None else selected.area / float(cleaned.size)
    confidence = 0.0 if selected is None else float(np.clip(area_ratio * 12.0, 0.0, 1.0))
    return SegmentationResult(selected_mask, raw, difference, selected, confidence)


def segment_foreground(
    frame: NDArray[np.uint8],
    background: NDArray[np.uint8],
    threshold: float = 25.0,
    roi: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> BoolArray:
    """API simple solicitada por la especificación: devuelve máscara booleana."""

    return segment_foreground_result(frame, background, threshold, roi, **kwargs).foreground_mask


class ForegroundSegmenter:
    """Estado temporal de segmentación con continuidad de componente."""

    def __init__(self, *, threshold: float = 25.0, min_area: int = 24) -> None:
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self.roi: tuple[float, float, float, float] | None = None
        self.previous_centroid: tuple[float, float] | None = None

    def reset_tracking(self) -> None:
        self.previous_centroid = None

    def process(self, frame: NDArray[np.uint8], background: NDArray[np.uint8]) -> SegmentationResult:
        result = segment_foreground_result(
            frame,
            background,
            threshold=self.threshold,
            roi=self.roi,
            min_area=self.min_area,
            previous_centroid=self.previous_centroid,
        )
        self.previous_centroid = None if result.component is None else result.component.centroid
        return result
