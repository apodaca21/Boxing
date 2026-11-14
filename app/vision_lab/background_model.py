"""Modelo de fondo por mediana y actualización conservadora."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BackgroundModel:
    """Referencia RGB/BGR local, sin asumir profundidad métrica."""

    background_rgb: NDArray[np.uint8] | None = None
    noise_floor: NDArray[np.float32] | None = None
    freeze_background: bool = False
    update_alpha: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.update_alpha <= 1.0:
            raise ValueError("update_alpha debe estar entre 0 y 1.")

    @property
    def calibrated(self) -> bool:
        return self.background_rgb is not None

    def calibrate(self, frames: list[NDArray[np.uint8]] | tuple[NDArray[np.uint8], ...]) -> NDArray[np.uint8]:
        if not frames:
            raise ValueError("Se necesita al menos un fotograma para calibrar.")
        arrays = [np.asarray(frame) for frame in frames]
        first_shape = arrays[0].shape
        if any(frame.shape != first_shape for frame in arrays):
            raise ValueError("Todos los fotogramas de calibración deben tener el mismo tamaño.")
        if any(frame.ndim != 3 or frame.shape[2] < 3 for frame in arrays):
            raise ValueError("Los fotogramas deben ser imágenes a color HxWx3.")
        stack = np.stack(arrays, axis=0).astype(np.float32)
        median = np.median(stack, axis=0)
        # MAD robusta por píxel: permite elevar el umbral únicamente donde la
        # cámara tiene ruido, sin convertir una variación global en silueta.
        mad = np.median(np.abs(stack - median), axis=0)
        self.background_rgb = np.clip(median, 0, 255).astype(np.uint8)
        self.noise_floor = np.maximum(
            1.0,
            1.4826 * np.mean(mad[:, :, :3], axis=2),
        ).astype(np.float32)
        return np.array(self.background_rgb, copy=True)

    def reset(self) -> None:
        self.background_rgb = None
        self.noise_floor = None

    def update(self, frame: NDArray[np.uint8], reliable_background: NDArray[np.bool_] | None = None) -> None:
        """Actualiza sólo píxeles confiables; nunca absorbe la silueta por defecto."""

        if self.freeze_background or self.background_rgb is None:
            return
        current = np.asarray(frame, dtype=np.uint8)
        if current.shape != self.background_rgb.shape:
            raise ValueError("El fotograma no coincide con el fondo calibrado.")
        if reliable_background is None:
            reliable_background = np.ones(current.shape[:2], dtype=bool)
        reliable = np.asarray(reliable_background, dtype=bool)
        if reliable.shape != current.shape[:2]:
            raise ValueError("La máscara de fondo confiable no coincide con la imagen.")
        alpha = self.update_alpha
        base = self.background_rgb.astype(np.float32)
        current_float = current.astype(np.float32)
        base[reliable] = (1.0 - alpha) * base[reliable] + alpha * current_float[reliable]
        self.background_rgb = np.clip(base, 0, 255).astype(np.uint8)
