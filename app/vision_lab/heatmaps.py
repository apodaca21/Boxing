"""Acumuladores numéricos para los mapas de la Fase 3."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


def accumulate_motion(heatmap: NDArray[np.float32], mask: NDArray[np.bool_], dt_s: float) -> NDArray[np.float32]:
    """Suma actividad ponderada por tiempo en el canvas de la Steren."""

    if dt_s < 0:
        raise ValueError("dt_s no puede ser negativo.")
    if heatmap.shape != mask.shape:
        raise ValueError("heatmap y mask deben tener el mismo tamaño.")
    heatmap += np.asarray(mask, dtype=np.float32) * float(dt_s)
    return heatmap


def heatmap_to_bgr(
    values: NDArray[np.float32],
    *,
    maximum: float | None = None,
) -> NDArray[np.uint8]:
    """Colorea un mapa con NumPy; no usa ``cv2.applyColorMap``.

    ``maximum`` permite comparar izquierda/derecha con una misma escala
    absoluta en lugar de normalizar cada brazo por separado.
    """

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("El heatmap debe ser 2D.")
    top = float(np.max(array)) if maximum is None else float(maximum)
    linear = np.zeros_like(array) if top <= 0 else np.clip(array / top, 0.0, 1.0)
    # Una escala lineal oculta casi toda la trayectoria cuando un punto tiene
    # mucha más actividad que el resto. La raíz conserva el orden, pero hace
    # visibles las zonas débiles sin inventar actividad en los ceros.
    normalized = np.sqrt(linear, dtype=np.float32)
    stops = np.asarray(
        (
            (8, 8, 35),
            (150, 35, 45),
            (230, 150, 25),
            (20, 235, 245),
        ),
        dtype=np.float32,
    )
    scaled = normalized * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (scaled - lower)[..., None]
    image = stops[lower] * (1.0 - fraction) + stops[upper] * fraction
    return np.clip(image, 0, 255).astype(np.uint8)


def overlay_heatmap_on_frame(
    frame_bgr: NDArray[np.uint8],
    values: NDArray[np.float32],
    *,
    maximum: float | None = None,
    alpha: float = 0.68,
    mirror: bool = False,
) -> NDArray[np.uint8]:
    """Superpone actividad sobre una captura real, conservando el contexto."""

    frame = np.asarray(frame_bgr, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3 or values.ndim != 2:
        raise ValueError("frame debe ser HxWx3 y values debe ser 2D.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe estar entre 0 y 1.")
    if mirror:
        frame = np.fliplr(frame)
        values = np.fliplr(values)
    target_height, target_width = frame.shape[:2]
    if values.shape != (target_height, target_width):
        values = resize_bilinear(values, target_height, target_width)
    top = float(np.max(values)) if maximum is None else float(maximum)
    linear = np.zeros_like(values, dtype=np.float32) if top <= 0 else np.clip(values / top, 0.0, 1.0)
    normalized = np.sqrt(linear, dtype=np.float32)
    color = heatmap_to_bgr(values, maximum=top)
    present = linear > 1e-6
    # Un punto tenue sigue coloreando la captura; los píxeles sin actividad
    # quedan intactos para conservar el contexto real de la webcam.
    weight = np.where(
        present[..., None],
        alpha * (0.12 + 0.88 * normalized[..., None]),
        0.0,
    )
    weight = np.clip(weight, 0.0, alpha)
    blended = frame.astype(np.float32) * (1.0 - weight) + color.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def enhance_reference_frame(frame_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Mejora una captura oscura sólo para la vista/exportación del mapa.

    No se usa para el clasificador ni para la segmentación. Conserva la
    composición de la cámara, pero aplica contraste por percentiles y una
    gamma suave para que una webcam con exposición baja no se vea negra.
    """

    frame = np.asarray(frame_bgr, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame_bgr debe ser una imagen HxWx3.")
    source = frame[:, :, :3].astype(np.float32)
    luminance = np.mean(source, axis=2)
    low, high = np.percentile(luminance, (2.0, 98.0))
    spread = float(high - low)
    if spread < 8.0:
        adjusted = source * 1.65 + 10.0
    else:
        adjusted = (source - float(low)) * (255.0 / spread)
    adjusted = np.clip(adjusted, 0.0, 255.0)
    median_level = float(np.median(np.mean(adjusted, axis=2))) / 255.0
    gamma = float(np.clip(0.70 + 0.48 * median_level, 0.70, 1.0))
    adjusted = 255.0 * np.power(adjusted / 255.0, gamma)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def add_heatmap_legend(
    image: NDArray[np.uint8],
    *,
    maximum: float,
    label: str,
) -> NDArray[np.uint8]:
    """Añade una leyenda baja-alta sin alterar el array numérico exportado."""

    output = np.array(image, copy=True)
    height, width = output.shape[:2]
    bar_width = max(1, min(220, max(40, width // 3), width - 24))
    x0, y0 = min(12, max(0, width - 1)), max(0, height - 34)
    bar_height = max(1, min(12, height - y0))
    gradient_values = np.linspace(maximum, 0.0, bar_width, dtype=np.float32)[None, :]
    gradient = heatmap_to_bgr(gradient_values, maximum=max(maximum, 1e-12))
    output[y0 : y0 + bar_height, x0 : x0 + bar_width] = np.repeat(gradient, bar_height, axis=0)
    cv2.rectangle(output, (x0, y0), (x0 + bar_width - 1, y0 + bar_height - 1), (245, 247, 250), 1, cv2.LINE_AA)
    if height >= 40:
        cv2.putText(output, "alta", (x0 + bar_width - 30, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (245, 247, 250), 1, cv2.LINE_AA)
        cv2.putText(output, "baja", (x0, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (245, 247, 250), 1, cv2.LINE_AA)
        cv2.putText(output, f"{label}  max {maximum:.2f}", (x0 + bar_width + 12, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 247, 250), 1, cv2.LINE_AA)
    return output


def radial_kernel(radius: int = 10) -> NDArray[np.float32]:
    if radius < 0:
        raise ValueError("radius no puede ser negativo.")
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    sigma = max(radius / 2.0, 1.0)
    return np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)


def resize_bilinear(
    values: NDArray[np.float32],
    height: int,
    width: int,
) -> NDArray[np.float32]:
    """Redimensiona un mapa 2D con interpolación bilineal NumPy propia."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or height < 1 or width < 1:
        raise ValueError("values debe ser 2D y el tamaño destino debe ser positivo.")
    if source.shape == (height, width):
        return np.array(source, copy=True)
    y = np.linspace(0.0, source.shape[0] - 1, height, dtype=np.float32)
    x = np.linspace(0.0, source.shape[1] - 1, width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    top = source[y0[:, None], x0[None, :]] * (1.0 - wx)
    top += source[y0[:, None], x1[None, :]] * wx
    bottom = source[y1[:, None], x0[None, :]] * (1.0 - wx)
    bottom += source[y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


@dataclass(slots=True)
class HeatmapAccumulator:
    shape: tuple[int, int]
    values: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError("shape debe ser una pareja de dimensiones positivas.")
        if self.values is None:
            self.values = np.zeros(self.shape, dtype=np.float32)
        elif self.values.shape != self.shape:
            raise ValueError("values no coincide con shape.")

    def add_motion(self, mask: NDArray[np.bool_], dt_s: float) -> None:
        assert self.values is not None
        accumulate_motion(self.values, mask, dt_s)

    def add_point(self, x: float, y: float, *, weight: float = 1.0, radius: int = 10) -> None:
        if self.values is None:
            raise RuntimeError("El acumulador no está inicializado.")
        if not np.isfinite((x, y)).all() or weight < 0:
            return
        height, width = self.shape
        center_x = int(round(x * (width - 1))) if 0.0 <= x <= 1.0 else int(round(x))
        center_y = int(round(y * (height - 1))) if 0.0 <= y <= 1.0 else int(round(y))
        kernel = radial_kernel(radius) * float(weight)
        radius_y, radius_x = radius, radius
        y0, y1 = max(0, center_y - radius_y), min(height, center_y + radius_y + 1)
        x0, x1 = max(0, center_x - radius_x), min(width, center_x + radius_x + 1)
        ky0, ky1 = y0 - center_y + radius_y, y1 - center_y + radius_y
        kx0, kx1 = x0 - center_x + radius_x, x1 - center_x + radius_x
        self.values[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]

    def normalized(self) -> NDArray[np.float32]:
        assert self.values is not None
        maximum = float(np.max(self.values))
        return np.zeros_like(self.values) if maximum <= 0 else self.values / maximum
