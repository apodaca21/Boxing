"""Morfología binaria y componentes conexos implementados con NumPy.

No usa ``cv2.morphologyEx`` ni ``cv2.findContours``. Las funciones operan en
arrays booleanos para que sean fáciles de probar con imágenes sintéticas.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]


def _check_mask(mask: NDArray[np.bool_]) -> BoolArray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("La máscara debe ser un array 2D.")
    return value


def _reduce_window(mask: BoolArray, *, operation: str, iterations: int) -> BoolArray:
    result = _check_mask(mask)
    if iterations < 0:
        raise ValueError("iterations no puede ser negativo.")
    for _ in range(iterations):
        pad_value = operation == "min"
        padded = np.pad(result, 1, mode="constant", constant_values=pad_value)
        windows = [
            padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        if operation == "min":
            result = np.logical_and.reduce(windows)
        else:
            result = np.logical_or.reduce(windows)
    return result


def erode(mask: NDArray[np.bool_], iterations: int = 1) -> BoolArray:
    """Erosión 3x3 propia mediante AND de ventanas desplazadas."""

    return _reduce_window(mask, operation="min", iterations=iterations)


def dilate(mask: NDArray[np.bool_], iterations: int = 1) -> BoolArray:
    """Dilatación 3x3 propia mediante OR de ventanas desplazadas."""

    return _reduce_window(mask, operation="max", iterations=iterations)


def opening(mask: NDArray[np.bool_], iterations: int = 1) -> BoolArray:
    return dilate(erode(mask, iterations), iterations)


def closing(mask: NDArray[np.bool_], iterations: int = 1) -> BoolArray:
    return erode(dilate(mask, iterations), iterations)


@dataclass(frozen=True, slots=True)
class ConnectedComponent:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    pixels: tuple[tuple[int, int], ...] = ()


def connected_components(mask: NDArray[np.bool_], min_area: int = 1) -> tuple[ConnectedComponent, ...]:
    """Encuentra componentes 8-conexos usando BFS explícito."""

    value = _check_mask(mask)
    if min_area < 1:
        raise ValueError("min_area debe ser positivo.")
    height, width = value.shape
    visited = np.zeros_like(value, dtype=bool)
    components: list[ConnectedComponent] = []
    label = 0
    for y in range(height):
        for x in range(width):
            if not value[y, x] or visited[y, x]:
                continue
            label += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            points: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                points.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and value[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            area = len(points)
            if area < min_area:
                continue
            ys = np.fromiter((point[0] for point in points), dtype=np.float64)
            xs = np.fromiter((point[1] for point in points), dtype=np.float64)
            components.append(
                ConnectedComponent(
                    label=label,
                    area=area,
                    bbox=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
                    centroid=(float(xs.mean()), float(ys.mean())),
                    pixels=tuple(points),
                )
            )
    return tuple(sorted(components, key=lambda item: item.area, reverse=True))


def keep_components(
    mask: NDArray[np.bool_],
    *,
    min_area: int = 1,
    max_components: int | None = None,
) -> BoolArray:
    """Conserva componentes grandes sin llamar a APIs de visión de alto nivel."""

    value = _check_mask(mask)
    components = connected_components(value, min_area=min_area)
    if max_components is not None:
        if max_components < 1:
            raise ValueError("max_components debe ser positivo.")
        components = components[:max_components]
    result = np.zeros_like(value, dtype=bool)
    for component in components:
        for y, x in component.pixels:
            result[y, x] = True
    return result
