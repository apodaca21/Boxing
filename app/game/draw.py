"""Interfaz del juego sobre el fotograma ya reflejado."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.dataset_manager import MOVEMENTS
from app.game.challenge import TARGET_STYLE, ChallengeSnapshot
from app.game.free_round import FreeSnapshot
from app.targets.placement import display_point, offset_for, project_target, radius_for


def _plain(text: str) -> str:
    return text.translate(str.maketrans("áéíóúÁÉÍÓÚñÑüÜ", "aeiouAEIOUnNuU"))


def _clock(ms: int) -> str:
    total = max(0, ms) // 1000
    return f"{total // 60:02d}:{total % 60:02d}"


def _panel(image: NDArray[np.uint8], width: int, height: int) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)


def _write_lines(
    image: NDArray[np.uint8],
    lines: list[str],
    *,
    x: int = 16,
    y: int = 28,
    step: int = 26,
    scale: float = 0.52,
) -> None:
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line[:36],
            (x, y + index * step),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 248, 252),
            1,
            cv2.LINE_AA,
        )


def draw_free(
    image: NDArray[np.uint8],
    view: FreeSnapshot,
    *,
    fps: float | None,
    tracking: bool,
) -> None:
    lines = [
        "EVOBOX AI  |  MODO LIBRE",
        f"TIEMPO {_clock(view.remaining_ms if view.phase == 'active' else view.elapsed_ms)}",
        *[f"{_plain(name).upper()}  {count}" for name, count, _side in view.rows],
        f"TOTAL {view.total}   ULTIMO {_plain(view.last_punch)}",
        f"COMBO {_plain(view.combo)}",
        f"{view.classifier.upper()}  |  {'SEGUIMIENTO OK' if tracking else 'SIN SEGUIMIENTO'}",
        f"FPS {0 if fps is None else round(fps)}",
    ]
    if view.phase == "idle":
        lines.append("ESPACIO PARA INICIAR")
    elif view.phase == "countdown":
        lines.append(_plain(view.message))
    elif view.phase == "active":
        lines.append(_plain(view.message))
    elif view.phase == "summary":
        pace = "—" if view.punches_per_minute is None else f"{view.punches_per_minute:.0f}"
        lines.extend(
            [
                f"RITMO {pace}/MIN  COMBO LARGO {view.longest_combo}",
                f"INCIERTOS {view.uncertain}   R repite   Q sale",
            ]
        )
    _panel(image, 340, 28 + len(lines) * 26)
    _write_lines(image, lines)
    if view.phase == "active" and view.elapsed_ms + view.remaining_ms > 0:
        span = view.elapsed_ms + view.remaining_ms
        width = int(image.shape[1] * view.remaining_ms / span)
        cv2.rectangle(image, (0, image.shape[0] - 10), (width, image.shape[0]), (80, 200, 120), -1)


def draw_challenge(
    image: NDArray[np.uint8],
    view: ChallengeSnapshot,
    *,
    center: tuple[int, int] | None,
    radius: int,
    fps: float | None,
    tracking: bool,
) -> None:
    if center is not None and radius > 0 and view.label_id in TARGET_STYLE:
        _title, color = TARGET_STYLE[view.label_id]
        thickness = 8 if view.flash else 3
        cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, center, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{view.title}  {view.progress}",
            (center[0] - 70, max(28, center[1] - radius - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    lines = [
        "EVOBOX AI  |  OBJETIVOS",
        f"PUNTOS {view.score}   ACIERTOS {view.hits}   FALLOS {view.misses}",
        f"COMBO {view.combo}   MEJOR {view.best_combo}",
        f"TIEMPO {_clock(view.remaining_ms)}   {view.difficulty.upper()}",
        f"{view.classifier.upper()}  |  {'SEGUIMIENTO OK' if tracking else 'SIN SEGUIMIENTO'}",
        f"FPS {0 if fps is None else round(fps)}",
    ]
    if view.phase == "idle":
        lines.append("ESPACIO PARA INICIAR")
    else:
        lines.append(_plain(view.message))
    if view.phase == "summary":
        lines.append("La puntuacion no es precision NEAT")
        lines.append("R repite   Q sale")
    _panel(image, 360, 28 + len(lines) * 26)
    _write_lines(image, lines)


def draw_menu(image: NDArray[np.uint8], stats: tuple[str, ...]) -> None:
    lines = [
        "EVOBOX AI",
        "1  BOXEO LIBRE",
        "2  GOLPEA LOS OBJETIVOS",
        "3  ENTRENAMIENTO / CAPTURA",
        "4  ESTADISTICAS",
        "Q  SALIR",
        *(_plain(item)[:36] for item in stats),
    ]
    _panel(image, 250, 20 + len(lines) * 20)
    _write_lines(image, lines, x=12, y=22, step=20, scale=0.42)


def target_screen(
    left: tuple[float, float],
    right: tuple[float, float],
    label_id: int,
    scale: float,
    width: int,
    height: int,
) -> tuple[tuple[int, int], int] | None:
    projected = project_target(
        left,
        right,
        offset_for(label_id),
        radius_for(label_id) * scale,
        width,
        height,
    )
    if projected is None:
        return None
    return (projected[0], projected[1]), projected[2]


def shown_point(
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    mirror: bool,
) -> tuple[float, float]:
    return display_point(x, y, width, height, mirror=mirror)


def class_name(label_id: int) -> str:
    return MOVEMENTS[label_id].display_name
