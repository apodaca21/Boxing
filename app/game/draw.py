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


def _panel(image: NDArray[np.uint8], width: int) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, image.shape[0]), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)


def draw_free(
    image: NDArray[np.uint8],
    view: FreeSnapshot,
    *,
    fps: float | None,
    tracking: bool,
) -> None:
    _panel(image, 300)
    lines = [
        "EVOBOX AI",
        "MODO LIBRE",
        f"TIEMPO {_clock(view.remaining_ms if view.phase == 'active' else view.elapsed_ms)}",
        f"JAB {view.counts[1]}",
        f"CROSS {view.counts[2]}",
        f"GANCHO IZQUIERDO {view.counts[3]}",
        f"GANCHO DERECHO {view.counts[4]}",
        f"TOTAL {view.total} GOLPES",
        f"ULTIMO {_plain(view.last_punch)}",
        f"COMBO {_plain(view.combo)}",
        f"CLASIFICADOR {view.classifier.upper()}",
        "SEGUIMIENTO OK" if tracking else "SIN SEGUIMIENTO",
        _plain(view.message),
        f"FPS {0 if fps is None else round(fps)}",
    ]
    if view.phase == "summary":
        pace = "—" if view.punches_per_minute is None else f"{view.punches_per_minute:.0f}"
        lines.extend(
            [
                f"RITMO {pace} / MIN",
                f"COMBO LARGO {view.longest_combo}",
                f"INCIERTOS {view.uncertain}",
                "R repite   Q sale",
            ]
        )
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line[:34],
            (16, 36 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 248, 252),
            1,
            cv2.LINE_AA,
        )
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
    _panel(image, 280)
    lines = [
        "EVOBOX AI",
        "BOXING TARGET CHALLENGE",
        f"PUNTOS {view.score}",
        f"ACIERTOS {view.hits}",
        f"FALLOS {view.misses}",
        f"COMBO {view.combo}",
        f"MEJOR {view.best_combo}",
        f"TIEMPO {_clock(view.remaining_ms)}",
        f"NIVEL {view.difficulty.upper()}",
        f"CLASIFICADOR {view.classifier.upper()}",
        "SEGUIMIENTO OK" if tracking else "SIN SEGUIMIENTO",
        _plain(view.message),
        f"FPS {0 if fps is None else round(fps)}",
    ]
    if view.phase == "idle":
        lines.insert(2, "ESPACIO PARA INICIAR")
    if view.phase == "summary":
        lines.append("La puntuacion no es precision NEAT")
        lines.append("R repite   Q sale")
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line[:36],
            (14, 32 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 248, 252),
            1,
            cv2.LINE_AA,
        )


def draw_menu(image: NDArray[np.uint8], stats: tuple[str, ...]) -> None:
    shade = image.copy()
    cv2.rectangle(shade, (0, 0), (image.shape[1], image.shape[0]), (8, 10, 16), -1)
    cv2.addWeighted(shade, 0.55, image, 0.45, 0, image)
    lines = [
        "EVOBOX AI",
        "1  BOXEO LIBRE",
        "2  GOLPEA LOS OBJETIVOS",
        "3  ENTRENAMIENTO / CAPTURA",
        "4  ESTADISTICAS",
        "Q  SALIR",
        *stats,
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            _plain(line)[:48],
            (48, 70 + index * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 248, 252),
            2,
            cv2.LINE_AA,
        )


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
