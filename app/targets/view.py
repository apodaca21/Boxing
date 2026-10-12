"""Estado visual del entrenamiento con objetivos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TargetView:
    phase: str
    title: str
    combo_text: str
    countdown_s: int | None
    remaining_ms: int | None
    score: int
    hits: int
    misses: int
    tracking: bool
    status_line: str
    progress_lines: tuple[str, ...]
    review_lines: tuple[str, ...]
    controls: str
    target_name: str | None
    target_color: tuple[int, int, int]
    screen_xy: tuple[int, int] | None
    screen_radius: int
    fade: float
    show_ring: bool
    skeleton_points: tuple[tuple[str, float, float], ...] = ()
    skeleton_links: tuple[tuple[int, int], ...] = ()
    playback_line: str = ""


def save_button_rect(width: int, height: int) -> tuple[int, int, int, int]:
    """Rectángulo del botón GUARDAR TODO, en píxeles de la imagen."""

    return (width // 2 - 150, height - 78, width // 2 + 150, height - 36)
