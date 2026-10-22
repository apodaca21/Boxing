"""Perfiles iniciales del detector provisional.

NORMAL conserva los umbrales con los que se validó el regreso completo a la
guardia. HIGH y MAXIMUM parten de esa misma escala y la reducen para que un
golpe corto pueda confirmarse al frenar, sin exigir la postura inicial.

No son una calibración con personas reales. La rutina de calibración puede
sustituirlos por una propuesta a partir del movimiento de la sesión.
"""

from __future__ import annotations

from app.trainer.heuristic import HeuristicSettings


PROFILE_NAMES = ("normal", "high", "maximum", "play")

# Alta: pensada para combinaciones. El desplazamiento de confirmación baja
# de 0.40 a 0.18 anchos de hombro y el cooldown por brazo de 280 ms a 80 ms.
# Así un jab y un cross alternados no se bloquean entre sí, y un segundo golpe
# del mismo brazo puede empezar en cuanto ese brazo frena.
_HIGH = HeuristicSettings(
    smooth_alpha=0.65,
    baseline_alpha=0.20,
    start_extension=0.08,
    commit_extension=0.16,
    straight_gain=0.18,
    straight_elbow=100.0,
    hook_lateral=0.28,
    hook_elbow_max=155.0,
    lateral_start=0.12,
    lateral_commit=0.22,
    retract_hysteresis=0.06,
    lateral_hysteresis=0.06,
    return_margin=0.35,
    return_lateral=0.35,
    min_wrist_speed=0.12,
    min_extension_rate=0.35,
    min_cycle_ms=40,
    max_cycle_ms=800,
    start_timeout_ms=320,
    cooldown_ms=80,
    max_missing_frames=8,
    allow_early_confirm=True,
    profile_name="high",
)

# Máxima: experimental. Favorece candidatos y puede contar balanceos amplios.
_MAXIMUM = HeuristicSettings(
    smooth_alpha=0.8,
    baseline_alpha=0.25,
    start_extension=0.05,
    commit_extension=0.10,
    straight_gain=0.12,
    straight_elbow=85.0,
    hook_lateral=0.18,
    hook_elbow_max=165.0,
    lateral_start=0.08,
    lateral_commit=0.14,
    retract_hysteresis=0.04,
    lateral_hysteresis=0.04,
    return_margin=0.45,
    return_lateral=0.45,
    min_wrist_speed=0.08,
    min_extension_rate=0.20,
    min_cycle_ms=30,
    max_cycle_ms=700,
    start_timeout_ms=280,
    cooldown_ms=50,
    max_missing_frames=10,
    allow_early_confirm=True,
    profile_name="maximum",
)


# Juego: confirma al frenar el golpe, sin exigir volver del todo a guardia.
_PLAY = HeuristicSettings(
    smooth_alpha=0.85,
    baseline_alpha=0.28,
    start_extension=0.04,
    commit_extension=0.08,
    straight_gain=0.08,
    straight_elbow=75.0,
    hook_lateral=0.12,
    hook_elbow_max=170.0,
    lateral_start=0.06,
    lateral_commit=0.10,
    retract_hysteresis=0.03,
    lateral_hysteresis=0.03,
    return_margin=0.50,
    return_lateral=0.50,
    min_wrist_speed=0.05,
    min_extension_rate=0.12,
    min_cycle_ms=25,
    max_cycle_ms=900,
    start_timeout_ms=350,
    cooldown_ms=35,
    max_missing_frames=12,
    allow_early_confirm=True,
    confirm_on_slowdown=True,
    profile_name="play",
)


def profile_settings(name: str) -> HeuristicSettings:
    """Devuelve una copia lógica del perfil. NORMAL es el comportamiento anterior."""

    key = name.strip().lower()
    if key == "normal":
        return HeuristicSettings()
    if key == "high":
        return _HIGH
    if key == "maximum":
        return _MAXIMUM
    if key == "play":
        return _PLAY
    raise ValueError("La sensibilidad debe ser normal, high, maximum o play.")


def sensitivity_label(name: str) -> str:
    labels = {
        "normal": "NORMAL",
        "high": "HIGH",
        "maximum": "MAXIMUM",
        "play": "PLAY",
        "calibrado": "CALIBRADO",
    }
    return labels.get(name, name.upper())
