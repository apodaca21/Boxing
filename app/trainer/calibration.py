"""Calibración local de umbrales. No crea muestras para NEAT."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from app.config import PROJECT_ROOT
from app.trainer.heuristic import HeuristicSettings


DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "results" / "sparring_calibration.json"
PHASES = (
    (0, 3000, "Permanece en guardia"),
    (3000, 6000, "Mueve los brazos suavemente"),
    (6000, 10000, "Lanza varios golpes rectos"),
    (10000, 14000, "Lanza varios ganchos"),
)


class MotionCalibrator:
    """Estima escalas de movimiento durante unos 14 segundos."""

    def __init__(self, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
        self.path = path
        self.active = False
        self._started_ms: int | None = None
        self._guard: list[float] = []
        self._soft_speed: list[float] = []
        self._straight_gain: list[float] = []
        self._hook_lateral: list[float] = []

    def start(self, timestamp_ms: int) -> None:
        self.active = True
        self._started_ms = timestamp_ms
        self._guard.clear()
        self._soft_speed.clear()
        self._straight_gain.clear()
        self._hook_lateral.clear()

    def prompt(self, timestamp_ms: int) -> str:
        if self._started_ms is None:
            return "Calibracion"
        elapsed = timestamp_ms - self._started_ms
        for start, end, text in PHASES:
            if start <= elapsed < end:
                return text
        return "Calibracion lista"

    def finished(self, timestamp_ms: int) -> bool:
        return self._started_ms is not None and timestamp_ms - self._started_ms >= 14000

    def observe(self, timestamp_ms: int, telemetry: dict[str, float | str | None]) -> None:
        if not self.active or self._started_ms is None:
            return
        elapsed = timestamp_ms - self._started_ms
        extensions = [
            value
            for key in ("left_extension", "right_extension")
            if isinstance((value := telemetry.get(key)), float)
        ]
        speeds = [
            value
            for key in ("left_speed", "right_speed")
            if isinstance((value := telemetry.get(key)), float)
        ]
        laterals = [
            value
            for key in ("left_lateral", "right_lateral")
            if isinstance((value := telemetry.get(key)), float)
        ]
        if elapsed < 3000:
            self._guard.extend(extensions)
            return
        guard = _median(self._guard)
        if 3000 <= elapsed < 6000:
            self._soft_speed.extend(speeds)
        elif 6000 <= elapsed < 10000 and guard is not None:
            self._straight_gain.extend(max(0.0, value - guard) for value in extensions)
        elif 10000 <= elapsed < 14000:
            self._hook_lateral.extend(laterals)

    def propose(self, current: HeuristicSettings) -> HeuristicSettings:
        """Propone umbrales por debajo de los golpes observados y por encima del movimiento suave."""

        soft = _percentile(self._soft_speed, 0.90)
        straight = _percentile(self._straight_gain, 0.50)
        hook = _percentile(self._hook_lateral, 0.50)
        speed = current.min_wrist_speed if soft is None else max(0.05, soft * 1.25)
        gain = current.straight_gain if straight is None else max(0.12, straight * 0.75)
        lateral = current.hook_lateral if hook is None else max(0.16, hook * 0.75)
        return replace(
            current,
            min_wrist_speed=speed,
            min_extension_rate=max(0.15, speed * 2.0),
            straight_gain=gain,
            start_extension=max(0.04, gain * 0.45),
            commit_extension=max(0.08, gain * 0.75),
            hook_lateral=lateral,
            lateral_start=max(0.06, lateral * 0.45),
            lateral_commit=max(0.10, lateral * 0.70),
            allow_early_confirm=True,
            profile_name="calibrado",
        )

    def save(self, settings: HeuristicSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "purpose": "Umbrales del detector heuristico. No son etiquetas para entrenar NEAT.",
            "contains_video": False,
            "contains_images": False,
            "profile_name": settings.profile_name,
            "min_wrist_speed": settings.min_wrist_speed,
            "straight_gain": settings.straight_gain,
            "hook_lateral": settings.hook_lateral,
            "start_extension": settings.start_extension,
            "commit_extension": settings.commit_extension,
            "cooldown_ms": settings.cooldown_ms,
            "allow_early_confirm": settings.allow_early_confirm,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.active = False


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]
