"""Recorta la ventana biomecánica de un objetivo."""

from __future__ import annotations

from app.dataset_manager import FrameRecord


def window_from_frames(
    frames: tuple[FrameRecord, ...] | list[FrameRecord],
    start_ms: int,
    duration_ms: int,
    *,
    min_frames: int,
    min_duration_ms: int,
    max_duration_ms: int,
) -> tuple[tuple[FrameRecord, ...], tuple[str, ...]]:
    """Selecciona fotogramas reales. No reescribe timestamps ni rellena huecos."""

    selected = tuple(
        frame
        for frame in frames
        if start_ms <= frame.timestamp_ms <= start_ms + duration_ms
    )
    if len(selected) < 2:
        return (), ("La ventana no tiene suficientes fotogramas.",)
    measured = selected[-1].timestamp_ms - selected[0].timestamp_ms
    reasons: list[str] = []
    if measured < min_duration_ms or measured > max_duration_ms:
        reasons.append("La duración real está fuera del rango permitido de 1.5 a 2.8 segundos.")
    if len(selected) < min_frames:
        reasons.append(
            f"La captura solo contiene {len(selected)} fotogramas; se requieren al menos {min_frames}."
        )
    return selected, tuple(reasons)
