"""Emparejamiento temporal sin fusión geométrica entre cámaras."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyncStatus:
    valid: bool
    delta_ms: float | None
    tolerance_ms: float


def nearest_timestamp(
    primary_timestamp_ns: int,
    auxiliary_timestamp_ns: int,
    *,
    tolerance_ms: float = 100.0,
) -> SyncStatus:
    """Valida sólo cercanía temporal; no afirma correspondencia física."""

    if tolerance_ms < 0:
        raise ValueError("tolerance_ms no puede ser negativo.")
    delta_ms = abs(primary_timestamp_ns - auxiliary_timestamp_ns) / 1_000_000.0
    return SyncStatus(delta_ms <= tolerance_ms, delta_ms, tolerance_ms)
