"""Contratos seguros para auditar y convertir datos externos de boxeo."""

from app.external_data.conversion import (
    CANONICAL_LABELS,
    ConversionResult,
    convert_external_label,
)

__all__ = [
    "CANONICAL_LABELS",
    "ConversionResult",
    "convert_external_label",
]
