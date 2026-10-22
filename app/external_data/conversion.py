"""Conversión explícita de etiquetas externas a las cinco clases de EvoBox.

Este módulo solo convierte etiquetas cuando la fuente aporta evidencia suficiente.
No convierte ``HOOK`` genérico, ``straight``, ``uppercut`` ni ``none`` por
proximidad semántica. La lateralidad anatómica es distinta de la posición visual
en la imagen.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


CANONICAL_LABELS = {
    0: "guardia_neutral",
    1: "jab_izquierdo",
    2: "cross_derecho",
    3: "gancho_izquierdo",
    4: "gancho_derecho",
}


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Resultado auditable de una conversión de etiqueta."""

    source_label: str
    canonical_label_id: int | None
    canonical_label: str | None
    status: str
    reason: str


def convert_external_label(
    source_label: str,
    *,
    dataset: str,
    stance: str | None = None,
    arm: str | None = None,
    arm_evidence: bool = False,
) -> ConversionResult:
    """Convierte una etiqueta externa sin inferir clases ausentes.

    ``arm`` solo es evidencia válida cuando la fuente o una medición biomecánica
    independiente la marca explícitamente (``arm_evidence=True``). Para
    ``lead/rear`` se exige postura orthodox; no se deduce desde el lado de la
    pantalla.
    """

    raw = str(source_label).strip()
    normalized = re.sub(r"[_-]+", " ", raw.lower())
    normalized = re.sub(r"\s+", " ", normalized)
    source = f"{dataset}:{raw}"

    if dataset.strip().lower() == "punch_dl" and normalized in {"0", "5", "6"}:
        reason = "no punch no demuestra guardia" if normalized == "0" else "uppercut excluido"
        return _excluded(source, reason)
    if dataset.strip().lower() == "punch_dl" and normalized in {"1", "2", "3", "4"}:
        return _mapped(source, int(normalized), "id de clase Punch_DL con lateralidad explícita")

    explicit = {
        "guard": 0,
        "neutral": 0,
        "jab left": 1,
        "left jab": 1,
        "jab izquierdo": 1,
        "cross right": 2,
        "right cross": 2,
        "cross derecho": 2,
        "left hook": 3,
        "hook left": 3,
        "gancho izquierdo": 3,
        "right hook": 4,
        "hook right": 4,
        "gancho derecho": 4,
    }
    if normalized in explicit:
        label_id = explicit[normalized]
        return _mapped(source, label_id, "lateralidad explícita en la fuente")

    if normalized in {"none", "no punch", "no_punch", "no golpe"}:
        return _excluded(source, "no golpe no demuestra guardia/neutral")
    if "uppercut" in normalized or "upper" in normalized:
        return _excluded(source, "uppercut no es una clase de gancho")
    if normalized in {"hook", "generic hook", "hook generic"}:
        if arm_evidence and arm in {"left", "right"}:
            label_id = 3 if arm == "left" else 4
            return _mapped(source, label_id, "HOOK con brazo confirmado independientemente")
        return _excluded(source, "HOOK genérico no aporta lado anatómico")
    if normalized in {"straight", "straight punch", "punch"}:
        return _excluded(source, "straight no distingue jab de cross")
    if normalized in {"block", "forearm block", "high guard", "parry"}:
        return _excluded(source, "bloqueo no se convierte en guardia/neutral")

    lead_rear = {
        "lead hook": 3,
        "rear hook": 4,
        "lead gancho": 3,
        "rear gancho": 4,
    }
    if normalized in lead_rear:
        if stance != "orthodox":
            return _excluded(source, "lead/rear requiere postura orthodox declarada")
        return _mapped(source, lead_rear[normalized], "lead/rear convertido con orthodox declarado")

    if normalized in {"jab", "cross"}:
        if not arm_evidence or arm not in {"left", "right"}:
            return _excluded(source, "jab/cross genérico sin evidencia de brazo")
        label_id = 1 if normalized == "jab" and arm == "left" else None
        if normalized == "cross" and arm == "right":
            label_id = 2
        if label_id is None:
            return _excluded(source, "brazo observado no coincide con orthodox solicitado")
        return _mapped(source, label_id, "clase y brazo confirmados por evidencia independiente")

    return _excluded(source, "etiqueta no reconocida o ambigua")


def _mapped(source: str, label_id: int, reason: str) -> ConversionResult:
    return ConversionResult(
        source_label=source,
        canonical_label_id=label_id,
        canonical_label=CANONICAL_LABELS[label_id],
        status="mapped",
        reason=reason,
    )


def _excluded(source: str, reason: str) -> ConversionResult:
    return ConversionResult(
        source_label=source,
        canonical_label_id=None,
        canonical_label=None,
        status="excluded",
        reason=reason,
    )
