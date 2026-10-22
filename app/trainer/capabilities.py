"""Clases nativas y límites de cada clasificador."""

from __future__ import annotations

from app.dataset_manager import MOVEMENTS
from app.trainer.acm_boxing import BLOCK_ACTIONS, PUNCH_ACTIONS


def classifier_capabilities(name: str) -> dict[str, object]:
    key = name.strip().lower()
    if key in {"heuristic", "neat"}:
        return {
            "classifier": key,
            "native_classes": [MOVEMENTS[index].name for index in range(5)],
            "native_display_names": [MOVEMENTS[index].display_name for index in range(5)],
            "side_policy": "left: jab/hook; right: cross/hook",
            "external_classes_are_unmapped": False,
        }
    if key == "external":
        return {
            "classifier": key,
            "native_classes": list(PUNCH_ACTIONS),
            "native_display_names": ["Jab", "Cross", "HOOK", "Uppercut"],
            "secondary_neutral_classes": list(BLOCK_ACTIONS),
            "side_policy": (
                "dominant_arm is recorded as evidence; only hook with clear arm "
                "maps to gancho izquierdo/derecho"
            ),
            "external_classes_are_unmapped": True,
        }
    raise ValueError(f"Clasificador desconocido: {name}")
