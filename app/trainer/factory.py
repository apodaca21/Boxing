"""Selecciona un clasificador sin cambiar el contrato que consume el entrenador."""

from __future__ import annotations

from pathlib import Path

from app.evolution.model_store import load_model
from app.trainer.external import ExternalClassifier, load_external_classifier
from app.trainer.heuristic import HeuristicClassifier
from app.trainer.neat_classifier import NeatClassifier, resolve_neat_model
from app.trainer.sensitivity import profile_settings


CLASSIFIER_NAMES = ("heuristic", "neat", "external")


def create_classifier(
    kind: str,
    *,
    neat_model: Path | None = None,
    results_dir: Path | None = None,
    external_dir: Path | None = None,
    sensitivity: str = "normal",
) -> HeuristicClassifier | NeatClassifier | ExternalClassifier:
    if kind == "heuristic":
        return HeuristicClassifier(profile_settings(sensitivity))
    if kind == "neat":
        if results_dir is None:
            raise ValueError("El clasificador NEAT necesita un directorio de resultados.")
        model_dir = resolve_neat_model(neat_model, results_dir)
        return NeatClassifier.from_model(load_model(model_dir))
    if kind == "external":
        return load_external_classifier(external_dir)
    raise ValueError(
        "El clasificador debe ser 'heuristic', 'neat' o 'external'."
    )
