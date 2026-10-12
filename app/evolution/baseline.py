"""Línea base opcional de topología fija con scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from app.evolution.metrics import ClassificationMetrics, evaluate_predictions
from app.evolution.neat_trainer import neat_parameter_count


@dataclass(frozen=True, slots=True)
class ModelComparison:
    name: str
    macro_f1: float | None
    neutral_false_positive_rate: float | None
    parameter_count: int
    inference_seconds: float
    metrics: ClassificationMetrics


def compare_models(
    neat_predictions: list[int],
    neat_genome: object,
    truth: list[int],
    train_vectors: list[tuple[float, ...]],
    train_labels: list[int],
    test_vectors: list[tuple[float, ...]],
    seed: int,
) -> tuple[ModelComparison, ModelComparison]:
    neat_started = time.perf_counter()
    neat_metrics = evaluate_predictions(truth, neat_predictions)
    neat_seconds = time.perf_counter() - neat_started
    neat_result = ModelComparison(
        name="neat",
        macro_f1=neat_metrics.macro_f1,
        neutral_false_positive_rate=neat_metrics.neutral_false_positive_rate,
        parameter_count=neat_parameter_count(neat_genome),
        inference_seconds=neat_seconds,
        metrics=neat_metrics,
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(np.asarray(train_vectors, dtype=float), np.asarray(train_labels))
    test_array = np.asarray(test_vectors, dtype=float)
    started = time.perf_counter()
    predicted = [int(label) for label in classifier.predict(test_array)]
    baseline_seconds = time.perf_counter() - started
    baseline_metrics = evaluate_predictions(truth, predicted)
    parameter_count = int(
        sum(coefficient.size for coefficient in classifier.coefs_)
        + sum(intercept.size for intercept in classifier.intercepts_)
    )
    baseline = ModelComparison(
        name="sklearn_mlp_fixed",
        macro_f1=baseline_metrics.macro_f1,
        neutral_false_positive_rate=baseline_metrics.neutral_false_positive_rate,
        parameter_count=parameter_count,
        inference_seconds=baseline_seconds,
        metrics=baseline_metrics,
    )
    return neat_result, baseline
