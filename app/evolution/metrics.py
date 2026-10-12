"""Métricas de clasificación y función objetivo evolutiva."""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.dataset_manager import MOVEMENTS


PUNCH_LABELS = (1, 2, 3, 4)
FITNESS_WEIGHTS = (0.65, 0.30, 0.05)


class FitnessError(ValueError):
    """El fitness no puede calcularse sin representar las clases necesarias."""


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    confusion: tuple[tuple[int, ...], ...]
    precision: dict[int, float | None]
    recall: dict[int, float | None]
    f1: dict[int, float | None]
    support: dict[int, int]
    macro_f1: float | None
    neutral_false_positive_rate: float | None


@dataclass(frozen=True, slots=True)
class FitnessBreakdown:
    punch_recall: float
    neutral_success: float
    complexity_score: float
    enabled_connections: int
    value: float


def predict_label(outputs: list[float] | tuple[float, ...]) -> int:
    if len(outputs) != len(MOVEMENTS):
        raise ValueError("La red debe producir cinco activaciones de salida.")
    if any(not math.isfinite(value) for value in outputs):
        raise ValueError("La red produjo una activación no finita.")
    best_index = 0
    best_value = outputs[0]
    for index, value in enumerate(outputs[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def confusion_matrix(
    truth: list[int],
    predicted: list[int],
) -> tuple[tuple[int, ...], ...]:
    if len(truth) != len(predicted):
        raise ValueError("Las etiquetas reales y predichas no tienen el mismo tamaño.")
    size = len(MOVEMENTS)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for actual, prediction in zip(truth, predicted):
        if actual not in MOVEMENTS or prediction not in MOVEMENTS:
            raise ValueError(f"Etiqueta fuera del rango 0-4: {(actual, prediction)}.")
        matrix[actual][prediction] += 1
    return tuple(tuple(row) for row in matrix)


def evaluate_predictions(truth: list[int], predicted: list[int]) -> ClassificationMetrics:
    matrix = confusion_matrix(truth, predicted)
    precision: dict[int, float | None] = {}
    recall: dict[int, float | None] = {}
    f1_scores: dict[int, float | None] = {}
    support: dict[int, int] = {}
    available: list[float] = []
    for label in MOVEMENTS:
        label_support = sum(matrix[label])
        predicted_count = sum(matrix[row][label] for row in MOVEMENTS)
        true_positive = matrix[label][label]
        support[label] = label_support
        precision[label] = (
            None if predicted_count == 0 else true_positive / predicted_count
        )
        recall[label] = None if label_support == 0 else true_positive / label_support
        if precision[label] is None or recall[label] is None:
            f1_scores[label] = None
        elif precision[label] + recall[label] == 0:
            f1_scores[label] = 0.0
        else:
            f1_scores[label] = (
                2 * precision[label] * recall[label] / (precision[label] + recall[label])
            )
        if f1_scores[label] is not None:
            available.append(f1_scores[label])

    non_neutral = sum(support[label] for label in PUNCH_LABELS)
    false_neutral = sum(matrix[label][0] for label in PUNCH_LABELS)
    neutral_fpr = None if non_neutral == 0 else false_neutral / non_neutral
    macro = None if not available else sum(available) / len(available)
    return ClassificationMetrics(
        confusion=matrix,
        precision=precision,
        recall=recall,
        f1=f1_scores,
        support=support,
        macro_f1=macro,
        neutral_false_positive_rate=neutral_fpr,
    )


def fitness_from_predictions(
    truth: list[int],
    predicted: list[int],
    enabled_connections: int,
) -> FitnessBreakdown:
    """Calcula F(g) solo cuando todas las clases requeridas están presentes."""

    if enabled_connections < 0:
        raise FitnessError("El número de conexiones no puede ser negativo.")
    metrics = evaluate_predictions(truth, predicted)
    missing = [
        MOVEMENTS[label].display_name
        for label in MOVEMENTS
        if metrics.support[label] < 1
    ]
    if missing:
        raise FitnessError(
            "No se evalúa el fitness porque el entrenamiento no representa: "
            + ", ".join(missing)
            + "."
        )
    punch_recall = sum(metrics.recall[label] or 0.0 for label in PUNCH_LABELS) / len(PUNCH_LABELS)
    neutral_success = metrics.recall[0] or 0.0
    complexity = 1.0 / (1.0 + enabled_connections)
    value = (
        FITNESS_WEIGHTS[0] * punch_recall
        + FITNESS_WEIGHTS[1] * neutral_success
        + FITNESS_WEIGHTS[2] * complexity
    )
    return FitnessBreakdown(
        punch_recall=punch_recall,
        neutral_success=neutral_success,
        complexity_score=complexity,
        enabled_connections=enabled_connections,
        value=value,
    )
