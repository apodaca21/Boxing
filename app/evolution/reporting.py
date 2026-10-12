"""Gráficas y tablas generadas solamente con métricas calculadas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.dataset_manager import MOVEMENTS
from app.evolution.metrics import ClassificationMetrics
from app.evolution.neat_trainer import GenerationRecord


def write_history_plots(history: tuple[GenerationRecord, ...], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    generations = [record.generation for record in history]
    _line_plot(
        generations,
        [record.best_fitness for record in history],
        directory / "fitness_mejor.png",
        "Mejor fitness de entrenamiento",
        "Fitness",
    )
    _line_plot(
        generations,
        [record.mean_fitness for record in history],
        directory / "fitness_promedio.png",
        "Fitness promedio de entrenamiento",
        "Fitness",
    )
    _line_plot(
        generations,
        [record.best_enabled_connections for record in history],
        directory / "conexiones_mejor.png",
        "Conexiones habilitadas del mejor individuo",
        "Conexiones",
    )


def write_confusion_plot(
    metrics: ClassificationMetrics,
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = [list(row) for row in metrics.confusion]
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix)
    labels = [MOVEMENTS[label].display_name for label in MOVEMENTS]
    axis.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicción")
    axis.set_ylabel("Etiqueta real")
    axis.set_title(title)
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_metrics_table(metrics_by_split: dict[str, ClassificationMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "conjunto,clase,precision,recall,f1,soporte,macro_f1,falsos_positivos_neutral"
    ]
    for split_name, metrics in metrics_by_split.items():
        for label, movement in MOVEMENTS.items():
            lines.append(
                ",".join(
                    (
                        split_name,
                        movement.name,
                        _format(metrics.precision[label]),
                        _format(metrics.recall[label]),
                        _format(metrics.f1[label]),
                        str(metrics.support[label]),
                        _format(metrics.macro_f1),
                        _format(metrics.neutral_false_positive_rate),
                    )
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _line_plot(
    x_values: list[int],
    y_values: list[float],
    path: Path,
    title: str,
    y_label: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x_values, y_values, marker="o")
    axis.set_title(title)
    axis.set_xlabel("Generación")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _format(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"
