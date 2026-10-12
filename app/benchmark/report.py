"""Informes de evaluación. No declaran un ganador por contar más golpes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.benchmark.matching import EvaluationMetrics, evaluate_repetitions
from app.benchmark.protocol import BenchmarkProtocol, RepetitionRecord
from app.dataset_manager import MOVEMENTS


COMPARISON_NOTE = (
    "Un perfil que registra más movimientos no es mejor por eso. "
    "La comparación usa solo repeticiones verificadas por el usuario."
)
GROUND_TRUTH_NOTE = (
    "La consigna en pantalla es la etiqueta prevista. "
    "La métrica usa la etiqueta confirmada o corregida, emparejada una sola vez."
)


def write_session_report(
    protocol: BenchmarkProtocol,
    directory: Path,
    *,
    classifier_name: str,
    sensitivity: str,
    tolerance_ms: int,
    saved_sample_ids: tuple[str, ...] = (),
    sample_rejections: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> EvaluationMetrics:
    directory.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_repetitions(protocol.records, tolerance_ms=tolerance_ms)
    payload = {
        "ground_truth": "user_verified",
        "synthetic": False,
        "note": GROUND_TRUTH_NOTE,
        "classifier": classifier_name,
        "sensitivity": sensitivity,
        "match_tolerance_ms": tolerance_ms,
        "verified_repetitions": metrics.verified_repetitions,
        "excluded_repetitions": metrics.excluded_repetitions,
        "duplicates": metrics.duplicates,
        "incorrect_detections": metrics.incorrect,
        "neutral_false_positives": metrics.neutral_false_positives,
        "neutral_verified": metrics.neutral_verified,
        "neutral_clean": metrics.neutral_clean,
        "unmapped_predictions": metrics.unmapped_predictions,
        "mean_latency_ms": metrics.mean_latency_ms,
        "latency_samples": metrics.latency_samples,
        "mean_fps": metrics.mean_fps,
        "saved_sample_ids": list(saved_sample_ids),
        "sample_rejections": [
            {"requested": name, "reasons": list(reasons)}
            for name, reasons in sample_rejections
        ],
        "classes": [_class_payload(metrics, label) for label in range(5)],
        "confusion": [list(row) for row in metrics.confusion],
        "confusion_columns": [MOVEMENTS[label].name for label in range(5)] + ["sin_deteccion"],
        "repetitions": [_repetition_payload(record) for record in protocol.records],
    }
    (directory / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_metrics_csv(directory / "metrics.csv", metrics)
    _write_events_csv(directory / "events.csv", protocol.records)
    _write_confusion_csv(directory / "confusion.csv", metrics)
    plot_path = directory / "confusion.png"
    if _enough_for_plot(metrics):
        _write_plot(metrics, plot_path, sensitivity)
        payload["plot_written"] = True
    else:
        payload["plot_written"] = False
        payload["plot_reason"] = (
            "Faltan repeticiones verificadas de alguna clase. No se dibujó la matriz."
        )
    (directory / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "summary.txt").write_text(
        summary_text(metrics, sensitivity, classifier_name),
        encoding="utf-8",
    )
    return metrics


def summary_text(
    metrics: EvaluationMetrics,
    sensitivity: str,
    classifier_name: str,
) -> str:
    lines = [
        f"Evaluación {classifier_name} / {sensitivity}",
        GROUND_TRUTH_NOTE,
        (
            f"Verificadas: {metrics.verified_repetitions}. "
            f"Excluidas: {metrics.excluded_repetitions}. "
            f"Duplicados: {metrics.duplicates}. "
            f"Incorrectas: {metrics.incorrect}. "
            f"Falsos positivos en guardia: {metrics.neutral_false_positives}."
        ),
    ]
    if metrics.mean_latency_ms is None:
        lines.append("Latencia de inferencia: sin muestras en aciertos emparejados.")
    else:
        lines.append(
            f"Latencia media de inferencia en aciertos: {metrics.mean_latency_ms:.1f} ms "
            f"({metrics.latency_samples} muestras)."
        )
    if metrics.mean_fps is None:
        lines.append("FPS medio del intervalo: sin muestras.")
    else:
        lines.append(f"FPS medio del intervalo: {metrics.mean_fps:.1f}.")
    for label in range(5):
        score = metrics.scores[label]
        lines.append(
            f"{MOVEMENTS[label].display_name}: "
            f"recall {_format(score.recall)}, "
            f"precision {_format(score.precision)}, "
            f"f1 {_format(score.f1)}, "
            f"soporte {score.support}, "
            f"duplicados {score.duplicates}"
        )
    lines.append("Una predicción no se asigna a dos golpes.")
    return "\n".join(lines) + "\n"


def update_comparison(benchmarks_root: Path) -> Path | None:
    """Junta informes reales ya guardados. No elige un perfil ganador."""

    rows = []
    if not benchmarks_root.is_dir():
        return None
    for path in sorted(benchmarks_root.glob("*/*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("synthetic") or payload.get("ground_truth") != "user_verified":
            continue
        if int(payload.get("verified_repetitions", 0)) <= 0:
            continue
        rows.append(
            {
                "sensitivity": payload.get("sensitivity"),
                "classifier": payload.get("classifier"),
                "verified_repetitions": payload.get("verified_repetitions"),
                "duplicates": payload.get("duplicates"),
                "incorrect_detections": payload.get("incorrect_detections"),
                "neutral_false_positives": payload.get("neutral_false_positives"),
                "mean_latency_ms": payload.get("mean_latency_ms"),
                "mean_fps": payload.get("mean_fps"),
                "macro_recall": _macro(payload, "recall"),
                "macro_precision": _macro(payload, "precision"),
                "report": path.parent.as_posix(),
            }
        )
    if not rows:
        return None
    destination = benchmarks_root / "comparison.json"
    destination.write_text(
        json.dumps(
            {"note": COMPARISON_NOTE, "winner": None, "profiles": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = benchmarks_root / "comparison.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _class_payload(metrics: EvaluationMetrics, label: int) -> dict[str, object]:
    score = metrics.scores[label]
    return {
        "label_id": label,
        "name": MOVEMENTS[label].name,
        "support": score.support,
        "true_positives": score.true_positives,
        "false_positives": score.false_positives,
        "false_negatives": score.false_negatives,
        "duplicates": score.duplicates,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
    }


def _repetition_payload(record: RepetitionRecord) -> dict[str, object]:
    return {
        "requested_label_id": record.requested_label_id,
        "requested_name": record.requested_name,
        "repetition_index": record.repetition_index,
        "window_start_ms": record.window_start_ms,
        "window_end_ms": record.window_end_ms,
        "status": record.status,
        "verified_label_id": record.verified_label_id,
        "verified_name": record.verified_name,
        "corrected": record.corrected,
        "prediction_count": len(record.predictions),
        "predictions": [
            {
                "timestamp_ms": mark.timestamp_ms,
                "label_id": mark.label_id,
                "display_name": mark.display_name,
                "latency_ms": mark.latency_ms,
            }
            for mark in record.predictions
        ],
        "fps_samples": len(record.fps_samples),
        "mean_fps": (
            None
            if not record.fps_samples
            else sum(record.fps_samples) / len(record.fps_samples)
        ),
    }


def _write_metrics_csv(path: Path, metrics: EvaluationMetrics) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "clase",
                "soporte",
                "precision",
                "recall",
                "f1",
                "verdaderos_positivos",
                "falsos_positivos",
                "falsos_negativos",
                "duplicados",
            )
        )
        for label in range(5):
            score = metrics.scores[label]
            writer.writerow(
                (
                    MOVEMENTS[label].name,
                    score.support,
                    _format(score.precision),
                    _format(score.recall),
                    _format(score.f1),
                    score.true_positives,
                    score.false_positives,
                    score.false_negatives,
                    score.duplicates,
                )
            )


def _write_events_csv(path: Path, records: list[RepetitionRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "solicitado",
                "repeticion",
                "inicio_ms",
                "fin_ms",
                "estado",
                "verificado",
                "corregido",
                "timestamp_prediccion_ms",
                "clase_predicha",
                "latencia_ms",
            )
        )
        for record in records:
            if not record.predictions:
                writer.writerow(
                    (
                        record.requested_name,
                        record.repetition_index,
                        record.window_start_ms,
                        record.window_end_ms,
                        record.status,
                        record.verified_name or "",
                        record.corrected,
                        "",
                        "",
                        "",
                    )
                )
                continue
            for mark in record.predictions:
                writer.writerow(
                    (
                        record.requested_name,
                        record.repetition_index,
                        record.window_start_ms,
                        record.window_end_ms,
                        record.status,
                        record.verified_name or "",
                        record.corrected,
                        mark.timestamp_ms,
                        mark.display_name,
                        "" if mark.latency_ms is None else f"{mark.latency_ms:.2f}",
                    )
                )


def _write_confusion_csv(path: Path, metrics: EvaluationMetrics) -> None:
    headers = [MOVEMENTS[label].name for label in range(5)] + ["sin_deteccion"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["verdadero", *headers])
        for label, row in enumerate(metrics.confusion):
            writer.writerow([MOVEMENTS[label].name, *row])


def _enough_for_plot(metrics: EvaluationMetrics) -> bool:
    return all(metrics.scores[label].support > 0 for label in range(5))


def _write_plot(metrics: EvaluationMetrics, path: Path, sensitivity: str) -> None:
    matrix = [list(row) for row in metrics.confusion]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    image = axis.imshow(matrix)
    columns = [MOVEMENTS[label].display_name for label in range(5)] + ["Sin deteccion"]
    rows = [MOVEMENTS[label].display_name for label in range(5)]
    axis.set_xticks(range(len(columns)), labels=columns, rotation=30, ha="right")
    axis.set_yticks(range(len(rows)), labels=rows)
    axis.set_xlabel("Predicción emparejada")
    axis.set_ylabel("Etiqueta verificada")
    axis.set_title(f"Evaluación verificada · {sensitivity}")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _macro(payload: dict[str, object], field: str) -> float | None:
    classes = payload.get("classes")
    if not isinstance(classes, list):
        return None
    values = [
        float(item[field])
        for item in classes
        if isinstance(item, dict) and item.get("support", 0) and item.get(field) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _format(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"
