"""Informes de evaluación. No declaran un ganador por contar más golpes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.benchmark.matching import EvaluationMetrics, evaluate_repetitions
from app.benchmark.protocol import BenchmarkProtocol, PredictionMark, RepetitionRecord
from app.dataset_manager import MOVEMENTS
from app.trainer.capabilities import classifier_capabilities


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
        "classifier_capabilities": classifier_capabilities(classifier_name),
        "sensitivity": sensitivity,
        "match_tolerance_ms": tolerance_ms,
        "verified_repetitions": metrics.verified_repetitions,
        "excluded_repetitions": metrics.excluded_repetitions,
        "repeated_repetitions": sum(record.status == "repeated" for record in protocol.records),
        "counter_events_seen": _counter_totals(protocol.records)[0],
        "counter_events_ignored": _counter_totals(protocol.records)[1],
        "duplicates": metrics.duplicates,
        "incorrect_detections": metrics.incorrect,
        "neutral_false_positives": metrics.neutral_false_positives,
        "neutral_verified": metrics.neutral_verified,
        "neutral_clean": metrics.neutral_clean,
        "unmapped_predictions": metrics.unmapped_predictions,
        "mean_latency_ms": metrics.mean_latency_ms,
        "latency_samples": metrics.latency_samples,
        "mean_fps": metrics.mean_fps,
        "recognition_metrics": {
            "classes": [_class_payload(metrics, label) for label in range(5)],
            "confusion": [list(row) for row in metrics.confusion],
            "unmapped_predictions": metrics.unmapped_predictions,
            "incorrect_detections": metrics.incorrect,
            "neutral_false_positives": metrics.neutral_false_positives,
        },
        "counter_metrics": {
            "events_seen": _counter_totals(protocol.records)[0],
            "events_ignored": _counter_totals(protocol.records)[1],
            "duplicates": metrics.duplicates,
            "policy": "un evento confirmado se suma una vez; no se reasigna a dos intentos",
        },
        "runtime_metrics": {
            "mean_fps": metrics.mean_fps,
            "mean_latency_ms": metrics.mean_latency_ms,
            "latency_samples": metrics.latency_samples,
        },
        "reconciliation": reconciliation_metrics(metrics, protocol.records),
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
    _write_attempts_csv(directory / "attempts.csv", protocol.records, classifier_name)
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
        summary_text(metrics, sensitivity, classifier_name, records=protocol.records),
        encoding="utf-8",
    )
    write_reconciliation_files(directory, metrics, protocol.records)
    return metrics


def reconciliation_metrics(
    metrics: EvaluationMetrics,
    records: list[RepetitionRecord] | tuple[RepetitionRecord, ...],
) -> dict[str, object]:
    """Separa golpes, clases erróneas, ausencia de detección y contador.

    La matriz usa una sola fila por intento verificado. En particular, una
    clasificación equivocada no se suma a ``sin_deteccion`` y un falso
    positivo en guardia no se suma a una clase de golpe.
    """

    matrix = metrics.confusion
    punch_hits = sum(matrix[label][label] for label in range(1, 5))
    punch_wrong = sum(
        matrix[truth][predicted]
        for truth in range(1, 5)
        for predicted in range(1, 5)
        if predicted != truth
    )
    punch_no_detection = sum(matrix[label][5] for label in range(1, 5))
    verified = tuple(record for record in records if record.status == "verified")
    punch_unmapped = sum(
        mark.label_id is None
        for record in verified
        if record.verified_label_id in (1, 2, 3, 4)
        for mark in record.predictions
    )
    additional = sum(max(0, len(record.predictions) - 1) for record in verified)
    return {
        "verified_punch_attempts": sum(record.verified_label_id in (1, 2, 3, 4) for record in verified),
        "punch_hits": punch_hits,
        "punch_wrong_classifications": punch_wrong,
        "punch_no_detection": punch_no_detection,
        "punch_unmapped_predictions": punch_unmapped,
        "verified_guard_intervals": metrics.neutral_verified,
        "guard_clean_intervals": metrics.neutral_clean,
        "guard_false_positives": metrics.neutral_false_positives,
        "additional_predictions_observed": additional,
        "duplicate_events": metrics.duplicates,
        "reported_incorrect_detections": metrics.incorrect,
        "counter_events_seen": _counter_totals(list(records))[0],
        "counter_events_ignored": _counter_totals(list(records))[1],
        "definition": {
            "hit": "predicción emparejada y clase igual a la etiqueta verificada",
            "wrong_classification": "predicción de otra clase de golpe en un intento de golpe",
            "no_detection": "fila del golpe verificado en la columna sin_deteccion",
            "guard_false_positive": "cualquier predicción dentro de una guardia verificada",
            "duplicate": "predicción extra de la misma ejecución ya emparejada",
        },
    }


def write_reconciliation_files(
    directory: Path,
    metrics: EvaluationMetrics,
    records: list[RepetitionRecord] | tuple[RepetitionRecord, ...],
    *,
    counter_events_seen: int | None = None,
    counter_events_ignored: int | None = None,
) -> tuple[Path, Path]:
    """Escribe una reconciliación derivada sin mezclar sesiones."""

    reconciliation = reconciliation_metrics(metrics, records)
    if counter_events_seen is not None:
        reconciliation["counter_events_seen"] = counter_events_seen
    if counter_events_ignored is not None:
        reconciliation["counter_events_ignored"] = counter_events_ignored
    payload = {
        "source_report": str(directory),
        "note": "Solo usa metrics, confusion y registros de esta sesión; no compara ni mezcla otras corridas.",
        "reconciliation": reconciliation,
    }
    json_path = directory / "metric_reconciliation.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = directory / "metric_reconciliation.csv"
    rows = [
        ("verified_punch_attempts", reconciliation["verified_punch_attempts"], "intentos de golpes verificados"),
        ("punch_hits", reconciliation["punch_hits"], "golpes con clase correcta"),
        ("punch_wrong_classifications", reconciliation["punch_wrong_classifications"], "golpes predichos con otra clase"),
        ("punch_no_detection", reconciliation["punch_no_detection"], "golpes en sin_deteccion"),
        ("punch_unmapped_predictions", reconciliation["punch_unmapped_predictions"], "predicciones sin etiqueta EvoBox en golpes"),
        ("guard_false_positives", reconciliation["guard_false_positives"], "predicciones durante guardia verificada"),
        ("additional_predictions_observed", reconciliation["additional_predictions_observed"], "predicciones adicionales por intento"),
        ("duplicate_events", reconciliation["duplicate_events"], "duplicados emparejados"),
        ("reported_incorrect_detections", reconciliation["reported_incorrect_detections"], "campo histórico incorrect_detections"),
        ("counter_events_seen", reconciliation["counter_events_seen"], "eventos recibidos por contador"),
        ("counter_events_ignored", reconciliation["counter_events_ignored"], "eventos ignorados por contador"),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metrica", "valor", "definicion"))
        writer.writerows(rows)
    return json_path, csv_path


def reconcile_saved_report(directory: Path) -> tuple[Path, Path]:
    """Audita un informe existente, sin regenerar sus archivos históricos."""

    metrics_path = directory / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    classes = payload.get("classes", [])
    scores = {}
    for item in classes:
        if isinstance(item, dict) and "label_id" in item:
            scores[int(item["label_id"])] = item
    matrix = tuple(tuple(int(value) for value in row) for row in payload.get("confusion", ()))
    if len(matrix) != 5 or any(len(row) != 6 for row in matrix):
        raise ValueError(f"La matriz histórica no tiene forma 5x6: {metrics_path}")
    records: list[RepetitionRecord] = []
    for item in payload.get("repetitions", ()):
        record = RepetitionRecord(
            requested_label_id=int(item["requested_label_id"]),
            repetition_index=int(item["repetition_index"]),
            status=str(item.get("status", "pending")),
            verified_label_id=item.get("verified_label_id"),
            predictions=[
                PredictionMark(
                    timestamp_ms=int(mark["timestamp_ms"]),
                    label_id=mark.get("label_id"),
                    display_name=str(mark.get("display_name", "")),
                    latency_ms=mark.get("latency_ms"),
                )
                for mark in item.get("predictions", ())
            ],
        )
        records.append(record)
    from app.benchmark.matching import ClassScore

    metric = EvaluationMetrics(
        scores={
            label: ClassScore(
                label,
                int(scores.get(label, {}).get("support", 0)),
                int(scores.get(label, {}).get("true_positives", 0)),
                int(scores.get(label, {}).get("false_positives", 0)),
                int(scores.get(label, {}).get("false_negatives", 0)),
                int(scores.get(label, {}).get("duplicates", 0)),
                scores.get(label, {}).get("precision"),
                scores.get(label, {}).get("recall"),
                scores.get(label, {}).get("f1"),
            )
            for label in range(5)
        },
        confusion=matrix,
        neutral_false_positives=int(payload.get("neutral_false_positives", 0)),
        neutral_verified=int(payload.get("neutral_verified", 0)),
        neutral_clean=int(payload.get("neutral_clean", 0)),
        duplicates=int(payload.get("duplicates", 0)),
        incorrect=int(payload.get("incorrect_detections", 0)),
        unmapped_predictions=int(payload.get("unmapped_predictions", 0)),
        verified_repetitions=int(payload.get("verified_repetitions", 0)),
        excluded_repetitions=int(payload.get("excluded_repetitions", 0)),
        mean_latency_ms=payload.get("mean_latency_ms"),
        latency_samples=int(payload.get("latency_samples", 0)),
        mean_fps=payload.get("mean_fps"),
        matched_pairs=sum(item.get("true_positives", 0) for item in scores.values()),
    )
    return write_reconciliation_files(
        directory,
        metric,
        records,
        counter_events_seen=int(payload.get("counter_events_seen", 0)),
        counter_events_ignored=int(payload.get("counter_events_ignored", 0)),
    )


def write_diagnostic_audit(directory: Path) -> Path:
    """Revisa un informe guardado sin sobrescribir sus archivos históricos."""

    metrics_path = directory / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    diagnoses = {}
    for item in payload.get("repetitions", ()):
        predictions = [
            PredictionMark(
                timestamp_ms=int(mark["timestamp_ms"]),
                label_id=mark.get("label_id"),
                display_name=str(mark.get("display_name", "")),
                latency_ms=mark.get("latency_ms"),
                arm=mark.get("arm"),
                event_start_ms=mark.get("event_start_ms"),
                event_end_ms=mark.get("event_end_ms"),
                event_duration_ms=mark.get("event_duration_ms"),
                cooldown_ms=mark.get("cooldown_ms"),
                score_labels=tuple(mark.get("score_labels", ())),
                scores=tuple(mark.get("scores", ())),
            )
            for mark in item.get("predictions", ())
        ]
        record = RepetitionRecord(
            requested_label_id=int(item["requested_label_id"]),
            repetition_index=int(item["repetition_index"]),
            status=str(item.get("status", "pending")),
            verified_label_id=item.get("verified_label_id"),
            predictions=predictions,
            evidence=list(item.get("evidence", ())),
        )
        revised = _diagnose_record(record)
        diagnoses[revised] = diagnoses.get(revised, 0) + 1
        evidence = record.evidence
        tracking = _numeric_values(evidence, "tracking_quality")
        rejections = [
            str(entry.get("rejection"))
            for entry in evidence
            if entry.get("rejection")
        ]
        rows.append(
            {
                "solicitado": record.requested_name,
                "repeticion": record.repetition_index,
                "diagnostico_anterior": item.get("diagnosis", ""),
                "diagnostico_revisado": revised,
                "rechazo_terminal": rejections[-1] if rejections else "",
                "predicciones": len(record.predictions),
                "tracking_min": _min_or_blank(tracking),
                "fotogramas_sin_pose": sum(
                    not bool(entry.get("pose_detected", False)) for entry in evidence
                ),
                "eventos_contador": sum(
                    int(entry.get("counter_delta", 0) or 0) for entry in evidence
                ),
                "eventos_ignorados_contador": sum(
                    int(entry.get("counter_ignored", 0) or 0) for entry in evidence
                ),
            }
        )
    csv_path = directory / "diagnostic_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else (
            "solicitado", "repeticion", "diagnostico_anterior", "diagnostico_revisado",
            "rechazo_terminal", "predicciones", "tracking_min", "fotogramas_sin_pose",
            "eventos_contador", "eventos_ignorados_contador",
        ))
        writer.writeheader()
        writer.writerows(rows)
    json_path = directory / "diagnostic_audit.json"
    json_path.write_text(
        json.dumps(
            {
                "source_report": str(metrics_path),
                "note": "Auditoría derivada; no reemplaza metrics.json ni attempts.csv.",
                "diagnosis_counts": diagnoses,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return json_path


def summary_text(
    metrics: EvaluationMetrics,
    sensitivity: str,
    classifier_name: str,
    *,
    records: list[RepetitionRecord] | tuple[RepetitionRecord, ...] = (),
) -> str:
    reconciliation = reconciliation_metrics(metrics, records)
    lines = [
        f"Evaluación {classifier_name} / {sensitivity}",
        GROUND_TRUTH_NOTE,
        (
            f"Verificadas: {metrics.verified_repetitions}. "
            f"Excluidas: {metrics.excluded_repetitions}. "
            f"Repetidas: {sum(record.status == 'repeated' for record in records)}. "
            f"Duplicados: {metrics.duplicates}. "
            f"Incorrectas: {metrics.incorrect}. "
            f"Falsos positivos en guardia: {metrics.neutral_false_positives}."
        ),
        (
            "Reconciliación de golpes: "
            f"aciertos {reconciliation['punch_hits']}, "
            f"clase incorrecta {reconciliation['punch_wrong_classifications']}, "
            f"sin detección {reconciliation['punch_no_detection']}, "
            f"no mapeadas {reconciliation['punch_unmapped_predictions']}."
        ),
        (
            "Guardia y contador separados: "
            f"falsos positivos de guardia {reconciliation['guard_false_positives']}, "
            f"duplicados {reconciliation['duplicate_events']}, "
            f"predicciones adicionales {reconciliation['additional_predictions_observed']}, "
            f"eventos ignorados por contador {reconciliation['counter_events_ignored']}."
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
        supports = tuple(
            int(item.get("support", 0)) for item in payload.get("classes", ())
        )
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
                "class_support": ",".join(str(value) for value in supports),
                "comparable_group": "|".join(str(value) for value in supports),
                "report": path.parent.as_posix(),
            }
        )
    if not rows:
        return None
    destination = benchmarks_root / "comparison.json"
    groups: dict[str, int] = {}
    for row in rows:
        group = str(row["comparable_group"])
        groups[group] = groups.get(group, 0) + 1
    for row in rows:
        row["comparable"] = groups[str(row["comparable_group"])] > 1
    destination.write_text(
        json.dumps(
            {
                "note": COMPARISON_NOTE,
                "winner": None,
                "comparable_rule": "solo perfiles con el mismo soporte verificado por clase",
                "profiles": rows,
            },
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
    diagnosis = _diagnose_record(record)
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
        "exclusion_reason": record.exclusion_reason,
        "diagnosis": diagnosis,
        "prediction_count": len(record.predictions),
        "predictions": [
            {
                "timestamp_ms": mark.timestamp_ms,
                "label_id": mark.label_id,
                "display_name": mark.display_name,
                "latency_ms": mark.latency_ms,
                "arm": mark.arm,
                "event_start_ms": mark.event_start_ms,
                "event_end_ms": mark.event_end_ms,
                "event_duration_ms": mark.event_duration_ms,
                "cooldown_ms": mark.cooldown_ms,
                "score_labels": list(mark.score_labels),
                "scores": list(mark.scores),
            }
            for mark in record.predictions
        ],
        "fps_samples": len(record.fps_samples),
        "mean_fps": (
            None
            if not record.fps_samples
            else sum(record.fps_samples) / len(record.fps_samples)
        ),
        "evidence": record.evidence,
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
                "brazo",
                "inicio_evento_ms",
                "fin_evento_ms",
                "duracion_evento_ms",
                "cooldown_ms",
                "score_labels",
                "scores",
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
                        *([""] * 10),
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
                        mark.arm or "",
                        "" if mark.event_start_ms is None else mark.event_start_ms,
                        "" if mark.event_end_ms is None else mark.event_end_ms,
                        "" if mark.event_duration_ms is None else mark.event_duration_ms,
                        "" if mark.cooldown_ms is None else mark.cooldown_ms,
                        json.dumps(mark.score_labels, ensure_ascii=False),
                        json.dumps(mark.scores, ensure_ascii=False),
                    )
                )


def _write_attempts_csv(
    path: Path,
    records: list[RepetitionRecord],
    classifier_name: str,
) -> None:
    """Escribe una fila por intento, incluso cuando no hubo predicción."""

    fieldnames = (
        "clasificador",
        "solicitado",
        "repeticion",
        "inicio_ms",
        "fin_ms",
        "duracion_ventana_ms",
        "estado",
        "verificado",
        "corregido",
        "predicciones",
        "resultado_conteo",
        "duplicados_observados",
        "clases_predichas",
        "brazos_predichos",
        "inicio_evento_ms",
        "fin_evento_ms",
        "duracion_evento_ms",
        "cooldown_ms",
        "motivo_diagnostico",
        "calidad_tracking_min",
        "calidad_tracking_media",
        "calidad_brazo_izquierdo_min",
        "calidad_brazo_derecho_min",
        "calidad_brazo_izquierdo_media",
        "calidad_brazo_derecho_media",
        "velocidad_izquierda_max",
        "velocidad_derecha_max",
        "extension_izquierda_max",
        "extension_derecha_max",
        "lateral_izquierda_max",
        "lateral_derecha_max",
        "angulo_codo_izquierdo_min",
        "angulo_codo_izquierdo_max",
        "angulo_codo_derecho_min",
        "angulo_codo_derecho_max",
        "trayectoria_muneca_json",
        "scores_json",
        "rechazos_observados_json",
        "cooldown_activo_observado",
        "fotogramas_sin_pose",
        "fps_medio",
        "latencia_media_ms",
        "eventos_recibidos_por_contador",
        "eventos_ignorados_por_contador",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            evidence = record.evidence
            predictions = record.predictions
            tracking = _numeric_values(evidence, "tracking_quality")
            left_tracking = _numeric_values(evidence, "left_arm_tracking_quality")
            right_tracking = _numeric_values(evidence, "right_arm_tracking_quality")
            left_speed = _numeric_values(evidence, "left_speed")
            right_speed = _numeric_values(evidence, "right_speed")
            left_extension = _numeric_values(evidence, "left_extension")
            right_extension = _numeric_values(evidence, "right_extension")
            left_lateral = _numeric_values(evidence, "left_lateral")
            right_lateral = _numeric_values(evidence, "right_lateral")
            left_elbow = _numeric_values(evidence, "left_elbow_angle")
            right_elbow = _numeric_values(evidence, "right_elbow_angle")
            duration = None
            if record.window_start_ms is not None and record.window_end_ms is not None:
                duration = record.window_end_ms - record.window_start_ms
            event_start = _first_mark_value(predictions, "event_start_ms")
            event_end = _first_mark_value(predictions, "event_end_ms")
            event_duration = _first_mark_value(predictions, "event_duration_ms")
            cooldown = _first_mark_value(predictions, "cooldown_ms")
            count_state = "ninguna" if not predictions else "una" if len(predictions) == 1 else "varias"
            trajectories = [
                {
                    "timestamp_ms": item.get("timestamp_ms"),
                    "left_wrist": (item.get("left_wrist_x"), item.get("left_wrist_y")),
                    "right_wrist": (item.get("right_wrist_x"), item.get("right_wrist_y")),
                }
                for item in evidence
            ]
            scores = [
                {
                    "timestamp_ms": item.get("timestamp_ms"),
                    "labels": list(item.get("score_labels", ())),
                    "values": list(item.get("scores", ())),
                }
                for item in evidence
                if item.get("scores")
            ]
            rejections = [
                {
                    "timestamp_ms": item.get("timestamp_ms"),
                    "rejection": item.get("rejection"),
                    "arm_rejections": list(item.get("arm_rejections", ())),
                }
                for item in evidence
                if item.get("rejection") or item.get("arm_rejections")
            ]
            writer.writerow(
                {
                    "clasificador": classifier_name,
                    "solicitado": record.requested_name,
                    "repeticion": record.repetition_index,
                    "inicio_ms": record.window_start_ms,
                    "fin_ms": record.window_end_ms,
                    "duracion_ventana_ms": duration,
                    "estado": record.status,
                    "verificado": record.verified_name or "",
                    "corregido": record.corrected,
                    "predicciones": len(predictions),
                    "resultado_conteo": count_state,
                    "duplicados_observados": max(0, len(predictions) - 1),
                    "clases_predichas": json.dumps([mark.display_name for mark in predictions], ensure_ascii=False),
                    "brazos_predichos": json.dumps([mark.arm for mark in predictions], ensure_ascii=False),
                    "inicio_evento_ms": event_start,
                    "fin_evento_ms": event_end,
                    "duracion_evento_ms": event_duration,
                    "cooldown_ms": cooldown,
                    "motivo_diagnostico": _diagnose_record(record),
                    "calidad_tracking_min": _min_or_blank(tracking),
                    "calidad_tracking_media": _mean_or_blank(tracking),
                    "calidad_brazo_izquierdo_min": _min_or_blank(left_tracking),
                    "calidad_brazo_derecho_min": _min_or_blank(right_tracking),
                    "calidad_brazo_izquierdo_media": _mean_or_blank(left_tracking),
                    "calidad_brazo_derecho_media": _mean_or_blank(right_tracking),
                    "velocidad_izquierda_max": _max_or_blank(left_speed),
                    "velocidad_derecha_max": _max_or_blank(right_speed),
                    "extension_izquierda_max": _max_or_blank(left_extension),
                    "extension_derecha_max": _max_or_blank(right_extension),
                    "lateral_izquierda_max": _max_or_blank(left_lateral),
                    "lateral_derecha_max": _max_or_blank(right_lateral),
                    "angulo_codo_izquierdo_min": _min_or_blank(left_elbow),
                    "angulo_codo_izquierdo_max": _max_or_blank(left_elbow),
                    "angulo_codo_derecho_min": _min_or_blank(right_elbow),
                    "angulo_codo_derecho_max": _max_or_blank(right_elbow),
                    "trayectoria_muneca_json": json.dumps(trajectories, ensure_ascii=False),
                    "scores_json": json.dumps(scores, ensure_ascii=False),
                    "rechazos_observados_json": json.dumps(rejections, ensure_ascii=False),
                    "cooldown_activo_observado": any(
                        bool(item.get("cooldown_active")) for item in evidence
                    ),
                    "fotogramas_sin_pose": sum(
                        not bool(item.get("pose_detected", False)) for item in evidence
                    ),
                    "fps_medio": _mean_or_blank(record.fps_samples),
                    "latencia_media_ms": _mean_or_blank(
                        [mark.latency_ms for mark in predictions if mark.latency_ms is not None]
                    ),
                    "eventos_recibidos_por_contador": sum(
                        int(item.get("counter_delta", 0) or 0) for item in evidence
                    ),
                    "eventos_ignorados_por_contador": sum(
                        int(item.get("counter_ignored", 0) or 0) for item in evidence
                    ),
                }
            )


def _diagnose_record(record: RepetitionRecord) -> str:
    """Clasifica la causa observada sin convertir una hipótesis en verdad."""

    if record.status == "repeated":
        return "repetido_por_usuario"
    if record.status == "excluded":
        return record.exclusion_reason or "excluido_por_usuario"
    if any(int(item.get("counter_ignored", 0) or 0) > 0 for item in record.evidence):
        return "contador_ignoro_evento_valido"
    if len(record.predictions) > 1:
        return "predicciones_multiples_o_duplicadas"
    if not record.predictions:
        evidence = record.evidence
        rejections = [
            str(item.get("rejection"))
            for item in evidence
            if item.get("rejection")
        ]
        terminal_rejection = rejections[-1].lower() if rejections else ""
        if terminal_rejection == "cooldown" or any(
            item.get("cooldown_active")
            and float(item.get("cooldown_remaining_ms") or 0) > 0
            and item is evidence[-1]
            for item in evidence
        ):
            return "cooldown"
        if terminal_rejection in {
            "activacion insuficiente",
            "margen insuficiente",
            "sin compromiso",
        }:
            return "activacion_insuficiente"
        if terminal_rejection == "seguimiento perdido" or any(
            not bool(item.get("pose_detected", False))
            and item.get("phase") not in (None, "neutral")
            for item in evidence
        ):
            return "seguimiento_perdido"
        if terminal_rejection == "tipo no distinguible" or any(
            item.get("status") in {"unknown", "incierto"} for item in evidence
        ):
            return "movimiento_no_identificado"
        if terminal_rejection == "ciclo excedido":
            return "fin_del_golpe_no_confirmado"
        if any(item.get("phase") not in (None, "neutral") for item in evidence):
            return "fin_del_golpe_no_confirmado"
        if any(
            item.get("max_speed_possible", False) and item.get("tracking_quality", 1.0) < 0.5
            for item in evidence
        ):
            return "posible_perdida_durante_movimiento_rapido"
        return "clasificador_sin_evento_o_causa_desconocida"
    if any(mark.label_id is None for mark in record.predictions):
        return "clase_externa_sin_mapeo_a_0_4"
    if record.verified_label_id is not None and record.predictions[0].label_id != record.verified_label_id:
        return "clase_incorrecta_o_brazo_confundido"
    return "evento_confirmado"


def _numeric_values(items: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for item in items:
        value = item.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _first_mark_value(marks: list[object], field: str) -> object:
    for mark in marks:
        value = getattr(mark, field, None)
        if value is not None:
            return value
    return ""


def _min_or_blank(values: list[float] | tuple[float, ...]) -> object:
    return "" if not values else min(values)


def _max_or_blank(values: list[float] | tuple[float, ...]) -> object:
    return "" if not values else max(values)


def _mean_or_blank(values: list[float] | tuple[float, ...]) -> object:
    return "" if not values else sum(values) / len(values)


def _counter_totals(records: list[RepetitionRecord]) -> tuple[int, int]:
    received = 0
    ignored = 0
    for record in records:
        for item in record.evidence:
            received += int(item.get("counter_delta", 0) or 0)
            ignored += int(item.get("counter_ignored", 0) or 0)
    return received, ignored


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
