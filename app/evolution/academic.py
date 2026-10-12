"""Auditoría y reporte académico. No sustituye el entrenamiento NEAT."""

from __future__ import annotations

from argparse import Namespace
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import PROJECT_ROOT
from app.dataset_manager import MOVEMENTS
from app.evolution.audit import DatasetAudit, audit_dataset
from app.evolution.metrics import FITNESS_WEIGHTS, ClassificationMetrics, evaluate_predictions
from app.evolution.model_store import load_model
from app.evolution.neat_trainer import PRESETS, neat_parameter_count, predict_vectors
from app.evolution.reporting import write_confusion_plot
from app.evolution.splitting import split_by_group, training_class_problem
from app.geometry import BiomechanicalFeatures, RelativePoint
from app.trainer.heuristic import HeuristicClassifier


PENDING_FILES = (
    "neat_evolution.csv",
    "evaluation_metrics.json",
    "confusion_matrix.png",
    "fitness_evolution.png",
    "model_topology.png",
)


@dataclass(frozen=True, slots=True)
class SplitCoverage:
    ready_to_train: bool
    split_computed: bool
    blocked_reason: str | None
    missing_by_split: dict[str, tuple[str, ...]]
    sessions_per_class: dict[int, int]


def build_academic_package(args: object) -> int:
    """Audita el dataset real y, si alcanza, ejecuta el entrenamiento ya existente."""

    data_dir = Path(getattr(args, "data_dir")).expanduser().resolve()
    results_dir = Path(getattr(args, "results_dir", PROJECT_ROOT / "results")).expanduser().resolve()
    academic_dir = results_dir / "academic"
    academic_dir.mkdir(parents=True, exist_ok=True)
    time_steps = int(getattr(args, "time_steps", 8))
    seed = int(getattr(args, "seed", 42))
    experiment = str(getattr(args, "experiment", "quick"))
    preset = PRESETS[experiment]
    population = int(getattr(args, "population") or preset["population"])
    generations = int(getattr(args, "generations") or preset["generations"])
    split_kind = str(getattr(args, "split", "session"))

    audit = audit_dataset(data_dir, time_steps=time_steps)
    coverage = _coverage(audit, seed=seed, split_kind=split_kind)
    summary = _dataset_summary(audit, coverage, time_steps)
    _write_json(academic_dir / "dataset_summary.json", summary)
    _print_summary(summary)

    config_payload = _experiment_config(
        experiment=experiment,
        population=population,
        generations=generations,
        seed=seed,
        split_kind=split_kind,
        time_steps=time_steps,
        executed=False,
        training_seconds=None,
        run_directory=None,
    )
    executed_metrics: dict[str, object] | None = None
    if coverage.ready_to_train:
        for name in PENDING_FILES:
            stale = academic_dir / name
            if stale.is_file():
                stale.unlink()
        runs_dir = academic_dir / "neat_run"
        training_args = Namespace(
            data_dir=data_dir,
            results_dir=runs_dir,
            time_steps=time_steps,
            seed=seed,
            experiment=experiment,
            population=population,
            generations=generations,
            split=split_kind,
        )
        started = time.perf_counter()
        from app.evolution.experiment import run_training

        code = run_training(training_args)
        training_seconds = time.perf_counter() - started
        run_dir = _latest_run(runs_dir)
        if code == 0 and run_dir is not None:
            executed_metrics = _export_real_run(
                academic_dir,
                run_dir,
                audit,
                seed=seed,
                split_kind=split_kind,
                training_seconds=training_seconds,
            )
            config_payload = _experiment_config(
                experiment=experiment,
                population=population,
                generations=generations,
                seed=seed,
                split_kind=split_kind,
                time_steps=time_steps,
                executed=True,
                training_seconds=training_seconds,
                run_directory=str(run_dir),
            )
        else:
            print("El pipeline NEAT existente no produjo un experimento utilizable.")
    else:
        print("No se ejecutó NEAT. El dataset real todavía no sostiene la división pedida.")
        if coverage.blocked_reason:
            print(coverage.blocked_reason)

    _write_json(academic_dir / "experiment_config.json", config_payload)
    pending = _pending_files(academic_dir)
    report = _academic_markdown(summary, coverage, config_payload, executed_metrics, pending)
    (academic_dir / "academic_report.md").write_text(report, encoding="utf-8")
    print(f"Reporte académico en {academic_dir}")
    if pending:
        print("Pendientes hasta tener un entrenamiento real: " + ", ".join(pending))
    return 0


def _coverage(audit: DatasetAudit, *, seed: int, split_kind: str) -> SplitCoverage:
    sessions_per_class = {label: 0 for label in MOVEMENTS}
    by_session: dict[str, set[int]] = {}
    labels = {}
    groups = {}
    for record in audit.records:
        labels[record.sample.sample_id] = record.sample.label_id
        group = record.person_id if split_kind == "person" and record.person_id else record.sample.session_id
        groups[record.sample.sample_id] = group
        by_session.setdefault(group, set()).add(record.sample.label_id)
    for label in MOVEMENTS:
        sessions_per_class[label] = sum(label in present for present in by_session.values())

    if not audit.ready or split_kind == "person" and not audit.person_split_available:
        reason = audit.reasons[0] if audit.reasons else "El dataset no está listo para entrenar."
        if split_kind == "person" and not audit.person_split_available:
            reason = "No hay al menos tres personas identificadas para esa división."
        return SplitCoverage(False, False, reason, {}, sessions_per_class)
    splits = split_by_group(groups, seed=seed, group_kind=split_kind)
    missing: dict[str, tuple[str, ...]] = {}
    for name, sample_ids in (
        ("entrenamiento", splits.train_ids),
        ("validacion", splits.validation_ids),
        ("prueba", splits.test_ids),
    ):
        problem = training_class_problem(labels, sample_ids)
        if problem:
            missing[name] = tuple(
                MOVEMENTS[label].display_name
                for label in MOVEMENTS
                if sum(labels[sample_id] == label for sample_id in sample_ids) < 1
            )
    train_problem = training_class_problem(labels, splits.train_ids)
    reason = train_problem
    if missing and reason is None:
        detail = ", ".join(f"{name}: {', '.join(labels)}" for name, labels in missing.items())
        reason = "Hay conjuntos sin alguna de las cinco clases. " + detail
    return SplitCoverage(not missing and train_problem is None, True, reason, missing, sessions_per_class)


def _dataset_summary(
    audit: DatasetAudit,
    coverage: SplitCoverage,
    time_steps: int,
) -> dict[str, object]:
    durations = [record.sample.duration_ms for record in audit.records]
    missing_values = 0
    total_values = 0
    sample_ids: list[str] = []
    session_classes: dict[str, dict[str, int]] = {}
    split_units: set[str] = set()
    for record in audit.records:
        sample_ids.append(record.sample.sample_id)
        split_units.add(record.sample.split_unit)
        for value in record.features:
            total_values += 1
            if value is None:
                missing_values += 1
        bucket = session_classes.setdefault(record.sample.session_id, {})
        name = MOVEMENTS[record.sample.label_id].name
        bucket[name] = bucket.get(name, 0) + 1
    duplicate_ids = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
    independent = not duplicate_ids and (not split_units or split_units == {"sample"})
    missing_ratio = None if total_values == 0 else missing_values / total_values
    classes_to_collect = [
        MOVEMENTS[label].display_name
        for label, sessions in coverage.sessions_per_class.items()
        if sessions < 3
    ]
    return {
        "contains_experimental_metrics": False,
        "contains_synthetic_fill": False,
        "sample_count": audit.sample_count,
        "valid_count": audit.valid_count,
        "rejected_count": len(audit.invalid_samples),
        "rejected_samples": list(audit.invalid_samples),
        "session_count": len(audit.session_counts),
        "session_ids_unique": len(audit.session_counts) == len(set(audit.session_counts)),
        "samples_per_class": {
            MOVEMENTS[label].display_name: count for label, count in audit.class_counts.items()
        },
        "samples_per_session": dict(audit.session_counts),
        "classes_per_session": session_classes,
        "sessions_per_class": {
            MOVEMENTS[label].display_name: count
            for label, count in coverage.sessions_per_class.items()
        },
        "duration_ms": {
            "count": len(durations),
            "min": None if not durations else min(durations),
            "max": None if not durations else max(durations),
            "mean": None if not durations else sum(durations) / len(durations),
        },
        "integrity": {
            "independent_samples": independent,
            "duplicate_sample_ids": duplicate_ids,
            "split_unit": sorted(split_units),
            "invalid_files": len(audit.invalid_samples),
        },
        "missing_feature_ratio": missing_ratio,
        "missing_feature_values": missing_values,
        "feature_values": total_values,
        "time_steps": time_steps,
        "ready_for_training": coverage.ready_to_train,
        "training_block": coverage.blocked_reason,
        "classes_missing_from_splits": {
            name: list(labels) for name, labels in coverage.missing_by_split.items()
        },
        "additional_samples_needed": classes_to_collect,
        "split_policy": (
            "Cada ejecución permanece en un solo conjunto. "
            "La prueba no se usa para elegir el genoma ni para ajustar hiperparámetros."
        ),
        "warnings": list(audit.warnings),
        "reasons": list(audit.reasons),
    }


def _experiment_config(
    *,
    experiment: str,
    population: int,
    generations: int,
    seed: int,
    split_kind: str,
    time_steps: int,
    executed: bool,
    training_seconds: float | None,
    run_directory: str | None,
) -> dict[str, object]:
    return {
        "experiment_executed": executed,
        "measured_results": executed,
        "experiment": experiment,
        "population": population,
        "generations": generations,
        "seed": seed,
        "split": split_kind,
        "time_steps": time_steps,
        "outputs": 5,
        "winner_selection": "mayor fitness calculado solamente con las muestras de entrenamiento",
        "test_set_used_for_selection": False,
        "test_set_used_for_hyperparameters": False,
        "fitness_formula": "F(g)=0.65*R_punch+0.30*S_neutral+0.05/(1+C)",
        "fitness_weights": {
            "punch_recall": FITNESS_WEIGHTS[0],
            "neutral_success": FITNESS_WEIGHTS[1],
            "complexity": FITNESS_WEIGHTS[2],
        },
        "fitness_coefficients_status": (
            "valores iniciales propuestos; todavía deben validarse experimentalmente"
        ),
        "training_seconds": training_seconds,
        "run_directory": run_directory,
        "algorithm_source": "app/evolution/neat_trainer.py",
    }


def _export_real_run(
    academic_dir: Path,
    run_dir: Path,
    audit: DatasetAudit,
    *,
    seed: int,
    split_kind: str,
    training_seconds: float,
) -> dict[str, object]:
    summary = json.loads((run_dir / "resumen.json").read_text(encoding="utf-8"))
    history = summary["history"]
    _write_history_csv(academic_dir / "neat_evolution.csv", history)
    _write_fitness_plot(academic_dir / "fitness_evolution.png", history)
    model = load_model(run_dir / "model")
    labels = {record.sample.sample_id: record.sample.label_id for record in audit.records}
    groups = {
        record.sample.sample_id: (
            record.person_id if split_kind == "person" and record.person_id else record.sample.session_id
        )
        for record in audit.records
    }
    splits = split_by_group(groups, seed=seed, group_kind=split_kind)
    by_id = {record.sample.sample_id: record for record in audit.records}
    config = model.neat_config()
    metrics_payload: dict[str, object] = {
        "experiment_executed": True,
        "winner_selected_on": "entrenamiento",
        "test_set_used_for_selection": False,
        "test_set_used_for_hyperparameters": False,
        "training_seconds": training_seconds,
        "topology": summary["topology"],
        "splits": {},
    }
    prepared_test = []
    test_samples = []
    test_truth = []
    for split_name, sample_ids in (
        ("entrenamiento", splits.train_ids),
        ("validacion", splits.validation_ids),
        ("prueba", splits.test_ids),
    ):
        vectors = [model.scaler.transform(by_id[sample_id].features) for sample_id in sample_ids]
        truth = [labels[sample_id] for sample_id in sample_ids]
        started = time.perf_counter()
        predicted = predict_vectors(model.genome, config, vectors)
        elapsed = time.perf_counter() - started
        metrics = evaluate_predictions(truth, predicted)
        metrics_payload["splits"][split_name] = _metrics_dict(metrics, elapsed, len(sample_ids))
        if split_name == "prueba":
            prepared_test = vectors
            test_samples = [by_id[sample_id].sample for sample_id in sample_ids]
            test_truth = truth
            write_confusion_plot(
                metrics,
                academic_dir / "confusion_matrix.png",
                "Matriz de confusión — prueba independiente",
            )
    _write_topology(academic_dir / "model_topology.png", model.genome)
    metrics_payload["heuristic_comparison"] = _heuristic_comparison(
        model,
        config,
        test_samples,
        test_truth,
        prepared_test,
    )
    metrics_payload["method_note"] = (
        "NEAT clasifica cada secuencia recortada y etiquetada. "
        "El detector heurístico en vivo cuenta eventos sobre video continuo. "
        "Esas dos tareas no comparten una sola cifra de precisión. "
        "La comparación incluida reevalúa el heurístico sobre las mismas secuencias de prueba, "
        "una etiqueta por muestra, y deja fuera el benchmark de video."
    )
    _write_json(academic_dir / "evaluation_metrics.json", metrics_payload)
    return metrics_payload


def _heuristic_comparison(model, config, samples, truth, vectors) -> dict[str, object]:
    heuristic = HeuristicClassifier()
    comparable_truth: list[int] = []
    comparable_heuristic: list[int] = []
    comparable_neat: list[int] = []
    ambiguous = 0
    started = time.perf_counter()
    heuristic_labels = [_heuristic_label(heuristic, sample) for sample in samples]
    heuristic_seconds = time.perf_counter() - started
    neat_started = time.perf_counter()
    neat_labels = predict_vectors(model.genome, config, vectors)
    neat_seconds = time.perf_counter() - neat_started
    for actual, heuristic_label, neat_label in zip(truth, heuristic_labels, neat_labels):
        if heuristic_label is None:
            ambiguous += 1
            continue
        comparable_truth.append(actual)
        comparable_heuristic.append(heuristic_label)
        comparable_neat.append(neat_label)
    heuristic_metrics = (
        None if not comparable_truth else evaluate_predictions(comparable_truth, comparable_heuristic)
    )
    neat_metrics = (
        None if not comparable_truth else evaluate_predictions(comparable_truth, comparable_neat)
    )
    return {
        "task": "una etiqueta por secuencia recortada",
        "excluded_task": "detección de eventos en video continuo",
        "test_sequences": len(samples),
        "compared_sequences": len(comparable_truth),
        "ambiguous_heuristic_sequences": ambiguous,
        "learned_parameters": {
            "neat": neat_parameter_count(model.genome),
            "heuristic": 0,
        },
        "inference_seconds": {
            "neat_test_sequences": neat_seconds,
            "heuristic_test_sequences": heuristic_seconds,
        },
        "heuristic_on_compared_sequences": None
        if heuristic_metrics is None
        else _metrics_dict(heuristic_metrics, heuristic_seconds, len(comparable_truth)),
        "neat_on_compared_sequences": None
        if neat_metrics is None
        else _metrics_dict(neat_metrics, neat_seconds, len(comparable_truth)),
        "ranking": None,
    }


def _heuristic_label(classifier: HeuristicClassifier, sample) -> int | None:
    classifier.reset()
    found: list[int] = []
    for frame in sample.frames:
        features = frame.features
        result = classifier.update(
            BiomechanicalFeatures(
                timestamp_ms=frame.timestamp_ms,
                left_elbow_angle=features.left_elbow_angle,
                right_elbow_angle=features.right_elbow_angle,
                left_shoulder_wrist_distance=features.left_shoulder_wrist_distance,
                right_shoulder_wrist_distance=features.right_shoulder_wrist_distance,
                left_wrist_relative=None
                if features.left_wrist_relative is None
                else RelativePoint(features.left_wrist_relative.x, features.left_wrist_relative.y),
                right_wrist_relative=None
                if features.right_wrist_relative is None
                else RelativePoint(features.right_wrist_relative.x, features.right_wrist_relative.y),
                wrist_separation=features.wrist_separation,
            ),
            pose_detected=frame.pose_detected,
        )
        for event in result.events:
            if event.label_id in (1, 2, 3, 4):
                found.append(event.label_id)
    if not found:
        return 0
    unique = set(found)
    if len(unique) == 1:
        return unique.pop()
    return None


def _metrics_dict(metrics: ClassificationMetrics, seconds: float, sequences: int) -> dict[str, object]:
    return {
        "sequences": sequences,
        "inference_seconds": seconds,
        "macro_f1": metrics.macro_f1,
        "neutral_false_positive_rate": metrics.neutral_false_positive_rate,
        "confusion": [list(row) for row in metrics.confusion],
        "per_class": [
            {
                "label": MOVEMENTS[label].display_name,
                "support": metrics.support[label],
                "precision": metrics.precision[label],
                "recall": metrics.recall[label],
                "f1": metrics.f1[label],
            }
            for label in MOVEMENTS
        ],
    }


def _write_history_csv(path: Path, history: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "generation",
                "best_fitness",
                "mean_fitness",
                "species_count",
                "best_enabled_connections",
            ),
        )
        writer.writeheader()
        for row in history:
            writer.writerow({name: row[name] for name in writer.fieldnames})


def _write_fitness_plot(path: Path, history: list[dict[str, object]]) -> None:
    generations = [int(row["generation"]) for row in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(generations, [float(row["best_fitness"]) for row in history], label="Fitness máximo")
    axis.plot(generations, [float(row["mean_fitness"]) for row in history], label="Fitness promedio")
    axis.set_xlabel("Generación")
    axis.set_ylabel("Fitness de entrenamiento")
    axis.set_title("Evolución medida en el entrenamiento")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_topology(path: Path, genome) -> None:
    inputs = sorted(key for key in genome.nodes if key < 0)
    outputs = sorted(key for key in genome.nodes if 0 <= key < len(MOVEMENTS))
    hidden = sorted(key for key in genome.nodes if key >= len(MOVEMENTS))
    positions: dict[int, tuple[float, float]] = {}
    for index, key in enumerate(inputs):
        positions[key] = (0.0, float(index))
    for index, key in enumerate(hidden):
        positions[key] = (1.0, float(index))
    for index, key in enumerate(outputs):
        positions[key] = (2.0, float(index))
    figure, axis = plt.subplots(figsize=(8, 5))
    for gene in genome.connections.values():
        if not gene.enabled:
            continue
        source, target = gene.key
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        axis.plot([x1, x2], [y1, y2], color="#8d97a8", linewidth=0.7)
    for key, (x, y) in positions.items():
        axis.scatter([x], [y], s=28, color="#2f6f4e" if key >= 0 else "#2458a6")
    axis.set_xticks([0, 1, 2], ["Entradas", "Ocultas", "Salidas"])
    axis.set_title("Topología del genoma ganador")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _latest_run(runs_dir: Path) -> Path | None:
    if not runs_dir.is_dir():
        return None
    runs = [path.parent for path in runs_dir.glob("*/resumen.json")]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def _pending_files(academic_dir: Path) -> list[str]:
    return [name for name in PENDING_FILES if not (academic_dir / name).is_file()]


def _academic_markdown(
    summary: dict[str, object],
    coverage: SplitCoverage,
    config: dict[str, object],
    metrics: dict[str, object] | None,
    pending: list[str],
) -> str:
    needed = summary["additional_samples_needed"]
    needed_text = ", ".join(needed) if isinstance(needed, list) and needed else "ninguna clase adicional identificada"
    lines = [
        "# EvoBox AI — Reporte de cómputo evolutivo",
        "",
        "## 1. Definición del problema",
        "",
        "El problema es clasificar una ejecución recortada de boxeo orthodox en una de cinco clases: "
        "guardia, jab izquierdo, cross derecho, gancho izquierdo y gancho derecho. "
        "La entrada es una secuencia biomecánica, no una imagen ni un video.",
        "",
        "## 2. Justificación del algoritmo NEAT",
        "",
        "NEAT permite que la topología de la red crezca junto con los pesos. "
        "Eso evita fijar de antemano el número de neuronas ocultas cuando el conjunto etiquetado todavía es pequeño. "
        "La función objetivo penaliza las conexiones habilitadas, de modo que una red más grande solo conviene si clasifica mejor.",
        "",
        "## 3. Representación del individuo",
        "",
        "Cada individuo es un genoma feedforward de neat-python. "
        f"La entrada tiene el tamaño del vector temporal de {summary['time_steps']} puntos. "
        "Hay cinco salidas, una por clase. Los nodos y las conexiones habilitadas forman la red que se evalúa. "
        "La etiqueta predicha es el índice de la mayor activación. Esas activaciones no son probabilidades calibradas.",
        "",
        "## 4. Operadores evolutivos",
        "",
        "La configuración de `config/neat_feedforward.cfg` usa reproducción con elitismo 1 y umbral de supervivencia 0.2. "
        "Las mutaciones estructurales configuradas son alta de conexión (0.3), baja de conexión (0.2), "
        "alta de nodo (0.2) y baja de nodo (0.1). "
        "Los pesos se mutan con probabilidad 0.8 y los sesgos con probabilidad 0.7. "
        "La especiación usa el umbral de compatibilidad 3.0 y la estagnación definida en esa misma plantilla. "
        "No se añadió ningún operador fuera de esa configuración.",
        "",
        "## 5. Función objetivo matemática",
        "",
        "Para un genoma g, con R_punch el recall medio de las cuatro clases de golpe, "
        "S_neutral el recall de la guardia y C el número de conexiones habilitadas:",
        "",
        "F(g) = 0.65 R_punch + 0.30 S_neutral + 0.05 / (1 + C)",
        "",
        "El fitness solo se calcula si el conjunto de entrenamiento contiene las cinco clases. "
        "Los coeficientes 0.65, 0.30 y 0.05 son una propuesta inicial y todavía no están validados.",
        "",
        "## 6. Metodología experimental",
        "",
        "Las muestras se separan por sesión completa. Una ejecución no puede quedar en dos conjuntos. "
        "El genoma ganador es el de mayor fitness de entrenamiento. "
        "El conjunto de prueba no participa en ese fitness, ni en la selección, ni en el ajuste de población, generaciones o semilla. "
        f"Semilla {config['seed']}, población {config['population']}, generaciones {config['generations']}, "
        f"división `{config['split']}`.",
        "",
        _split_gap_text(coverage),
        "",
        "El detector heurístico cuenta ciclos de un brazo en una transmisión continua. "
        "NEAT asigna una clase a una ventana ya recortada. "
        "Un recall de eventos en video y un F1 de secuencias etiquetadas no describen la misma tarea.",
        "",
        "## 7. Resultados reales",
        "",
        _results_section(summary, metrics),
        "",
        "## 8. Interpretación del fitness",
        "",
        "Un fitness alto en entrenamiento indica que, en esas muestras, el genoma recuperó golpes y guardia "
        "sin usar demasiadas conexiones. No demuestra por sí solo generalización. "
        "La lectura de generalización corresponde a las métricas de validación y, sobre todo, a las de prueba. "
        "Si el soporte de una clase en prueba es cero, esa clase no quedó evaluada.",
        "",
        "## 9. Limitaciones",
        "",
        "Los coeficientes de la función objetivo no han sido calibrados con un estudio aparte. "
        "La división por sesión exige que cada clase aparezca en varias sesiones; de lo contrario una clase puede quedar fuera de la prueba. "
        f"Clases que hoy no están en al menos tres sesiones: {needed_text}. "
        "El heurístico comparado sobre secuencias usa las características guardadas, no la pose 3D del modo en vivo. "
        "No se generaron muestras sintéticas para completar clases.",
        "",
        "## 10. Trabajo pendiente",
        "",
    ]
    if pending:
        lines.append(
            "Faltan estos productos porque no hay un entrenamiento real que los respalde: "
            + ", ".join(pending)
            + "."
        )
    else:
        lines.append("Los archivos de evolución y evaluación de esta corrida están en `results/academic/`.")
    lines.extend(
        [
            "",
            "Sigue pendiente recopilar sesiones reales suficientes, repetir este comando y contrastar, "
            "en otro informe, el benchmark de video continuo con esta clasificación de secuencias.",
            "",
        ]
    )
    if coverage.blocked_reason and metrics is None:
        lines.append(f"Bloqueo actual: {coverage.blocked_reason}")
        lines.append("")
    return "\n".join(lines)


def _split_gap_text(coverage: SplitCoverage) -> str:
    if not coverage.split_computed:
        return (
            "La división por sesión no se aplicó porque el dataset todavía no cumple "
            "el mínimo de clases y de sesiones. No se fabricaron muestras para completarla."
        )
    if not coverage.missing_by_split:
        return (
            "Con la semilla actual, cada conjunto contiene las cinco clases. "
            "Eso no sustituye a nuevas sesiones reales cuando el soporte de prueba sea pequeño."
        )
    parts = [
        f"{name}: {', '.join(labels)}"
        for name, labels in coverage.missing_by_split.items()
    ]
    return (
        "Esta división no evalúa todas las clases en todos los conjuntos. "
        "Faltan " + "; ".join(parts) + ". "
        "Hay que recopilar esas clases en sesiones distintas, sin fabricar muestras."
    )


def _results_section(summary: dict[str, object], metrics: dict[str, object] | None) -> str:
    duration = summary["duration_ms"]
    assert isinstance(duration, dict)
    text = [
        f"Muestras encontradas: {summary['sample_count']}. Válidas: {summary['valid_count']}. "
        f"Sesiones: {summary['session_count']}. Rechazadas: {summary['rejected_count']}.",
        f"Muestras por clase: {summary['samples_per_class']}.",
        f"Duración en milisegundos: mínimo {duration['min']}, máximo {duration['max']}, media {duration['mean']}.",
        f"Proporción de características ausentes: {summary['missing_feature_ratio']}.",
        f"Muestras independientes: {summary['integrity']}.",
    ]
    if metrics is None:
        text.append(
            "No hay fitness, matriz de confusión, F1 ni latencia de un entrenamiento real en este directorio. "
            "Esas cifras quedan pendientes."
        )
        return "\n\n".join(text)
    splits = metrics["splits"]
    assert isinstance(splits, dict)
    text.append(
        f"Entrenamiento medido: {metrics['training_seconds']:.3f} s. "
        f"Topología ganadora: {metrics['topology']}."
    )
    for name in ("entrenamiento", "validacion", "prueba"):
        block = splits[name]
        assert isinstance(block, dict)
        text.append(
            f"{name}: secuencias {block['sequences']}, "
            f"F1 macro {block['macro_f1']}, "
            f"falsos positivos de guardia {block['neutral_false_positive_rate']}, "
            f"inferencia {block['inference_seconds']:.6f} s."
        )
    comparison = metrics["heuristic_comparison"]
    assert isinstance(comparison, dict)
    text.append(
        "La comparación con el heurístico usa solo secuencias de prueba con una etiqueta inequívoca. "
        f"Secuencias comparadas: {comparison['compared_sequences']} de {comparison['test_sequences']}. "
        f"Parámetros aprendidos: {comparison['learned_parameters']}. "
        "No se declara un modelo ganador."
    )
    return "\n\n".join(text)


def _print_summary(summary: dict[str, object]) -> None:
    print("Auditoría del dataset real:")
    print(f"- muestras: {summary['sample_count']}")
    print(f"- válidas: {summary['valid_count']}")
    print(f"- sesiones: {summary['session_count']}")
    print(f"- rechazadas: {summary['rejected_count']}")
    print(f"- por clase: {summary['samples_per_class']}")
    print(f"- por sesión: {summary['samples_per_session']}")
    print(f"- duración ms: {summary['duration_ms']}")
    print(f"- características ausentes: {summary['missing_feature_ratio']}")
    print(f"- integridad: {summary['integrity']}")
    print(f"- el entrenamiento existente puede ejecutarse: {summary['ready_for_training']}")
    if summary["classes_missing_from_splits"]:
        print(f"- clases ausentes por conjunto: {summary['classes_missing_from_splits']}")
    if summary["additional_samples_needed"]:
        print("- todavía hay que recopilar: " + ", ".join(summary["additional_samples_needed"]))
    if summary["training_block"]:
        print(f"- bloqueo: {summary['training_block']}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
