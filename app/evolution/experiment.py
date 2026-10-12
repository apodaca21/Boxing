"""Orquestación del experimento. No abre la cámara ni inventa métricas."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from app.config import PROJECT_ROOT
from app.evolution.audit import audit_dataset
from app.evolution.baseline import compare_models
from app.evolution.features import FeatureScaler, vector_size
from app.evolution.metrics import evaluate_predictions
from app.evolution.model_store import EvolvedModel, save_model
from app.evolution.neat_trainer import (
    PRESETS,
    load_neat_config,
    predict_vectors,
    render_config,
    train_population,
)
from app.evolution.reporting import write_confusion_plot, write_history_plots, write_metrics_table
from app.evolution.splitting import split_by_group, training_class_problem


CONFIG_TEMPLATE = PROJECT_ROOT / "config" / "neat_feedforward.cfg"


def run_training(args: object) -> int:
    data_dir = Path(getattr(args, "data_dir")).expanduser().resolve()
    results_dir = Path(getattr(args, "results_dir", PROJECT_ROOT / "results")).expanduser().resolve()
    time_steps = int(getattr(args, "time_steps", 8))
    seed = int(getattr(args, "seed", 42))
    experiment = str(getattr(args, "experiment", "quick"))
    preset = PRESETS[experiment]
    population = int(getattr(args, "population") or preset["population"])
    generations = int(getattr(args, "generations") or preset["generations"])
    split_kind = str(getattr(args, "split", "session"))
    if population < 2 or generations < 1 or time_steps < 2:
        print("La población, las generaciones y los puntos temporales deben ser válidos.")
        return 2

    results_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_dir, time_steps=time_steps)
    _write_json(results_dir / "data_audit.json", audit.to_dict())
    print("Resumen de datos reales:")
    print(f"- muestras encontradas: {audit.sample_count}")
    print(f"- muestras válidas: {audit.valid_count}")
    for label, count in audit.class_counts.items():
        print(f"- clase {label}: {count}")
    print(f"- sesiones: {len(audit.session_counts)}")
    for reason in audit.reasons:
        print(f"- bloqueo: {reason}")
    for warning in audit.warnings:
        print(f"- aviso: {warning}")
    if not audit.ready:
        print("Entrenamiento experimental bloqueado. No se generaron métricas ni gráficas.")
        return 2
    if split_kind == "person" and not audit.person_split_available:
        print(
            "No existe información suficiente de personas distintas. "
            "La evaluación no se presentará como generalización entre usuarios."
        )
        return 2

    groups = {
        record.sample.sample_id: (
            record.person_id if split_kind == "person" else record.sample.session_id
        )
        for record in audit.records
    }
    labels = {record.sample.sample_id: record.sample.label_id for record in audit.records}
    splits = split_by_group(groups, seed=seed, group_kind=split_kind)
    for split_name, sample_ids in (
        ("entrenamiento", splits.train_ids),
        ("validacion", splits.validation_ids),
        ("prueba", splits.test_ids),
    ):
        class_problem = training_class_problem(labels, sample_ids)
        if class_problem:
            print(f"{split_name}: {class_problem}")
            print("No se entrenó. Cada conjunto de la división por sesión debe tener las cinco clases.")
            return 2

    by_id = {record.sample.sample_id: record for record in audit.records}
    train_rows = [by_id[sample_id].features for sample_id in splits.train_ids]
    scaler = FeatureScaler.fit(train_rows, time_steps)
    prepared = {
        sample_id: scaler.transform(record.features)
        for sample_id, record in by_id.items()
    }
    run_dir = results_dir / (
        datetime.now().astimezone().strftime("%Y%m%dT%H%M%S") + f"_{experiment}"
    )
    run_dir.mkdir(parents=True)
    config_text = render_config(
        CONFIG_TEMPLATE.read_text(encoding="utf-8"),
        population=population,
        inputs=vector_size(time_steps),
        max_stagnation=int(preset["max_stagnation"]),
    )
    config_path = run_dir / "neat.cfg"
    config_path.write_text(config_text, encoding="utf-8")
    config = load_neat_config(config_path)
    print(
        f"Experimento {experiment}: población {population}, "
        f"generaciones {generations}, semilla {seed}."
    )
    winner, history = train_population(
        [prepared[sample_id] for sample_id in splits.train_ids],
        [labels[sample_id] for sample_id in splits.train_ids],
        config,
        generations=generations,
        seed=seed,
    )
    metrics = {}
    predictions = {}
    for split_name, sample_ids in (
        ("entrenamiento", splits.train_ids),
        ("validacion", splits.validation_ids),
        ("prueba", splits.test_ids),
    ):
        vectors = [prepared[sample_id] for sample_id in sample_ids]
        truth = [labels[sample_id] for sample_id in sample_ids]
        predicted = predict_vectors(winner, config, vectors)
        predictions[split_name] = predicted
        metrics[split_name] = evaluate_predictions(truth, predicted)

    write_history_plots(history, run_dir)
    write_confusion_plot(
        metrics["entrenamiento"],
        run_dir / "confusion_entrenamiento.png",
        "Matriz de confusión — entrenamiento",
    )
    write_confusion_plot(
        metrics["validacion"],
        run_dir / "confusion_validacion.png",
        "Matriz de confusión — validación",
    )
    write_confusion_plot(
        metrics["prueba"],
        run_dir / "confusion_prueba.png",
        "Matriz de confusión — prueba independiente",
    )
    write_metrics_table(metrics, run_dir / "metricas_por_clase.csv")
    neat_comparison, baseline_comparison = compare_models(
        predictions["prueba"],
        winner,
        [labels[sample_id] for sample_id in splits.test_ids],
        [prepared[sample_id] for sample_id in splits.train_ids],
        [labels[sample_id] for sample_id in splits.train_ids],
        [prepared[sample_id] for sample_id in splits.test_ids],
        seed,
    )
    summary = {
        "experiment_executed": True,
        "evaluation_scope": (
            "División por grupos completos. El conjunto de prueba no participó "
            "en el fitness ni en la selección del ganador."
        ),
        "independent_evaluation": split_kind == "session",
        "person_generalization": split_kind == "person",
        "fitness_coefficients_status": (
            "valores iniciales propuestos; todavía deben validarse experimentalmente"
        ),
        "winner_selection": "fitness de entrenamiento",
        "history": [asdict(record) for record in history],
        "topology": {
            "nodes": len(winner.nodes),
            "hidden_nodes": len(winner.nodes) - 5,
            "enabled_connections": sum(1 for gene in winner.connections.values() if gene.enabled),
            "disabled_connections": sum(1 for gene in winner.connections.values() if not gene.enabled),
        },
        "splits": {
            "kind": split_kind,
            "seed": seed,
            "train_groups": list(splits.train_groups),
            "validation_groups": list(splits.validation_groups),
            "test_groups": list(splits.test_groups),
        },
        "comparison_note": (
            "Comparación descriptiva sobre la misma partición. "
            "No implica que NEAT sea superior."
        ),
        "comparison": {
            "neat": _comparison_dict(neat_comparison),
            "sklearn_mlp_fixed": _comparison_dict(baseline_comparison),
        },
    }
    _write_json(run_dir / "resumen.json", summary)
    save_model(
        run_dir / "model",
        EvolvedModel(
            genome=winner,
            config_text=config_text,
            scaler=scaler,
            seed=seed,
            training={
                "experiment": experiment,
                "population": population,
                "generations": generations,
                "time_steps": time_steps,
                "vector_size": vector_size(time_steps),
                "split": split_kind,
            },
        ),
    )
    print(f"Resultados guardados en {run_dir}")
    return 0


def _comparison_dict(comparison: object) -> dict[str, object]:
    return {
        "macro_f1": comparison.macro_f1,  # type: ignore[attr-defined]
        "neutral_false_positive_rate": comparison.neutral_false_positive_rate,  # type: ignore[attr-defined]
        "parameter_count": comparison.parameter_count,  # type: ignore[attr-defined]
        "inference_seconds": comparison.inference_seconds,  # type: ignore[attr-defined]
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
