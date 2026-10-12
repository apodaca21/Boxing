"""Lectura e integridad del dataset real recopilado en la Fase 2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from app.config import AppConfig
from app.dataset_manager import MOVEMENTS, DatasetError, DatasetManager, MovementSample
from app.evolution.features import extract_raw_features
from app.sample_validator import validate_sample


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample: MovementSample
    person_id: str | None
    features: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    ready: bool
    reasons: tuple[str, ...]
    sample_count: int
    valid_count: int
    class_counts: dict[int, int]
    session_counts: dict[str, int]
    invalid_samples: tuple[dict[str, object], ...]
    warnings: tuple[str, ...]
    person_ids: tuple[str, ...]
    person_split_available: bool
    records: tuple[SampleRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_type": "data_audit",
            "contains_experimental_metrics": False,
            "ready_for_training": self.ready,
            "reasons": list(self.reasons),
            "sample_count": self.sample_count,
            "valid_count": self.valid_count,
            "class_counts": {
                f"{label_id}:{MOVEMENTS[label_id].name}": count
                for label_id, count in self.class_counts.items()
            },
            "session_counts": self.session_counts,
            "invalid_samples": list(self.invalid_samples),
            "warnings": list(self.warnings),
            "person_split_available": self.person_split_available,
            "person_ids": list(self.person_ids),
            "minimum_requirements": {
                "samples_per_class": 1,
                "distinct_sessions": 3,
                "split_unit": "sample",
            },
        }


def audit_dataset(
    data_dir: Path,
    *,
    time_steps: int = 8,
    config: AppConfig | None = None,
) -> DatasetAudit:
    settings = config or AppConfig()
    paths, discovery_errors = _discover_paths(data_dir)
    invalid: list[dict[str, object]] = list(discovery_errors)
    records: list[SampleRecord] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            sample = DatasetManager(data_dir).load_sample(path)
            result = validate_sample(sample, settings)
        except (OSError, json.JSONDecodeError, DatasetError, KeyError, TypeError, ValueError) as error:
            invalid.append({"path": str(path), "reasons": [str(error)]})
            continue
        if not result.accepted:
            invalid.append({"path": str(path), "reasons": list(result.reasons)})
            continue
        person_id = payload.get("person_id") if isinstance(payload, dict) else None
        records.append(
            SampleRecord(
                sample=sample,
                person_id=person_id if isinstance(person_id, str) and person_id else None,
                features=extract_raw_features(sample, time_steps),
            )
        )

    class_counts = {label_id: 0 for label_id in MOVEMENTS}
    session_counts: dict[str, int] = {}
    for record in records:
        class_counts[record.sample.label_id] += 1
        session_counts[record.sample.session_id] = session_counts.get(record.sample.session_id, 0) + 1
    person_ids = tuple(sorted({record.person_id for record in records if record.person_id}))
    person_split_available = (
        len(records) > 0
        and all(record.person_id for record in records)
        and len(person_ids) >= 3
    )
    reasons, warnings = _readiness_reasons(records, class_counts, session_counts, invalid)
    return DatasetAudit(
        ready=not reasons,
        reasons=tuple(reasons),
        sample_count=len(paths),
        valid_count=len(records),
        class_counts=class_counts,
        session_counts=session_counts,
        invalid_samples=tuple(invalid),
        warnings=tuple(warnings),
        person_ids=person_ids,
        person_split_available=person_split_available,
        records=tuple(records),
    )


def _discover_paths(data_dir: Path) -> tuple[list[Path], list[dict[str, object]]]:
    errors: list[dict[str, object]] = []
    if not data_dir.exists():
        return [], errors
    index_path = data_dir / "index.csv"
    if index_path.is_file():
        rows = DatasetManager(data_dir).load_index()
        paths: list[Path] = []
        for row in rows:
            path = data_dir / row["relative_path"]
            if path.is_file():
                paths.append(path)
            else:
                errors.append(
                    {
                        "path": str(path),
                        "reasons": ["El índice referencia una muestra que no existe."],
                    }
                )
        return paths, errors
    return sorted((data_dir / "samples").glob("*/*.json")), errors


def _readiness_reasons(
    records: list[SampleRecord],
    class_counts: dict[int, int],
    session_counts: dict[str, int],
    invalid: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    if not records:
        reasons.append("No hay muestras reales válidas en el directorio de datos.")
    missing = [
        f"{label_id} {MOVEMENTS[label_id].display_name}"
        for label_id, count in class_counts.items()
        if count < 1
    ]
    if missing:
        reasons.append("Faltan muestras reales de: " + ", ".join(missing) + ".")
    if len(session_counts) < 3:
        reasons.append(
            f"Hay {len(session_counts)} sesiones distintas. Se necesitan al menos 3 "
            "para una división por sesión sin fuga entre capturas consecutivas."
        )
    if invalid:
        warnings.append(
            f"{len(invalid)} archivos fueron rechazados por problemas de integridad."
        )
    return reasons, warnings
