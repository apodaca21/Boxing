"""Divisiones reproducibles que conservan cada ejecución en un solo conjunto."""

from __future__ import annotations

from dataclasses import dataclass
import random

from app.dataset_manager import MOVEMENTS


class SplitError(ValueError):
    """La partición solicitada filtraría información o no es viable."""


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    group_kind: str
    seed: int

    def assignment(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for sample_id in self.train_ids:
            mapping[sample_id] = "train"
        for sample_id in self.validation_ids:
            mapping[sample_id] = "validation"
        for sample_id in self.test_ids:
            mapping[sample_id] = "test"
        return mapping


def split_by_group(
    sample_groups: dict[str, str],
    *,
    seed: int,
    group_kind: str,
) -> DatasetSplits:
    """Asigna grupos completos y deja el conjunto de prueba fijo para la semilla."""

    if group_kind not in {"session", "person"}:
        raise SplitError("La división debe hacerse por sesión o por persona.")
    grouped: dict[str, list[str]] = {}
    for sample_id, group_id in sample_groups.items():
        grouped.setdefault(group_id, []).append(sample_id)
    groups = sorted(grouped)
    if len(groups) < 3:
        subject = "sesiones" if group_kind == "session" else "personas"
        raise SplitError(
            f"Hay {len(groups)} {subject}. Se necesitan al menos 3 para separar "
            "entrenamiento, validación y prueba sin mezclar capturas del mismo grupo."
        )

    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    test_count = max(1, round(len(ordered) * 0.20))
    validation_count = max(1, round(len(ordered) * 0.20))
    if test_count + validation_count >= len(ordered):
        test_count = 1
        validation_count = 1
    test_groups = tuple(sorted(ordered[:test_count]))
    validation_groups = tuple(sorted(ordered[test_count:test_count + validation_count]))
    train_groups = tuple(sorted(ordered[test_count + validation_count:]))
    if not train_groups:
        raise SplitError("La división no dejó sesiones para entrenamiento.")

    def collect(selected: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(sample_id for group in selected for sample_id in grouped[group]))

    splits = DatasetSplits(
        train_ids=collect(train_groups),
        validation_ids=collect(validation_groups),
        test_ids=collect(test_groups),
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
        group_kind=group_kind,
        seed=seed,
    )
    _assert_disjoint(splits)
    return splits


def training_class_problem(labels: dict[str, int], sample_ids: tuple[str, ...]) -> str | None:
    counts = {label_id: 0 for label_id in MOVEMENTS}
    for sample_id in sample_ids:
        counts[labels[sample_id]] += 1
    missing = [
        f"{label_id} {MOVEMENTS[label_id].display_name}"
        for label_id, count in counts.items()
        if count < 1
    ]
    if not missing:
        return None
    return (
        "El entrenamiento no tiene suficientes muestras de: "
        + ", ".join(missing)
        + "."
    )


def _assert_disjoint(splits: DatasetSplits) -> None:
    groups = (set(splits.train_groups), set(splits.validation_groups), set(splits.test_groups))
    samples = (set(splits.train_ids), set(splits.validation_ids), set(splits.test_ids))
    if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
        raise SplitError("Un mismo grupo quedó en más de un conjunto.")
    if any(left & right for index, left in enumerate(samples) for right in samples[index + 1:]):
        raise SplitError("Una misma muestra quedó en más de un conjunto.")
