"""Empareja cada predicción con un solo golpe verificado.

La consigna de la pantalla no cuenta como acierto. Una repetición entra aquí
solo después de que la persona confirma o corrige lo que ejecutó.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.benchmark.protocol import PredictionMark, RepetitionRecord


MISS = 5


@dataclass(frozen=True, slots=True)
class ClassScore:
    label_id: int
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    duplicates: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Métricas de una sesión verificada. Los nulos indican que no hubo soporte."""

    scores: dict[int, ClassScore]
    confusion: tuple[tuple[int, ...], ...]
    neutral_false_positives: int
    neutral_verified: int
    neutral_clean: int
    duplicates: int
    incorrect: int
    unmapped_predictions: int
    verified_repetitions: int
    excluded_repetitions: int
    mean_latency_ms: float | None
    latency_samples: int
    mean_fps: float | None
    matched_pairs: int


def evaluate_repetitions(
    records: tuple[RepetitionRecord, ...] | list[RepetitionRecord],
    *,
    tolerance_ms: int = 400,
) -> EvaluationMetrics:
    if tolerance_ms < 0:
        raise ValueError("La tolerancia temporal no puede ser negativa.")
    verified = tuple(record for record in records if record.status == "verified")
    excluded = sum(record.status == "excluded" for record in records)
    predictions = [
        (record, mark)
        for record in verified
        for mark in record.predictions
    ]
    events = [
        record
        for record in verified
        if record.verified_label_id not in (None, 0)
        and record.window_start_ms is not None
        and record.window_end_ms is not None
    ]
    matches = _assign(events, predictions, tolerance_ms)
    return _summarize(verified, excluded, predictions, matches, tolerance_ms)


def _assign(
    events: list[RepetitionRecord],
    predictions: list[tuple[RepetitionRecord, PredictionMark]],
    tolerance_ms: int,
) -> dict[int, int]:
    """Devuelve predicción usada -> índice del evento. Cada lado se usa una vez."""

    candidates: list[tuple[int, float, int, int]] = []
    for event_index, event in enumerate(events):
        assert event.window_start_ms is not None and event.window_end_ms is not None
        center = (event.window_start_ms + event.window_end_ms) / 2
        for prediction_index, (_, mark) in enumerate(predictions):
            if mark.label_id is None:
                continue
            if not _inside(mark.timestamp_ms, event, tolerance_ms):
                continue
            agreement = 0 if mark.label_id == event.verified_label_id else 1
            distance = abs(mark.timestamp_ms - center)
            candidates.append((agreement, distance, event_index, prediction_index))
    candidates.sort()
    used_events: set[int] = set()
    used_predictions: set[int] = set()
    assignment: dict[int, int] = {}
    for _, _, event_index, prediction_index in candidates:
        if event_index in used_events or prediction_index in used_predictions:
            continue
        used_events.add(event_index)
        used_predictions.add(prediction_index)
        assignment[prediction_index] = event_index
    return assignment


def _summarize(
    verified: tuple[RepetitionRecord, ...],
    excluded: int,
    predictions: list[tuple[RepetitionRecord, PredictionMark]],
    assignment: dict[int, int],
    tolerance_ms: int,
) -> EvaluationMetrics:
    true_positives = {label: 0 for label in range(5)}
    false_positives = {label: 0 for label in range(5)}
    false_negatives = {label: 0 for label in range(5)}
    duplicates = {label: 0 for label in range(5)}
    confusion = [[0 for _ in range(6)] for _ in range(5)]
    latencies: list[float] = []
    neutral_false_positives = 0
    unmapped = 0
    incorrect = 0
    punch_events = [
        record
        for record in verified
        if record.verified_label_id not in (None, 0)
    ]
    matched_events = set(assignment.values())

    for prediction_index, (source, mark) in enumerate(predictions):
        event_index = assignment.get(prediction_index)
        if event_index is None:
            continue
        event = punch_events[event_index]
        predicted = mark.label_id
        truth = event.verified_label_id
        assert predicted is not None and truth is not None
        confusion[truth][predicted] += 1
        if predicted == truth:
            true_positives[truth] += 1
            if mark.latency_ms is not None:
                latencies.append(mark.latency_ms)
        else:
            false_negatives[truth] += 1
            false_positives[predicted] += 1
            incorrect += 1

    for event_index, event in enumerate(punch_events):
        if event_index not in matched_events:
            truth = event.verified_label_id
            assert truth is not None
            false_negatives[truth] += 1
            confusion[truth][MISS] += 1

    matched_prediction_indexes = set(assignment)
    for prediction_index, (_, mark) in enumerate(predictions):
        if prediction_index in matched_prediction_indexes:
            continue
        host = _host_repetition(mark, verified, tolerance_ms)
        if host is None or host.verified_label_id is None:
            continue
        if mark.label_id is None:
            unmapped += 1
            if host.verified_label_id == 0:
                neutral_false_positives += 1
            incorrect += 1
            continue
        if host.verified_label_id == 0:
            neutral_false_positives += 1
            false_positives[mark.label_id] += 1
            incorrect += 1
            continue
        if mark.label_id == host.verified_label_id and _event_was_matched(host, punch_events, matched_events):
            duplicates[mark.label_id] += 1
            false_positives[mark.label_id] += 1
        else:
            false_positives[mark.label_id] += 1
            incorrect += 1

    neutral_verified = sum(record.verified_label_id == 0 for record in verified)
    neutral_clean = 0
    for record in verified:
        if record.verified_label_id != 0:
            continue
        detected = [
            mark
            for mark in record.predictions
            if _inside(mark.timestamp_ms, record, tolerance_ms)
        ]
        if not detected:
            neutral_clean += 1
            true_positives[0] += 1
            confusion[0][0] += 1
        else:
            false_negatives[0] += 1
            sided = next((mark.label_id for mark in detected if mark.label_id is not None), None)
            confusion[0][MISS if sided is None else sided] += 1
    scores = {
        label: _score(
            label,
            _support(verified, label),
            true_positives[label],
            false_positives[label],
            false_negatives[label],
            duplicates[label],
        )
        for label in range(5)
    }
    fps_values = [value for record in verified for value in record.fps_samples]
    return EvaluationMetrics(
        scores=scores,
        confusion=tuple(tuple(row) for row in confusion),
        neutral_false_positives=neutral_false_positives,
        neutral_verified=neutral_verified,
        neutral_clean=neutral_clean,
        duplicates=sum(duplicates.values()),
        incorrect=incorrect,
        unmapped_predictions=unmapped,
        verified_repetitions=len(verified),
        excluded_repetitions=excluded,
        mean_latency_ms=None if not latencies else sum(latencies) / len(latencies),
        latency_samples=len(latencies),
        mean_fps=None if not fps_values else sum(fps_values) / len(fps_values),
        matched_pairs=len(assignment),
    )


def _support(records: tuple[RepetitionRecord, ...], label: int) -> int:
    return sum(record.verified_label_id == label for record in records)


def _score(
    label: int,
    support: int,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    duplicates: int,
) -> ClassScore:
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    if label == 0:
        precision = None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return ClassScore(
        label,
        support,
        true_positives,
        false_positives,
        false_negatives,
        duplicates,
        precision,
        recall,
        f1,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _inside(timestamp_ms: int, record: RepetitionRecord, tolerance_ms: int) -> bool:
    if record.window_start_ms is None or record.window_end_ms is None:
        return False
    return (
        record.window_start_ms - tolerance_ms
        <= timestamp_ms
        <= record.window_end_ms + tolerance_ms
    )


def _host_repetition(
    mark: PredictionMark,
    records: tuple[RepetitionRecord, ...],
    tolerance_ms: int,
) -> RepetitionRecord | None:
    """Una predicción se atribuye solo a la ventana verificada más cercana."""

    best: RepetitionRecord | None = None
    best_distance: float | None = None
    for record in records:
        if not _inside(mark.timestamp_ms, record, tolerance_ms):
            continue
        assert record.window_start_ms is not None and record.window_end_ms is not None
        center = (record.window_start_ms + record.window_end_ms) / 2
        distance = abs(mark.timestamp_ms - center)
        if best_distance is None or distance < best_distance:
            best = record
            best_distance = distance
    return best


def _event_was_matched(
    record: RepetitionRecord,
    events: list[RepetitionRecord],
    matched_events: set[int],
) -> bool:
    for index, event in enumerate(events):
        if event is record:
            return index in matched_events
    return False
