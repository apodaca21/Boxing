"""Rutina de calibración individual basada en una webcam y revisión humana.

La rutina solo propone parámetros. Nunca modifica el clasificador en memoria,
no convierte la consigna en verdad y no crea muestras para NEAT.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import json
import math
from pathlib import Path
from uuid import uuid4

from app.dataset_manager import MOVEMENTS
from app.trainer.heuristic import HeuristicSettings


@dataclass(frozen=True, slots=True)
class CalibrationStep:
    key: str
    title: str
    instruction: str
    label_id: int
    repetitions: int
    active_ms: int


CALIBRATION_STEPS = (
    CalibrationStep("guard", "Guardia inmóvil", "Mantén la guardia sin golpear.", 0, 1, 5000),
    CalibrationStep("smooth", "Movimiento suave", "Mueve ambos brazos suavemente, sin golpear.", 0, 1, 5000),
    CalibrationStep("jab", "Jab izquierdo", "Ejecuta un jab izquierdo y vuelve a guardia.", 1, 5, 2200),
    CalibrationStep("cross", "Cross derecho", "Ejecuta un cross derecho y vuelve a guardia.", 2, 5, 2200),
    CalibrationStep("left_hook", "Gancho izquierdo", "Ejecuta un gancho izquierdo y vuelve a guardia.", 3, 5, 2200),
    CalibrationStep("right_hook", "Gancho derecho", "Ejecuta un gancho derecho y vuelve a guardia.", 4, 5, 2200),
)


@dataclass
class CalibrationRecord:
    step_key: str
    phase_name: str
    requested_label_id: int
    repetition_index: int
    window_start_ms: int | None = None
    window_end_ms: int | None = None
    status: str = "pending"
    verified_label_id: int | None = None
    corrected: bool = False
    exclusion_reason: str | None = None
    observations: list[dict[str, object]] = field(default_factory=list)
    predictions: list[dict[str, object]] = field(default_factory=list)

    @property
    def requested_name(self) -> str:
        return MOVEMENTS[self.requested_label_id].display_name

    @property
    def verified_name(self) -> str | None:
        if self.verified_label_id is None:
            return None
        return MOVEMENTS[self.verified_label_id].display_name


@dataclass(frozen=True, slots=True)
class CalibrationView:
    phase: str
    instruction: str
    detail: str
    progress: str
    review: bool
    finished: bool


class CalibrationProtocol:
    """Temporizador de calibración con una decisión humana por intento."""

    def __init__(
        self,
        *,
        prepare_ms: int = 2500,
        return_ms: int = 1000,
        next_ms: int = 900,
        steps: tuple[CalibrationStep, ...] = CALIBRATION_STEPS,
    ) -> None:
        if prepare_ms < 0 or return_ms < 0 or next_ms < 0:
            raise ValueError("Los intervalos de calibración no pueden ser negativos.")
        if not steps:
            raise ValueError("La calibración necesita al menos una fase.")
        self.prepare_ms = prepare_ms
        self.return_ms = return_ms
        self.next_ms = next_ms
        self.steps = steps
        self.phase = "preflight"
        self._step_index = 0
        self._rep_cursor = 0
        self._phase_started_ms: int | None = None
        self._current: CalibrationRecord | None = None
        self._pending_correction: int | None = None
        self.records: list[CalibrationRecord] = []

    @property
    def current_step(self) -> CalibrationStep:
        return self.steps[self._step_index]

    @property
    def current(self) -> CalibrationRecord | None:
        return self._current

    @property
    def finished(self) -> bool:
        return self.phase == "finished"

    @property
    def started(self) -> bool:
        return self.phase != "preflight"

    def start(self, timestamp_ms: int) -> None:
        if self.started:
            return
        self.phase = "prepare"
        self._phase_started_ms = timestamp_ms
        self._begin_record()

    def advance(self, timestamp_ms: int) -> bool:
        if self.phase in {"preflight", "review", "finished"}:
            return False
        if self._phase_started_ms is None:
            self._phase_started_ms = timestamp_ms
            return True
        elapsed = timestamp_ms - self._phase_started_ms
        if self.phase == "prepare" and elapsed >= self.prepare_ms:
            self.phase = "execute"
            self._phase_started_ms = timestamp_ms
            if self._current is not None:
                self._current.window_start_ms = timestamp_ms
            return True
        if self.phase == "execute" and elapsed >= self.current_step.active_ms:
            self.phase = "return_guard"
            self._phase_started_ms = timestamp_ms
            if self._current is not None:
                self._current.window_end_ms = timestamp_ms
            return True
        if self.phase == "return_guard" and elapsed >= self.return_ms:
            self.phase = "review"
            self._phase_started_ms = timestamp_ms
            return True
        if self.phase == "next" and elapsed >= self.next_ms:
            self._advance_schedule(timestamp_ms)
            return True
        return False

    def observe(
        self,
        timestamp_ms: int,
        evidence: dict[str, object],
        *,
        events: tuple[object, ...] = (),
    ) -> None:
        if self.phase != "execute" or self._current is None:
            return
        item = dict(evidence)
        item["timestamp_ms"] = timestamp_ms
        self._current.observations.append(item)
        for event in events:
            self._current.predictions.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "label_id": getattr(event, "label_id", None),
                    "display_name": getattr(event, "display_name", ""),
                    "arm": getattr(event, "arm", None),
                    "event_start_ms": item.get("event_start_ms"),
                    "event_end_ms": item.get("event_end_ms"),
                    "event_duration_ms": _event_duration(item),
                }
            )

    def correct(self, label_id: int) -> bool:
        if self.phase != "review" or self._current is None:
            return False
        if self._current.requested_label_id == 0:
            return False
        if label_id not in MOVEMENTS:
            return False
        self._pending_correction = label_id
        return True

    def confirm(self) -> CalibrationRecord | None:
        if self.phase != "review" or self._current is None:
            return None
        requested = self._current.requested_label_id
        verified = requested if self._pending_correction is None else self._pending_correction
        self._current.status = "verified"
        self._current.verified_label_id = verified
        self._current.corrected = verified != requested
        record = self._current
        self._pending_correction = None
        self._enter_next_or_finish(record.window_end_ms or 0)
        return record

    def exclude(self, reason: str = "excluido_por_usuario") -> CalibrationRecord | None:
        if self.phase != "review" or self._current is None:
            return None
        self._current.status = "excluded"
        self._current.exclusion_reason = reason
        self._current.verified_label_id = None
        record = self._current
        self._pending_correction = None
        self._enter_next_or_finish(record.window_end_ms or 0)
        return record

    def repeat(self, timestamp_ms: int, reason: str = "repetido_por_usuario") -> CalibrationRecord | None:
        if self.phase != "review" or self._current is None:
            return None
        old = self._current
        old.status = "repeated"
        old.exclusion_reason = reason
        self._current = CalibrationRecord(
            step_key=old.step_key,
            phase_name=old.phase_name,
            requested_label_id=old.requested_label_id,
            repetition_index=old.repetition_index,
        )
        self.records.append(self._current)
        self._pending_correction = None
        self.phase = "prepare"
        self._phase_started_ms = timestamp_ms
        return self._current

    def view(self) -> CalibrationView:
        if self.phase == "preflight":
            return CalibrationView(
                self.phase,
                "Preflight de camara",
                "Selecciona F frontal o A para 30-45 grados. ENTER inicia.",
                "Antes de calibrar",
                False,
                False,
            )
        if self.phase == "finished":
            return CalibrationView(
                self.phase,
                "Calibracion terminada",
                "Se guardaron mediciones y una propuesta; no se cambiaron umbrales automaticamente.",
                f"Intentos registrados: {len(self.records)}",
                False,
                True,
            )
        step = self.current_step
        progress = f"{step.title}  {self._rep_cursor + 1}/{step.repetitions}"
        if self.phase == "prepare":
            return CalibrationView(self.phase, "Preparacion", step.instruction, progress, False, False)
        if self.phase == "execute":
            return CalibrationView(self.phase, step.instruction, "Intervalo activo: ejecuta solo lo indicado.", progress, False, False)
        if self.phase == "return_guard":
            return CalibrationView(self.phase, "Regresa a guardia", "Espera la revision humana.", progress, False, False)
        if self.phase == "next":
            return CalibrationView(self.phase, "Siguiente intento", "La medicion anterior espera una confirmacion.", progress, False, False)
        chosen = self._current.requested_name if self._pending_correction is None else MOVEMENTS[self._pending_correction].display_name
        if step.label_id == 0:
            detail = "Y/ENTER confirma que completaste la fase. R repite, X excluye."
        else:
            detail = f"Solicitado: {self._current.requested_name}. Elegido: {chosen}. Y/ENTER confirma, 0-4 corrige, R repite, X excluye."
        return CalibrationView("review", "Revisa tu ejecucion", detail, progress, True, False)

    def _begin_record(self) -> None:
        step = self.current_step
        self._current = CalibrationRecord(
            step_key=step.key,
            phase_name=step.title,
            requested_label_id=step.label_id,
            repetition_index=self._rep_cursor + 1,
        )
        self.records.append(self._current)
        self._pending_correction = None

    def _enter_next_or_finish(self, timestamp_ms: int) -> None:
        step = self.current_step
        if self._rep_cursor + 1 < step.repetitions:
            self.phase = "next"
            self._phase_started_ms = timestamp_ms
            return
        if self._step_index + 1 >= len(self.steps):
            self.phase = "finished"
            self._phase_started_ms = timestamp_ms
            return
        self.phase = "next"
        self._phase_started_ms = timestamp_ms

    def _advance_schedule(self, timestamp_ms: int) -> None:
        step = self.current_step
        if self._rep_cursor + 1 < step.repetitions:
            self._rep_cursor += 1
        else:
            self._step_index += 1
            self._rep_cursor = 0
        self._begin_record()
        self.phase = "prepare"
        self._phase_started_ms = timestamp_ms


def summarize_record(record: CalibrationRecord) -> dict[str, object]:
    """Resume un intento y conserva solo lo observado; no rellena ausencias."""

    observations = record.observations
    result: dict[str, object] = {
        "step_key": record.step_key,
        "phase_name": record.phase_name,
        "requested_label_id": record.requested_label_id,
        "requested_name": record.requested_name,
        "repetition_index": record.repetition_index,
        "status": record.status,
        "verified_label_id": record.verified_label_id,
        "verified_name": record.verified_name,
        "corrected": record.corrected,
        "exclusion_reason": record.exclusion_reason,
        "window_start_ms": record.window_start_ms,
        "window_end_ms": record.window_end_ms,
        "duration_ms": _duration(record),
        "prediction_count": len(record.predictions),
        "predicted_labels": [item.get("label_id") for item in record.predictions],
        "predicted_arms": [item.get("arm") for item in record.predictions],
    }
    for side in ("left", "right"):
        speeds = _values(observations, f"{side}_speed")
        extensions = _values(observations, f"{side}_extension")
        lateral = _values(observations, f"{side}_lateral")
        angles = _values(observations, f"{side}_elbow_angle")
        quality = _values(observations, f"{side}_arm_tracking_quality")
        xs = _values(observations, f"{side}_wrist_x")
        ys = _values(observations, f"{side}_wrist_y")
        result.update(
            {
                f"{side}_tracking_quality_min": _min_or_none(quality),
                f"{side}_tracking_quality_mean": _mean_or_none(quality),
                f"{side}_speed_max": _max_or_none(speeds),
                f"{side}_speed_p95": _percentile(speeds, 0.95),
                f"{side}_extension_range": _range_or_none(extensions),
                f"{side}_lateral_range": _range_or_none(lateral),
                f"{side}_elbow_angle_min": _min_or_none(angles),
                f"{side}_elbow_angle_max": _max_or_none(angles),
                f"{side}_elbow_angle_change": _range_or_none(angles),
                f"{side}_wrist_displacement_x": _range_or_none(xs),
                f"{side}_wrist_displacement_y": _range_or_none(ys),
                f"{side}_wrist_path": _path_length(xs, ys),
            }
        )
    tracking = _values(observations, "required_landmarks_quality")
    fps = _values(observations, "fps")
    result.update(
        {
            "tracking_quality_min": _min_or_none(tracking),
            "tracking_quality_mean": _mean_or_none(tracking),
            "fps_mean": _mean_or_none(fps),
            "pose_missing_frames": sum(not bool(item.get("pose_detected", False)) for item in observations),
            "cooldown_frames": sum(bool(item.get("cooldown_active", False)) for item in observations),
            "rejections": sorted(
                {
                    str(item.get("rejection"))
                    for item in observations
                    if item.get("rejection")
                }
            ),
            "observed_event_durations_ms": [
                item.get("event_duration_ms")
                for item in record.predictions
                if item.get("event_duration_ms") is not None
            ],
        }
    )
    return result


def propose_settings(
    records: list[CalibrationRecord],
    current: HeuristicSettings,
) -> tuple[HeuristicSettings, dict[str, object]]:
    """Calcula una propuesta explícita desde intentos humanos válidos.

    Los cuantiles y fórmulas se guardan junto a la propuesta. Si no hay
    separación entre ruido y golpes, se conserva el valor actual y se informa
    que la evidencia fue insuficiente.
    """

    valid = [
        record
        for record in records
        if record.status == "verified"
        and record.requested_label_id in (1, 2, 3, 4)
        and record.verified_label_id == record.requested_label_id
    ]
    baseline = [
        item
        for record in records
        if record.status == "verified" and record.step_key in {"guard", "smooth"}
        for item in record.observations
    ]
    speed_noise = _percentile(
        [value for item in baseline for value in _values((item,), "left_speed") + _values((item,), "right_speed")],
        0.95,
    )
    punch_peaks = [
        float(summary[f"{_side_for_label(record.requested_label_id)}_speed_max"])
        for record in valid
        for summary in (summarize_record(record),)
        if isinstance(summary.get(f"{_side_for_label(record.requested_label_id)}_speed_max"), (int, float))
    ]
    speed_signal = _percentile([float(value) for value in punch_peaks], 0.10)
    proposal = current
    formulas: dict[str, object] = {}
    notes: list[str] = []
    if speed_noise is not None and speed_signal is not None and speed_signal > speed_noise:
        candidate = (speed_noise + speed_signal) / 2.0
        proposal = replace(proposal, min_wrist_speed=round(max(0.05, candidate), 5))
        formulas["min_wrist_speed"] = {
            "formula": "midpoint(guard_smooth_speed_p95, verified_punch_speed_p10)",
            "guard_smooth_speed_p95": speed_noise,
            "verified_punch_speed_p10": speed_signal,
        }
    else:
        notes.append("sin_separacion_suficiente_entre_guardia_y_golpe_para_velocidad")

    straight = [
        summarize_record(record)
        for record in valid
        if record.requested_label_id in (1, 2)
    ]
    hooks = [
        summarize_record(record)
        for record in valid
        if record.requested_label_id in (3, 4)
    ]
    straight_ranges = [
        float(item[f"{side}_extension_range"])
        for record, item in ((record, summarize_record(record)) for record in valid if record.requested_label_id in (1, 2))
        for side in (_side_for_label(record.requested_label_id),)
        if isinstance(item.get(f"{side}_extension_range"), (int, float))
    ]
    hook_ranges = [
        float(item[f"{side}_lateral_range"])
        for record, item in ((record, summarize_record(record)) for record in valid if record.requested_label_id in (3, 4))
        for side in (_side_for_label(record.requested_label_id),)
        if isinstance(item.get(f"{side}_lateral_range"), (int, float))
    ]
    if straight_ranges:
        value = _percentile(straight_ranges, 0.10)
        if value is not None and value > 0:
            proposal = replace(proposal, straight_gain=round(value, 5))
            formulas["straight_gain"] = {"formula": "verified_straight_extension_range_p10", "value": value}
    else:
        notes.append("sin_intentos_rectos_verificados")
    if hook_ranges:
        value = _percentile(hook_ranges, 0.10)
        if value is not None and value > 0:
            proposal = replace(proposal, hook_lateral=round(value, 5))
            formulas["hook_lateral"] = {"formula": "verified_hook_lateral_range_p10", "value": value}
    else:
        notes.append("sin_intentos_de_gancho_verificados")

    return proposal, {
        "valid_punch_attempts": len(valid),
        "baseline_attempts": sum(record.status == "verified" and record.step_key in {"guard", "smooth"} for record in records),
        "speed_noise_p95": speed_noise,
        "punch_speed_p10": speed_signal,
        "formulas": formulas,
        "notes": notes,
        "automatic_application": False,
    }


def _side_for_label(label_id: int) -> str:
    return "left" if label_id in (1, 3) else "right"


def save_calibration(
    results_root: Path,
    *,
    orientation: str,
    preflight_rows: list[dict[str, object]],
    records: list[CalibrationRecord],
    current: HeuristicSettings,
) -> Path:
    """Guarda JSON, intentos, visibilidad y parámetros sin sobrescribir sesiones."""

    created_at = datetime.now().astimezone()
    calibration_id = created_at.strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    directory = results_root / "calibrations" / calibration_id
    directory.mkdir(parents=True, exist_ok=False)
    proposed, analysis = propose_settings(records, current)
    payload = {
        "calibration_id": calibration_id,
        "created_at": created_at.isoformat(timespec="seconds"),
        "orientation": orientation,
        "stance": "orthodox",
        "classifier": "heuristic",
        "contains_video": False,
        "contains_images": False,
        "note": "Propuesta derivada de una webcam y confirmaciones humanas; no es verdad de entrenamiento.",
        "current_settings_backup": asdict(current),
        "proposed_settings": asdict(proposed),
        "analysis": analysis,
        "records": [
            {
                **summarize_record(record),
                "observations": record.observations,
                "predictions": record.predictions,
            }
            for record in records
        ],
        "preflight_frames": len(preflight_rows),
    }
    (directory / "calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "params.json").write_text(
        json.dumps(
            {
                "calibration_id": calibration_id,
                "current_settings_backup": asdict(current),
                "proposed_settings": asdict(proposed),
                "analysis": analysis,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_attempts_csv(directory / "attempts.csv", records)
    _write_preflight_csv(directory / "visibility.csv", preflight_rows)
    index_path = results_root / "calibrations" / "index.csv"
    _append_index(index_path, calibration_id, created_at.isoformat(timespec="seconds"), orientation, records, analysis)
    update_calibration_comparison(results_root)
    return directory


def load_proposed_settings(path: Path) -> HeuristicSettings:
    """Carga solo una propuesta revisada explícitamente por la persona."""

    candidate = path / "calibration.json" if path.is_dir() else path
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    values = payload.get("proposed_settings")
    if not isinstance(values, dict):
        raise ValueError(f"No hay proposed_settings en la calibración: {candidate}")
    allowed = set(HeuristicSettings.__dataclass_fields__)
    kwargs = {key: value for key, value in values.items() if key in allowed}
    return HeuristicSettings(**kwargs)


def update_calibration_comparison(results_root: Path) -> Path | None:
    """Compara orientaciones solo si existen sesiones reales de ambas."""

    root = results_root / "calibrations"
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/calibration.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = []
        for record in payload.get("records", ()):
            frames.extend(record.get("observations", ()))
        quality = [
            float(item["required_landmarks_quality"])
            for item in frames
            if isinstance(item.get("required_landmarks_quality"), (int, float))
        ]
        left_quality = [
            float(item["left_arm_tracking_quality"])
            for item in frames
            if isinstance(item.get("left_arm_tracking_quality"), (int, float))
        ]
        right_quality = [
            float(item["right_arm_tracking_quality"])
            for item in frames
            if isinstance(item.get("right_arm_tracking_quality"), (int, float))
        ]
        rows.append(
            {
                "calibration_id": payload.get("calibration_id", path.parent.name),
                "orientation": payload.get("orientation", "no_especificada"),
                "frames": len(frames),
                "required_landmarks_quality_mean": _mean_or_none(quality),
                "left_arm_tracking_quality_mean": _mean_or_none(left_quality),
                "right_arm_tracking_quality_mean": _mean_or_none(right_quality),
                "body_visible_ratio": (
                    sum(bool(item.get("body_visible")) for item in frames) / len(frames)
                    if frames
                    else None
                ),
                "fps_mean": _mean_or_none(
                    [float(item["fps"]) for item in frames if isinstance(item.get("fps"), (int, float))]
                ),
                "verified_punch_attempts": payload.get("analysis", {}).get("valid_punch_attempts", 0),
            }
        )
    if not rows:
        return None
    orientations = {str(row["orientation"]) for row in rows}
    comparable = {orientation: [row for row in rows if row["orientation"] == orientation] for orientation in orientations}
    means = {
        orientation: _mean_or_none(
            [float(row["required_landmarks_quality_mean"]) for row in items if isinstance(row.get("required_landmarks_quality_mean"), (int, float))]
        )
        for orientation, items in comparable.items()
    }
    best = None
    if {"frontal", "angulo_30_45"}.issubset(orientations):
        front = means.get("frontal")
        angled = means.get("angulo_30_45")
        if front is not None and angled is not None and front != angled:
            best = "frontal" if front > angled else "angulo_30_45"
    output = {
        "note": "La orientación se compara por calidad de landmarks; la lateralidad siempre es anatómica y se mantiene separada por brazo.",
        "comparable": {"frontal": "frontal" in orientations, "angulo_30_45": "angulo_30_45" in orientations},
        "better_tracking_orientation": best,
        "sessions": rows,
        "means_by_orientation": means,
    }
    destination = root / "comparison.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = root / "comparison.csv"
    fields = tuple(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _write_attempts_csv(path: Path, records: list[CalibrationRecord]) -> None:
    rows = [summarize_record(record) for record in records]
    fields = (
        "step_key", "phase_name", "requested_label_id", "requested_name", "repetition_index",
        "status", "verified_label_id", "verified_name", "corrected", "window_start_ms",
        "window_end_ms", "duration_ms", "prediction_count", "predicted_labels", "predicted_arms",
        "tracking_quality_min", "tracking_quality_mean", "fps_mean", "pose_missing_frames",
        "cooldown_frames", "rejections", "left_tracking_quality_min", "right_tracking_quality_min",
        "left_speed_max", "right_speed_max", "left_extension_range", "right_extension_range",
        "left_lateral_range", "right_lateral_range", "left_elbow_angle_min", "right_elbow_angle_min",
        "left_elbow_angle_change", "right_elbow_angle_change", "left_wrist_displacement_x",
        "right_wrist_displacement_x", "left_wrist_displacement_y", "right_wrist_displacement_y",
        "left_wrist_path", "right_wrist_path", "observed_event_durations_ms",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _write_preflight_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "timestamp_ms", "orientation", "pose_detected", "body_visible", "required_landmarks_quality",
        "left_arm_tracking_quality", "right_arm_tracking_quality", "torso_tracking_quality",
        "left_elbow_angle", "right_elbow_angle", "fps",
        "landmark_left_shoulder", "landmark_right_shoulder", "landmark_left_elbow",
        "landmark_right_elbow", "landmark_left_wrist", "landmark_right_wrist",
        "landmark_left_hip", "landmark_right_hip",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _append_index(path: Path, calibration_id: str, created_at: str, orientation: str, records: list[CalibrationRecord], analysis: dict[str, object]) -> None:
    fields = ("calibration_id", "created_at", "orientation", "attempts", "verified_punch_attempts", "path")
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "calibration_id": calibration_id,
                "created_at": created_at,
                "orientation": orientation,
                "attempts": len(records),
                "verified_punch_attempts": analysis.get("valid_punch_attempts", 0),
                "path": str(path.parent / calibration_id),
            }
        )


def _event_duration(item: dict[str, object]) -> int | None:
    start = item.get("event_start_ms")
    end = item.get("event_end_ms")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
        return int(end - start)
    return None


def _duration(record: CalibrationRecord) -> int | None:
    if record.window_start_ms is None or record.window_end_ms is None:
        return None
    return max(0, record.window_end_ms - record.window_start_ms)


def _values(items: tuple[dict[str, object], ...] | list[dict[str, object]], key: str) -> list[float]:
    return [
        float(item[key])
        for item in items
        if isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool)
        and math.isfinite(float(item[key]))
    ]


def _min_or_none(values: list[float]) -> float | None:
    return None if not values else min(values)


def _max_or_none(values: list[float]) -> float | None:
    return None if not values else max(values)


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _range_or_none(values: list[float]) -> float | None:
    return None if not values else max(values) - min(values)


def _path_length(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return sum(math.hypot(xs[index] - xs[index - 1], ys[index] - ys[index - 1]) for index in range(1, len(xs)))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value
