"""Revisión posterior. El detector no se convierte en etiqueta verificada."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

from app.config import STANCE_DESCRIPTION, AppConfig
from app.dataset_manager import (
    CameraMetadata,
    DatasetError,
    DatasetManager,
    DuplicateDatasetEntryError,
    FrameRecord,
    MOVEMENTS,
    SessionRecord,
    assemble_sample,
    sample_from_dict,
    sample_to_dict,
)
from app.sample_validator import validate_sample
from app.targets.capture import window_from_frames


@dataclass(frozen=True, slots=True)
class CaptureDraft:
    requested_label: int
    outcome: str
    detector_label: int | None
    detector_name: str
    frames: tuple[FrameRecord, ...]
    window_start_ms: int
    window_ms: int
    target_offset: tuple[float, float]
    radius: float
    game_score: int
    wrist_distance: float | None = None


class ReviewStore:
    """Guarda provisionales fuera del dataset que consume NEAT."""

    def __init__(self, data_dir: Path, session_id: str | None = None) -> None:
        self.session_id = session_id or str(uuid4())
        self.root = data_dir / "review" / self.session_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._dataset_opened = False
        self._cached_stamp: tuple[tuple[str, int], ...] | None = None
        self._cached_entries: list[tuple[Path, dict[str, object]]] | None = None

    def ingest(
        self,
        draft: CaptureDraft,
        config: AppConfig,
        *,
        width: int,
        height: int,
    ) -> dict[str, object]:
        frames, window_reasons = window_from_frames(
            draft.frames,
            draft.window_start_ms,
            draft.window_ms,
            min_frames=config.min_sample_frames,
            min_duration_ms=config.min_sample_duration_ms,
            max_duration_ms=config.max_sample_duration_ms,
        )
        sample = None
        quality = list(window_reasons)
        if frames and not window_reasons:
            sample = assemble_sample(
                session_id=self.session_id,
                label=MOVEMENTS[draft.requested_label],
                frames=frames,
                stance=config.stance,
                stance_description=STANCE_DESCRIPTION,
                camera=CameraMetadata(
                    index=config.camera_index,
                    requested_width=config.frame_width,
                    requested_height=config.frame_height,
                    actual_width=width,
                    actual_height=height,
                    mirror_view=config.mirror_view,
                ),
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            validation = validate_sample(sample, config)
            quality.extend(validation.reasons)
            if not validation.accepted:
                sample = None
        status = _status(draft.outcome, sample is not None)
        payload = {
            "verification_status": status,
            "requested_label_id": draft.requested_label,
            "requested_label_name": MOVEMENTS[draft.requested_label].name,
            "approved_label_id": None,
            "detector_label_id": draft.detector_label,
            "detector_display_name": draft.detector_name,
            "outcome": draft.outcome,
            "target": {
                "x": draft.target_offset[0],
                "y": draft.target_offset[1],
                "radius": draft.radius,
            },
            "quality_reasons": quality,
            "game_score": draft.game_score,
            "wrist_distance": draft.wrist_distance,
            "reason_text": reason_text(draft.outcome, quality),
            "sample": None if sample is None else sample_to_dict(sample),
        }
        path = self.root / f"{uuid4()}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    def entries(self) -> list[tuple[Path, dict[str, object]]]:
        paths = sorted(self.root.glob("*.json"))
        stamp = tuple((str(path), path.stat().st_mtime_ns) for path in paths)
        if stamp == self._cached_stamp and self._cached_entries is not None:
            return self._cached_entries
        found = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
        self._cached_stamp = stamp
        self._cached_entries = found
        return found

    def counts(self) -> dict[str, int]:
        summary = {"provisional": 0, "needs_review": 0, "invalid": 0, "verified": 0, "discarded": 0}
        for _, payload in self.entries():
            status = str(payload.get("verification_status"))
            if status in summary:
                summary[status] += 1
        return summary

    def verified_counts(self) -> dict[int, int]:
        counts = {label: 0 for label in MOVEMENTS}
        for _, payload in self.entries():
            if payload.get("verification_status") != "verified":
                continue
            label = payload.get("approved_label_id")
            if label is None:
                label = payload.get("requested_label_id")
            if isinstance(label, int) and label in counts:
                counts[label] += 1
        return counts

    def correct(self, path: Path, label_id: int) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("verification_status") in {"verified", "discarded", "invalid"}:
            return
        payload["approved_label_id"] = MOVEMENTS[label_id].label_id
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def discard(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("verification_status") == "verified":
            return
        payload["verification_status"] = "discarded"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def approve(self, path: Path, dataset: DatasetManager, config: AppConfig) -> bool:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("verification_status") not in {"provisional", "needs_review"}:
            return False
        label_id = payload.get("approved_label_id")
        if not isinstance(label_id, int):
            return False
        sample_payload = payload.get("sample")
        if not isinstance(sample_payload, dict):
            return False
        sample = sample_from_dict(sample_payload)
        movement = MOVEMENTS[label_id]
        sample = replace(sample, label_id=movement.label_id, label_name=movement.name)
        validation = validate_sample(sample, config)
        if not validation.accepted:
            payload["quality_reasons"] = list(validation.reasons)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return False
        self._ensure_session(dataset, sample.camera, config)
        try:
            dataset.save_sample(sample)
        except DuplicateDatasetEntryError:
            return False
        payload["verification_status"] = "verified"
        payload["approved_label_id"] = movement.label_id
        payload["sample"] = sample_to_dict(sample)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True

    def relabel_verified(
        self,
        path: Path,
        label_id: int,
        dataset: DatasetManager,
        config: AppConfig,
    ) -> bool:
        """Corrige una muestra ya verificada. Conserva el identificador y la predicción del detector."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("verification_status") != "verified":
            return False
        sample_payload = payload.get("sample")
        if not isinstance(sample_payload, dict) or not isinstance(sample_payload.get("sample_id"), str):
            return False
        updated = dataset.relabel_sample(sample_payload["sample_id"], label_id, config)
        payload["approved_label_id"] = updated.label_id
        payload["sample"] = sample_to_dict(updated)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True

    def approve_provisional(self, dataset: DatasetManager, config: AppConfig) -> int:
        saved = 0
        for path, payload in self.entries():
            if payload.get("verification_status") != "provisional":
                continue
            if self.approve(path, dataset, config):
                saved += 1
        return saved

    def _ensure_session(
        self,
        dataset: DatasetManager,
        camera: CameraMetadata,
        config: AppConfig,
    ) -> None:
        if self._dataset_opened:
            return
        session_path = dataset.session_path(self.session_id)
        if not session_path.is_file():
            dataset.create_session(
                SessionRecord(
                    session_id=self.session_id,
                    started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    stance=config.stance,
                    stance_description=STANCE_DESCRIPTION,
                    camera=camera,
                    saved_sample_count=0,
                    sample_ids=(),
                )
            )
        self._dataset_opened = True


def list_review_sessions(data_dir: Path) -> list[dict[str, object]]:
    """Lee sesiones ya guardadas. No borra ni reescribe sus capturas."""

    root = data_dir / "review"
    if not root.is_dir():
        return []
    found: list[dict[str, object]] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        captures = list(folder.glob("*.json"))
        if not captures:
            continue
        store = ReviewStore(data_dir, folder.name)
        counts = store.counts()
        found.append(
            {
                "session_id": folder.name,
                "captures": len(captures),
                **counts,
            }
        )
    return found


def open_review_session(data_dir: Path, session_id: str) -> ReviewStore:
    folder = data_dir / "review" / session_id
    if not folder.is_dir() or not any(folder.glob("*.json")):
        raise FileNotFoundError(f"No hay capturas guardadas para la sesión {session_id}.")
    return ReviewStore(data_dir, session_id)


def reason_text(outcome: str, quality_reasons: list[str] | tuple[str, ...] = ()) -> str:
    """Explica el estado sin convertir la predicción del detector en verdad."""

    if quality_reasons:
        return "Problema de la captura: " + "; ".join(quality_reasons)
    return {
        "acierto": (
            "El detector coincidió con lo pedido y la muñeca entró al objetivo. "
            "Sigue pendiente de tu confirmación."
        ),
        "guardia": (
            "No hubo un golpe durante la guardia pedida. Sigue pendiente de tu confirmación."
        ),
        "equivocado": "Golpe reconocido con una clase distinta a la pedida.",
        "lejos": "La clase coincidió, pero la muñeca quedó fuera del objetivo.",
        "fallo": "Golpe no reconocido: el tiempo terminó sin un evento completo.",
        "no_identificado": "Hubo un evento, pero el detector no lo clasificó.",
        "guardia_rota": "Se detectó un golpe durante la guardia pedida.",
        "duplicado": "Evento duplicado. No se contabilizó otra vez.",
        "sin_seguimiento": "Se perdió el seguimiento corporal.",
    }.get(outcome, "Captura que necesita verificación humana.")


def capture_details(payload: dict[str, object]) -> tuple[str, ...]:
    """Resume la biomecánica disponible. No existe un video para reproducir."""

    outcome = str(payload.get("outcome") or "")
    quality = payload.get("quality_reasons")
    reasons = quality if isinstance(quality, list) else []
    stored_reason = payload.get("reason_text")
    reason = stored_reason if isinstance(stored_reason, str) and stored_reason else reason_text(outcome, reasons)
    sample = payload.get("sample")
    frames = sample.get("frames") if isinstance(sample, dict) else None
    frame_list = frames if isinstance(frames, list) else []
    duration = sample.get("duration_ms") if isinstance(sample, dict) else None
    distance = payload.get("wrist_distance")
    distance_text = "no se guardó en esta captura" if not isinstance(distance, (int, float)) else f"{distance:.2f} anchos de hombro"
    chosen = payload.get("approved_label_id")
    chosen_text = "sin elegir" if not isinstance(chosen, int) else MOVEMENTS[chosen].display_name
    return (
        reason,
        f"Pedido: {payload.get('requested_label_name')} | Detector: {payload.get('detector_display_name') or 'sin evento'}",
        f"Estado: {payload.get('verification_status')} | Etiqueta elegida: {chosen_text}",
        f"Duración: {duration} ms | Fotogramas: {len(frame_list)} | Distancia al objetivo: {distance_text}",
        f"Codo izquierdo {_feature_span(frame_list, 'left_elbow_angle')} | Codo derecho {_feature_span(frame_list, 'right_elbow_angle')}",
        f"Alcance izquierdo {_feature_span(frame_list, 'left_shoulder_wrist_distance')} | Alcance derecho {_feature_span(frame_list, 'right_shoulder_wrist_distance')}",
        "No hay video ni imagen: no se puede reproducir cómo se vio el golpe.",
    )


def dataset_gap_lines(data_dir: Path, goal: int) -> tuple[str, ...]:
    """Cuenta solo muestras ya guardadas en el dataset de NEAT."""

    manager = DatasetManager(data_dir)
    try:
        rows = manager.load_index()
    except (OSError, DatasetError, KeyError, ValueError):
        rows = []
    counts = {label: 0 for label in MOVEMENTS}
    sessions: set[str] = set()
    for row in rows:
        try:
            counts[int(row["label_id"])] += 1
            sessions.add(row["session_id"])
        except (KeyError, TypeError, ValueError):
            continue
    lines = [
        f"Dataset NEAT: {len(rows)} muestras verificadas en {len(sessions)} sesiones.",
    ]
    missing = []
    for label, movement in MOVEMENTS.items():
        lines.append(f"{movement.display_name}: {counts[label]} verificadas, meta {goal}.")
        if counts[label] < 1:
            missing.append(movement.display_name)
    if len(sessions) < 3:
        lines.append(f"Faltan sesiones para entrenar: hay {len(sessions)} y se necesitan al menos 3.")
    if missing:
        lines.append("Faltan clases en el dataset: " + ", ".join(missing) + ".")
    elif len(sessions) >= 3:
        lines.append("El mínimo para intentar dataset-report ya está en el dataset.")
    return tuple(lines)


def _feature_span(frames: list[object], key: str) -> str:
    values: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        features = frame.get("features")
        if not isinstance(features, dict):
            continue
        value = features.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return "sin dato"
    return f"{min(values):.0f}-{max(values):.0f}"


def _status(outcome: str, sample_ok: bool) -> str:
    if not sample_ok:
        return "invalid"
    return "needs_review"
