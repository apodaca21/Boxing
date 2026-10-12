"""Contratos y persistencia local de muestras biomecánicas."""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from app.config import AppConfig


SCHEMA_VERSION = 1
SPLIT_UNIT = "sample"
COORDINATE_SPACE = "anatomical_normalized"
TRACKED_LANDMARK_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
INDEX_COLUMNS = (
    "sample_id",
    "session_id",
    "label_id",
    "label_name",
    "created_at",
    "duration_ms",
    "frame_count",
    "approximate_fps",
    "stance",
    "relative_path",
    "schema_version",
)
_IDENTIFIER = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class DatasetError(RuntimeError):
    """Error de lectura o escritura del dataset local."""


class DuplicateDatasetEntryError(DatasetError):
    """La muestra o la sesión ya existe y no debe sobrescribirse."""


@dataclass(frozen=True, slots=True)
class MovementLabel:
    """Clase elegida manualmente por la persona usuaria."""

    label_id: int
    name: str
    display_name: str


MOVEMENTS: dict[int, MovementLabel] = {
    0: MovementLabel(0, "guardia_neutral", "Guardia / Neutral"),
    1: MovementLabel(1, "jab_izquierdo", "Jab izquierdo"),
    2: MovementLabel(2, "cross_derecho", "Cross derecho"),
    3: MovementLabel(3, "gancho_izquierdo", "Gancho izquierdo"),
    4: MovementLabel(4, "gancho_derecho", "Gancho derecho"),
}


def movement_by_id(label_id: int) -> MovementLabel:
    try:
        return MOVEMENTS[label_id]
    except KeyError as error:
        raise ValueError(
            "Etiqueta de movimiento no válida: "
            f"{label_id}. Usa un número del 0 al 4."
        ) from error


@dataclass(frozen=True, slots=True)
class PlaneCoordinate:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class LandmarkRecord:
    x: float | None
    y: float | None
    z: float | None
    visibility: float | None
    presence: float | None
    reliable: bool


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    left_elbow_angle: float | None
    right_elbow_angle: float | None
    left_shoulder_wrist_distance: float | None
    right_shoulder_wrist_distance: float | None
    left_wrist_relative: PlaneCoordinate | None
    right_wrist_relative: PlaneCoordinate | None
    wrist_separation: float | None

    @classmethod
    def missing(cls) -> FeatureRecord:
        return cls(None, None, None, None, None, None, None)


@dataclass(frozen=True, slots=True)
class FrameRecord:
    timestamp_ms: int
    elapsed_ms: int
    pose_detected: bool
    landmarks: dict[str, LandmarkRecord | None]
    features: FeatureRecord


@dataclass(frozen=True, slots=True)
class CameraMetadata:
    index: int
    requested_width: int
    requested_height: int
    actual_width: int | None
    actual_height: int | None
    mirror_view: bool
    coordinate_space: str = COORDINATE_SPACE


@dataclass(frozen=True, slots=True)
class MovementSample:
    schema_version: int
    split_unit: str
    sample_id: str
    session_id: str
    label_id: int
    label_name: str
    created_at: str
    duration_ms: int
    frame_count: int
    approximate_fps: float | None
    stance: str
    stance_description: str
    camera: CameraMetadata
    contains_images: bool
    contains_audio: bool
    contains_video: bool
    frames: tuple[FrameRecord, ...]


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    started_at: str
    stance: str
    stance_description: str
    camera: CameraMetadata
    saved_sample_count: int
    sample_ids: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION


def assemble_sample(
    *,
    session_id: str,
    label: MovementLabel,
    frames: tuple[FrameRecord, ...],
    stance: str,
    stance_description: str,
    camera: CameraMetadata,
    created_at: str,
    sample_id: str | None = None,
) -> MovementSample:
    """Construye una ejecución completa con metadatos temporales coherentes."""

    if not frames:
        raise ValueError("No hay fotogramas para construir la muestra.")
    origin = frames[0].timestamp_ms
    normalized = tuple(
        replace(frame, elapsed_ms=frame.timestamp_ms - origin) for frame in frames
    )
    duration_ms = normalized[-1].timestamp_ms - origin
    approximate_fps = (
        None
        if duration_ms <= 0
        else (len(normalized) - 1) * 1000 / duration_ms
    )
    return MovementSample(
        schema_version=SCHEMA_VERSION,
        split_unit=SPLIT_UNIT,
        sample_id=sample_id or str(uuid4()),
        session_id=session_id,
        label_id=label.label_id,
        label_name=label.name,
        created_at=created_at,
        duration_ms=duration_ms,
        frame_count=len(normalized),
        approximate_fps=approximate_fps,
        stance=stance,
        stance_description=stance_description,
        camera=camera,
        contains_images=False,
        contains_audio=False,
        contains_video=False,
        frames=normalized,
    )


def _identifier(value: str, kind: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise DatasetError(f"El identificador de {kind} no es un UUID válido.")
    return value.lower()


def _plane_to_dict(value: PlaneCoordinate | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"x": value.x, "y": value.y}


def _plane_from_dict(value: object) -> PlaneCoordinate | None:
    if value is None:
        return None
    if not isinstance(value, dict) or "x" not in value or "y" not in value:
        raise DatasetError("Una coordenada relativa está incompleta.")
    return PlaneCoordinate(float(value["x"]), float(value["y"]))


def _landmark_to_dict(value: LandmarkRecord | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "x": value.x,
        "y": value.y,
        "z": value.z,
        "visibility": value.visibility,
        "presence": value.presence,
        "reliable": value.reliable,
    }


def _landmark_from_dict(value: object) -> LandmarkRecord | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise DatasetError("Una articulación tiene un formato inválido.")
    return LandmarkRecord(
        x=None if value.get("x") is None else float(value["x"]),
        y=None if value.get("y") is None else float(value["y"]),
        z=None if value.get("z") is None else float(value["z"]),
        visibility=None if value.get("visibility") is None else float(value["visibility"]),
        presence=None if value.get("presence") is None else float(value["presence"]),
        reliable=bool(value.get("reliable", False)),
    )


def _features_to_dict(value: FeatureRecord) -> dict[str, object]:
    return {
        "left_elbow_angle": value.left_elbow_angle,
        "right_elbow_angle": value.right_elbow_angle,
        "left_shoulder_wrist_distance": value.left_shoulder_wrist_distance,
        "right_shoulder_wrist_distance": value.right_shoulder_wrist_distance,
        "left_wrist_relative": _plane_to_dict(value.left_wrist_relative),
        "right_wrist_relative": _plane_to_dict(value.right_wrist_relative),
        "wrist_separation": value.wrist_separation,
    }


def _features_from_dict(value: object) -> FeatureRecord:
    if not isinstance(value, dict):
        raise DatasetError("Las características del fotograma son inválidas.")
    return FeatureRecord(
        left_elbow_angle=_optional_float(value.get("left_elbow_angle")),
        right_elbow_angle=_optional_float(value.get("right_elbow_angle")),
        left_shoulder_wrist_distance=_optional_float(
            value.get("left_shoulder_wrist_distance")
        ),
        right_shoulder_wrist_distance=_optional_float(
            value.get("right_shoulder_wrist_distance")
        ),
        left_wrist_relative=_plane_from_dict(value.get("left_wrist_relative")),
        right_wrist_relative=_plane_from_dict(value.get("right_wrist_relative")),
        wrist_separation=_optional_float(value.get("wrist_separation")),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]


def _camera_to_dict(value: CameraMetadata) -> dict[str, object]:
    return {
        "index": value.index,
        "requested_width": value.requested_width,
        "requested_height": value.requested_height,
        "actual_width": value.actual_width,
        "actual_height": value.actual_height,
        "mirror_view": value.mirror_view,
        "coordinate_space": value.coordinate_space,
    }


def _camera_from_dict(value: object) -> CameraMetadata:
    if not isinstance(value, dict):
        raise DatasetError("La configuración de cámara es inválida.")
    return CameraMetadata(
        index=int(value["index"]),
        requested_width=int(value["requested_width"]),
        requested_height=int(value["requested_height"]),
        actual_width=(
            None if value.get("actual_width") is None else int(value["actual_width"])
        ),
        actual_height=(
            None if value.get("actual_height") is None else int(value["actual_height"])
        ),
        mirror_view=bool(value["mirror_view"]),
        coordinate_space=str(value.get("coordinate_space", COORDINATE_SPACE)),
    )


def _frame_to_dict(value: FrameRecord) -> dict[str, object]:
    return {
        "timestamp_ms": value.timestamp_ms,
        "elapsed_ms": value.elapsed_ms,
        "pose_detected": value.pose_detected,
        "landmarks": {
            name: _landmark_to_dict(value.landmarks.get(name))
            for name in TRACKED_LANDMARK_NAMES
        },
        "features": _features_to_dict(value.features),
    }


def _frame_from_dict(value: object) -> FrameRecord:
    if not isinstance(value, dict):
        raise DatasetError("Un fotograma tiene un formato inválido.")
    raw_landmarks = value.get("landmarks")
    if not isinstance(raw_landmarks, dict):
        raise DatasetError("Faltan las articulaciones de un fotograma.")
    return FrameRecord(
        timestamp_ms=int(value["timestamp_ms"]),
        elapsed_ms=int(value["elapsed_ms"]),
        pose_detected=bool(value["pose_detected"]),
        landmarks={
            name: _landmark_from_dict(raw_landmarks.get(name))
            for name in TRACKED_LANDMARK_NAMES
        },
        features=_features_from_dict(value.get("features")),
    )


def sample_to_dict(sample: MovementSample) -> dict[str, object]:
    return {
        "schema_version": sample.schema_version,
        "split_unit": sample.split_unit,
        "sample_id": sample.sample_id,
        "session_id": sample.session_id,
        "label_id": sample.label_id,
        "label_name": sample.label_name,
        "created_at": sample.created_at,
        "duration_ms": sample.duration_ms,
        "frame_count": sample.frame_count,
        "approximate_fps": sample.approximate_fps,
        "stance": sample.stance,
        "stance_description": sample.stance_description,
        "camera": _camera_to_dict(sample.camera),
        "privacy": {
            "contains_images": sample.contains_images,
            "contains_audio": sample.contains_audio,
            "contains_video": sample.contains_video,
        },
        "frames": [_frame_to_dict(frame) for frame in sample.frames],
    }


def sample_from_dict(value: object) -> MovementSample:
    if not isinstance(value, dict):
        raise DatasetError("La muestra no contiene un objeto JSON.")
    privacy = value.get("privacy")
    if not isinstance(privacy, dict):
        raise DatasetError("Falta la información de privacidad de la muestra.")
    frames = value.get("frames")
    if not isinstance(frames, list):
        raise DatasetError("La secuencia de fotogramas es inválida.")
    return MovementSample(
        schema_version=int(value["schema_version"]),
        split_unit=str(value["split_unit"]),
        sample_id=str(value["sample_id"]),
        session_id=str(value["session_id"]),
        label_id=int(value["label_id"]),
        label_name=str(value["label_name"]),
        created_at=str(value["created_at"]),
        duration_ms=int(value["duration_ms"]),
        frame_count=int(value["frame_count"]),
        approximate_fps=_optional_float(value.get("approximate_fps")),
        stance=str(value["stance"]),
        stance_description=str(value["stance_description"]),
        camera=_camera_from_dict(value.get("camera")),
        contains_images=bool(privacy.get("contains_images", True)),
        contains_audio=bool(privacy.get("contains_audio", True)),
        contains_video=bool(privacy.get("contains_video", True)),
        frames=tuple(_frame_from_dict(frame) for frame in frames),
    )


def session_to_dict(session: SessionRecord) -> dict[str, object]:
    return {
        "schema_version": session.schema_version,
        "session_id": session.session_id,
        "started_at": session.started_at,
        "stance": session.stance,
        "stance_description": session.stance_description,
        "camera": _camera_to_dict(session.camera),
        "saved_sample_count": session.saved_sample_count,
        "sample_ids": list(session.sample_ids),
    }


def session_from_dict(value: object) -> SessionRecord:
    if not isinstance(value, dict):
        raise DatasetError("La sesión no contiene un objeto JSON.")
    sample_ids = value.get("sample_ids", [])
    if not isinstance(sample_ids, list):
        raise DatasetError("La lista de muestras de la sesión es inválida.")
    return SessionRecord(
        schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        session_id=str(value["session_id"]),
        started_at=str(value["started_at"]),
        stance=str(value["stance"]),
        stance_description=str(value["stance_description"]),
        camera=_camera_from_dict(value.get("camera")),
        saved_sample_count=int(value.get("saved_sample_count", 0)),
        sample_ids=tuple(str(sample_id) for sample_id in sample_ids),
    )


class DatasetManager:
    """Guarda muestras atómicas, sesiones y el índice CSV."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index_path = root / "index.csv"

    def session_path(self, session_id: str) -> Path:
        return self.root / "sessions" / f"{_identifier(session_id, 'sesión')}.json"

    def sample_path(self, session_id: str, sample_id: str) -> Path:
        safe_session = _identifier(session_id, "sesión")
        safe_sample = _identifier(sample_id, "muestra")
        return self.root / "samples" / safe_session / f"{safe_sample}.json"

    def create_session(self, session: SessionRecord) -> Path:
        path = self.session_path(session.session_id)
        if path.exists():
            raise DuplicateDatasetEntryError(
                f"Ya existe la sesión {session.session_id}."
            )
        self._write_exclusive_json(path, session_to_dict(session))
        return path

    def load_session(self, session_id: str) -> SessionRecord:
        path = self.session_path(session_id)
        if not path.is_file():
            raise DatasetError(f"No existe la sesión {session_id}.")
        return session_from_dict(self._read_json(path))

    def update_session(self, session: SessionRecord) -> None:
        path = self.session_path(session.session_id)
        if not path.is_file():
            raise DatasetError(f"No existe la sesión {session.session_id}.")
        self._replace_json(path, session_to_dict(session))

    def save_sample(self, sample: MovementSample) -> Path:
        _identifier(sample.sample_id, "muestra")
        session = self.load_session(sample.session_id)
        if sample.sample_id in self._indexed_ids() or sample.sample_id in session.sample_ids:
            raise DuplicateDatasetEntryError(
                "Ya existe una muestra con el identificador "
                f"{sample.sample_id}. No se sobrescribió ningún archivo."
            )
        path = self.sample_path(sample.session_id, sample.sample_id)
        if path.exists():
            raise DuplicateDatasetEntryError(
                f"Ya existe el archivo de muestra {path.name}."
            )
        self._write_exclusive_json(path, sample_to_dict(sample))
        try:
            self._append_index(sample, path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self._refresh_session(sample.session_id)
        return path

    def load_sample(self, path: Path) -> MovementSample:
        return sample_from_dict(self._read_json(path))

    def relabel_sample(self, sample_id: str, label_id: int, config: AppConfig) -> MovementSample:
        """Cambia la etiqueta de una muestra ya guardada. No crea otra ni borra el archivo."""

        from app.sample_validator import validate_sample

        movement = movement_by_id(label_id)
        rows = self.load_index()
        matches = [index for index, row in enumerate(rows) if row.get("sample_id") == sample_id]
        if len(matches) != 1:
            raise DatasetError(f"La muestra {sample_id} no está una sola vez en el índice.")
        row = rows[matches[0]]
        path = self.root / row["relative_path"]
        sample = replace(self.load_sample(path), label_id=movement.label_id, label_name=movement.name)
        validation = validate_sample(sample, config)
        if not validation.accepted:
            raise DatasetError("La muestra no se reetiquetó: " + "; ".join(validation.reasons))
        self._replace_json(path, sample_to_dict(sample))
        row["label_id"] = str(movement.label_id)
        row["label_name"] = movement.name
        self._rewrite_index(rows)
        return sample

    def load_index(self) -> list[dict[str, str]]:
        if not self.index_path.is_file():
            return []
        with self.index_path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _rewrite_index(self, rows: list[dict[str, str]]) -> None:
        temporary = self.index_path.with_name(f".{self.index_path.name}.{uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in INDEX_COLUMNS})
        os.replace(temporary, self.index_path)

    def _indexed_ids(self) -> set[str]:
        return {row["sample_id"] for row in self.load_index()}

    def _append_index(self, sample: MovementSample, path: Path) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.index_path.exists()
        row = {
            "sample_id": sample.sample_id,
            "session_id": sample.session_id,
            "label_id": str(sample.label_id),
            "label_name": sample.label_name,
            "created_at": sample.created_at,
            "duration_ms": str(sample.duration_ms),
            "frame_count": str(sample.frame_count),
            "approximate_fps": (
                ""
                if sample.approximate_fps is None
                else f"{sample.approximate_fps:.6f}"
            ),
            "stance": sample.stance,
            "relative_path": path.relative_to(self.root).as_posix(),
            "schema_version": str(sample.schema_version),
        }
        with self.index_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _refresh_session(self, session_id: str) -> None:
        session = self.load_session(session_id)
        sample_ids = tuple(
            row["sample_id"]
            for row in self.load_index()
            if row["session_id"] == session.session_id
        )
        self.update_session(
            replace(
                session,
                sample_ids=sample_ids,
                saved_sample_count=len(sample_ids),
            )
        )

    @staticmethod
    def _read_json(path: Path) -> object:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_exclusive_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise DuplicateDatasetEntryError(
                f"Ya existe {path.name} y no se sobrescribió."
            ) from error
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _replace_json(path: Path, payload: object) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
