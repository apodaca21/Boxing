"""Máquina de estados para recopilar ejecuciones etiquetadas manualmente."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
import math
from uuid import uuid4

from app.config import STANCE_DESCRIPTION, AppConfig
from app.dataset_manager import (
    TRACKED_LANDMARK_NAMES,
    CameraMetadata,
    DatasetError,
    DatasetManager,
    DuplicateDatasetEntryError,
    FeatureRecord,
    FrameRecord,
    LandmarkRecord,
    MovementLabel,
    MOVEMENTS,
    PlaneCoordinate,
    SessionRecord,
    assemble_sample,
)
from app.geometry import BiomechanicalFeatures, RelativePoint
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from app.sample_validator import validate_sample


class CollectorPhase(Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    RESULT = "result"


LANDMARK_IDS: dict[str, LandmarkId] = {
    "left_shoulder": LandmarkId.LEFT_SHOULDER,
    "right_shoulder": LandmarkId.RIGHT_SHOULDER,
    "left_elbow": LandmarkId.LEFT_ELBOW,
    "right_elbow": LandmarkId.RIGHT_ELBOW,
    "left_wrist": LandmarkId.LEFT_WRIST,
    "right_wrist": LandmarkId.RIGHT_WRIST,
    "left_hip": LandmarkId.LEFT_HIP,
    "right_hip": LandmarkId.RIGHT_HIP,
}


@dataclass(frozen=True, slots=True)
class CollectionView:
    """Estado visual del modo de recopilación."""

    movement_name: str
    state_label: str
    countdown_seconds: int | None
    saved_count: int
    message: str
    outcome: str | None
    instructions: str


class DataCollector:
    """Coordina la cuenta regresiva, la captura y el guardado validado."""

    def __init__(
        self,
        config: AppConfig,
        dataset: DatasetManager,
        *,
        clock: Callable[[], datetime] | None = None,
        session_id_factory: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        if set(LANDMARK_IDS) != set(TRACKED_LANDMARK_NAMES):
            raise RuntimeError("Las articulaciones registradas no están sincronizadas.")
        self._config = config
        self._dataset = dataset
        self._clock = clock or (lambda: datetime.now().astimezone())
        self.session_id = session_id_factory()
        self._phase = CollectorPhase.IDLE
        self._label: MovementLabel = MOVEMENTS[0]
        self._phase_started_ms: int | None = None
        self._frames: list[FrameRecord] = []
        self._capture_size: tuple[int, int] | None = None
        self._size_changed = False
        self._message = "Selecciona un movimiento y presiona Espacio."
        self._outcome: str | None = None
        self._saved_count = 0
        self._last_timestamp_ms: int | None = None
        self._session = SessionRecord(
            session_id=self.session_id,
            started_at=self._clock().isoformat(timespec="seconds"),
            stance=config.stance,
            stance_description=STANCE_DESCRIPTION,
            camera=CameraMetadata(
                index=config.camera_index,
                requested_width=config.frame_width,
                requested_height=config.frame_height,
                actual_width=None,
                actual_height=None,
                mirror_view=config.mirror_view,
            ),
            saved_sample_count=0,
            sample_ids=(),
        )
        dataset.create_session(self._session)

    @property
    def saved_count(self) -> int:
        return self._saved_count

    def handle_key(self, key: int, timestamp_ms: int) -> bool:
        """Procesa el teclado. Devuelve True cuando corresponde cerrar."""

        self._last_timestamp_ms = timestamp_ms
        if key in (ord("q"), ord("Q"), 27):
            return True
        if self._phase in {CollectorPhase.COUNTDOWN, CollectorPhase.RECORDING}:
            return False
        if ord("0") <= key <= ord("4"):
            self._label = MOVEMENTS[key - ord("0")]
            self._phase = CollectorPhase.IDLE
            self._outcome = None
            self._message = "Movimiento seleccionado. Presiona Espacio para grabar."
            return False
        if key == ord(" "):
            self._phase = CollectorPhase.COUNTDOWN
            self._phase_started_ms = timestamp_ms
            self._frames = []
            self._capture_size = None
            self._size_changed = False
            self._outcome = None
            self._message = f"Prepárate para ejecutar: {self._label.display_name}."
            return False
        return False

    def advance(
        self,
        timestamp_ms: int,
        pose: PoseObservation,
        features: BiomechanicalFeatures,
        width: int,
        height: int,
    ) -> str | None:
        """Avanza temporizadores y registra el fotograma si está grabando."""

        self._last_timestamp_ms = timestamp_ms
        self._remember_camera_size(width, height)
        if (
            self._phase is CollectorPhase.COUNTDOWN
            and self._phase_started_ms is not None
            and timestamp_ms - self._phase_started_ms >= self._config.countdown_ms
        ):
            self._phase = CollectorPhase.RECORDING
            self._frames = []
            self._capture_size = (width, height)
            self._size_changed = False
            self._message = "Grabando movimiento."
            self._outcome = None

        if self._phase is not CollectorPhase.RECORDING:
            return None
        if self._capture_size != (width, height):
            self._size_changed = True
        self._frames.append(self._frame_record(timestamp_ms, pose, features))
        origin = self._frames[0].timestamp_ms
        if timestamp_ms - origin >= self._config.capture_duration_ms:
            return self._finalize()
        return None

    def view(self) -> CollectionView:
        countdown_seconds = None
        if (
            self._phase is CollectorPhase.COUNTDOWN
            and self._phase_started_ms is not None
            and self._last_timestamp_ms is not None
        ):
            remaining_ms = self._config.countdown_ms - (
                self._last_timestamp_ms - self._phase_started_ms
            )
            countdown_seconds = max(1, math.ceil(remaining_ms / 1000))
        labels = {
            CollectorPhase.IDLE: "LISTO",
            CollectorPhase.COUNTDOWN: "PREPARATE",
            CollectorPhase.RECORDING: "GRABANDO",
            CollectorPhase.RESULT: (
                "CAPTURA VALIDA"
                if self._outcome == "accepted"
                else "CAPTURA RECHAZADA"
            ),
        }
        return CollectionView(
            movement_name=self._label.display_name,
            state_label=labels[self._phase],
            countdown_seconds=countdown_seconds,
            saved_count=self._saved_count,
            message=self._message,
            outcome=self._outcome,
            instructions="0-4: movimiento    Espacio: grabar    Q: salir",
        )

    def _remember_camera_size(self, width: int, height: int) -> None:
        camera = self._session.camera
        if camera.actual_width is not None:
            return
        self._session = replace(
            self._session,
            camera=replace(camera, actual_width=width, actual_height=height),
        )
        self._dataset.update_session(self._session)

    def _frame_record(
        self,
        timestamp_ms: int,
        pose: PoseObservation,
        features: BiomechanicalFeatures,
    ) -> FrameRecord:
        return build_frame_record(self._config, timestamp_ms, pose, features)

    def _finalize(self) -> str:
        self._phase = CollectorPhase.RESULT
        if self._size_changed or self._capture_size is None:
            return self._reject(("La resolución cambió durante la captura.",))
        width, height = self._capture_size
        sample = assemble_sample(
            session_id=self.session_id,
            label=self._label,
            frames=tuple(self._frames),
            stance=self._config.stance,
            stance_description=STANCE_DESCRIPTION,
            camera=replace(
                self._session.camera,
                actual_width=width,
                actual_height=height,
            ),
            created_at=self._clock().isoformat(timespec="seconds"),
        )
        result = validate_sample(sample, self._config)
        if not result.accepted:
            return self._reject(result.reasons)
        try:
            self._dataset.save_sample(sample)
        except DuplicateDatasetEntryError as error:
            return self._reject((str(error),))
        except DatasetError as error:
            return self._reject((str(error),))
        self._saved_count = self._dataset.load_session(self.session_id).saved_sample_count
        self._frames = []
        self._outcome = "accepted"
        self._message = "Captura válida guardada."
        return self._message

    def _reject(self, reasons: tuple[str, ...]) -> str:
        self._phase = CollectorPhase.RESULT
        self._outcome = "rejected"
        self._frames = []
        self._message = "; ".join(reasons)
        return self._message


def build_frame_record(
    config: AppConfig,
    timestamp_ms: int,
    pose: PoseObservation,
    features: BiomechanicalFeatures,
) -> FrameRecord:
    """Fotograma biomecánico del mismo esquema que usa la recopilación."""

    if not pose.detected:
        landmarks: dict[str, LandmarkRecord | None] = {
            name: None for name in TRACKED_LANDMARK_NAMES
        }
        feature_record = FeatureRecord.missing()
    else:
        landmarks = {
            name: _landmark_record(pose.landmark(LANDMARK_IDS[name]), config)
            for name in TRACKED_LANDMARK_NAMES
        }
        feature_record = _feature_record(features)
    return FrameRecord(
        timestamp_ms=timestamp_ms,
        elapsed_ms=0,
        pose_detected=pose.detected,
        landmarks=landmarks,
        features=feature_record,
    )


def _landmark_record(
    observation: LandmarkObservation | None,
    config: AppConfig,
) -> LandmarkRecord | None:
    if observation is None or not observation.is_finite:
        return None
    return LandmarkRecord(
        x=observation.x,
        y=observation.y,
        z=observation.z,
        visibility=observation.visibility,
        presence=observation.presence,
        reliable=observation.is_reliable(
            config.min_landmark_visibility,
            config.min_landmark_presence,
        ),
    )


def _plane(value: RelativePoint | None) -> PlaneCoordinate | None:
    if value is None:
        return None
    return PlaneCoordinate(value.x, value.y)


def _feature_record(features: BiomechanicalFeatures) -> FeatureRecord:
    return FeatureRecord(
        left_elbow_angle=features.left_elbow_angle,
        right_elbow_angle=features.right_elbow_angle,
        left_shoulder_wrist_distance=features.left_shoulder_wrist_distance,
        right_shoulder_wrist_distance=features.right_shoulder_wrist_distance,
        left_wrist_relative=_plane(features.left_wrist_relative),
        right_wrist_relative=_plane(features.right_wrist_relative),
        wrist_separation=features.wrist_separation,
    )
