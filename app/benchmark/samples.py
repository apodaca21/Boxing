"""Guarda ventanas verificadas con el mismo esquema que ya consume NEAT."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from app.config import STANCE_DESCRIPTION, AppConfig
from app.benchmark.protocol import RepetitionRecord
from app.dataset_manager import (
    CameraMetadata,
    DatasetManager,
    FrameRecord,
    MOVEMENTS,
    SessionRecord,
    assemble_sample,
)
from app.sample_validator import validate_sample


@dataclass
class SampleSaveResult:
    saved: bool
    sample_id: str | None
    reasons: tuple[str, ...]


class VerifiedSampleStore:
    """Persiste solo repeticiones confirmadas. No escribe imagen ni video."""

    def __init__(self, config: AppConfig, dataset: DatasetManager) -> None:
        self._config = config
        self._dataset = dataset
        self.session_id = str(uuid4())
        self._opened = False
        self.saved_ids: list[str] = []
        self.rejected: list[tuple[str, tuple[str, ...]]] = []

    def save(
        self,
        record: RepetitionRecord,
        frames: tuple[FrameRecord, ...] | list[FrameRecord],
        *,
        width: int,
        height: int,
        created_at: str | None = None,
    ) -> SampleSaveResult:
        if record.status != "verified" or record.verified_label_id is None:
            raise ValueError(
                "Solo se puede guardar una repetición verificada por el usuario."
            )
        if not frames:
            reasons = ("La ventana no contiene fotogramas.",)
            self.rejected.append((record.requested_name, reasons))
            return SampleSaveResult(False, None, reasons)
        sample = assemble_sample(
            session_id=self.session_id,
            label=MOVEMENTS[record.verified_label_id],
            frames=tuple(frames),
            stance=self._config.stance,
            stance_description=STANCE_DESCRIPTION,
            camera=CameraMetadata(
                index=self._config.camera_index,
                requested_width=self._config.frame_width,
                requested_height=self._config.frame_height,
                actual_width=width,
                actual_height=height,
                mirror_view=self._config.mirror_view,
            ),
            created_at=created_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        validation = validate_sample(sample, self._config)
        if not validation.accepted:
            self.rejected.append((record.requested_name, validation.reasons))
            return SampleSaveResult(False, None, validation.reasons)
        self._ensure_session(sample.camera)
        self._dataset.save_sample(sample)
        self.saved_ids.append(sample.sample_id)
        return SampleSaveResult(True, sample.sample_id, ())

    def _ensure_session(self, camera: CameraMetadata) -> None:
        if self._opened:
            return
        self._dataset.create_session(
            SessionRecord(
                session_id=self.session_id,
                started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                stance=self._config.stance,
                stance_description=STANCE_DESCRIPTION,
                camera=camera,
                saved_sample_count=0,
                sample_ids=(),
            )
        )
        self._opened = True
