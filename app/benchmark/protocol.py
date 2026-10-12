"""Rutina guiada. Una repetición solo entra en la evaluación si la persona la verifica."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.dataset_manager import MOVEMENTS
from app.trainer.results import PunchEvent


CLASS_ORDER = (1, 2, 3, 4, 0)
PHASES = ("prepare", "execute", "return_guard", "review", "next", "finished")


@dataclass(frozen=True, slots=True)
class PredictionMark:
    """Golpe confirmado por el clasificador, todavía sin juzgar si es correcto."""

    timestamp_ms: int
    label_id: int | None
    display_name: str
    latency_ms: float | None


@dataclass
class RepetitionRecord:
    """Una repetición. La etiqueta verificada puede diferir de la solicitada."""

    requested_label_id: int
    repetition_index: int
    window_start_ms: int | None = None
    window_end_ms: int | None = None
    predictions: list[PredictionMark] = field(default_factory=list)
    fps_samples: list[float] = field(default_factory=list)
    status: str = "pending"
    verified_label_id: int | None = None
    corrected: bool = False
    exclusion_reason: str | None = None

    @property
    def requested_name(self) -> str:
        return MOVEMENTS[self.requested_label_id].display_name

    @property
    def verified_name(self) -> str | None:
        if self.verified_label_id is None:
            return None
        return MOVEMENTS[self.verified_label_id].display_name


@dataclass(frozen=True, slots=True)
class BenchmarkView:
    """Texto de la rutina. No afirma que el clasificador haya acertado."""

    instruction: str
    detail: str
    progress: str
    sensitivity: str
    classifier_name: str
    detected_count: int
    review: bool


class BenchmarkProtocol:
    """Avanza la rutina y espera una decisión humana antes de puntuar."""

    def __init__(
        self,
        *,
        repetitions: int = 10,
        prepare_ms: int = 1500,
        active_ms: int = 2000,
        return_ms: int = 1000,
        next_ms: int = 800,
        classes: tuple[int, ...] = CLASS_ORDER,
    ) -> None:
        if repetitions < 1:
            raise ValueError("La evaluación necesita al menos una repetición por clase.")
        if prepare_ms < 0 or active_ms <= 0 or return_ms < 0 or next_ms < 0:
            raise ValueError("Los intervalos de la rutina no son válidos.")
        if any(label not in MOVEMENTS for label in classes):
            raise ValueError("Hay una clase de evaluación desconocida.")
        self.repetitions = repetitions
        self.prepare_ms = prepare_ms
        self.active_ms = active_ms
        self.return_ms = return_ms
        self.next_ms = next_ms
        self.classes = classes
        self.phase = "prepare"
        self._class_cursor = 0
        self._rep_cursor = 0
        self._phase_started_ms: int | None = None
        self._pending_correction: int | None = None
        self.records: list[RepetitionRecord] = []
        self._current: RepetitionRecord | None = None

    @property
    def requested_label_id(self) -> int:
        return self.classes[self._class_cursor]

    @property
    def finished(self) -> bool:
        return self.phase == "finished"

    def start(self, timestamp_ms: int) -> None:
        self._phase_started_ms = timestamp_ms
        self._begin_repetition()

    def advance(self, timestamp_ms: int) -> bool:
        """Avanza los intervalos automáticos. La revisión no se salta sola."""

        if self.phase == "finished":
            return False
        if self._phase_started_ms is None:
            self.start(timestamp_ms)
            return True
        elapsed = timestamp_ms - self._phase_started_ms
        if self.phase == "prepare" and elapsed >= self.prepare_ms:
            self._enter_execute(timestamp_ms)
            return True
        if self.phase == "execute" and elapsed >= self.active_ms:
            self._enter_return(timestamp_ms)
            return True
        if self.phase == "return_guard" and elapsed >= self.return_ms:
            self._enter_review(timestamp_ms)
            return True
        if self.phase == "next" and elapsed >= self.next_ms:
            self._advance_schedule(timestamp_ms)
            return True
        return False

    def observe(
        self,
        timestamp_ms: int,
        events: tuple[PunchEvent, ...],
        *,
        latency_ms: float | None = None,
        fps: float | None = None,
    ) -> None:
        """Guarda predicciones solo dentro del intervalo de ejecución."""

        if self.phase != "execute" or self._current is None:
            return
        if fps is not None:
            self._current.fps_samples.append(fps)
        for event in events:
            label_id = event.label_id if event.label_id in (1, 2, 3, 4) else None
            self._current.predictions.append(
                PredictionMark(
                    timestamp_ms=timestamp_ms,
                    label_id=label_id,
                    display_name=event.display_name,
                    latency_ms=latency_ms,
                )
            )

    def correct(self, label_id: int) -> bool:
        if self.phase != "review" or label_id not in MOVEMENTS:
            return False
        self._pending_correction = label_id
        return True

    def confirm(self) -> RepetitionRecord | None:
        """La persona afirma qué ejecutó. Una predicción previa no basta."""

        if self.phase != "review" or self._current is None:
            return None
        requested = self._current.requested_label_id
        verified = requested if self._pending_correction is None else self._pending_correction
        self._current.status = "verified"
        self._current.verified_label_id = verified
        self._current.corrected = verified != requested
        self._pending_correction = None
        record = self._current
        self._enter_next_or_finish(record.window_end_ms or 0)
        return record

    def exclude(self, reason: str = "excluida por el usuario") -> RepetitionRecord | None:
        if self.phase != "review" or self._current is None:
            return None
        self._current.status = "excluded"
        self._current.exclusion_reason = reason
        self._current.verified_label_id = None
        self._pending_correction = None
        record = self._current
        self._enter_next_or_finish(record.window_end_ms or 0)
        return record

    def verified_records(self) -> tuple[RepetitionRecord, ...]:
        return tuple(record for record in self.records if record.status == "verified")

    def view(self, *, sensitivity: str, classifier_name: str) -> BenchmarkView:
        current = self._current
        detected = 0 if current is None else len(current.predictions)
        if self.phase == "finished":
            instruction = "Evaluación terminada."
            detail = "El resumen usa solo las repeticiones verificadas."
        elif self.phase == "prepare":
            instruction = "Prepárate."
            detail = f"Después: {self._execute_instruction()}"
        elif self.phase == "execute":
            instruction = self._execute_instruction()
            detail = "El intervalo está abierto. Aún no hay una etiqueta verificada."
        elif self.phase == "return_guard":
            instruction = "Regresa a guardia."
            detail = "Las detecciones de este tramo no se evalúan."
        elif self.phase == "next":
            instruction = "Siguiente repetición."
            detail = "La anterior queda fuera hasta que se confirme."
        else:
            requested = MOVEMENTS[self.requested_label_id].display_name
            chosen = requested
            if self._pending_correction is not None:
                chosen = MOVEMENTS[self._pending_correction].display_name
            instruction = "Confirma tu ejecución."
            detail = (
                f"Solicitado: {requested}. Etiqueta propuesta: {chosen}. "
                f"Detectados: {detected}. Y confirma, 0-4 corrige, X excluye."
            )
        progress = self._progress_text()
        return BenchmarkView(
            instruction=instruction,
            detail=detail,
            progress=progress,
            sensitivity=sensitivity.upper(),
            classifier_name=classifier_name,
            detected_count=detected,
            review=self.phase == "review",
        )

    def _execute_instruction(self) -> str:
        movement = MOVEMENTS[self.requested_label_id]
        if movement.label_id == 0:
            return "Mantén la guardia."
        return f"Ejecuta un {movement.display_name[0].lower()}{movement.display_name[1:]}."

    def _progress_text(self) -> str:
        if self.phase == "finished":
            verified = len(self.verified_records())
            return f"Verificadas {verified}"
        name = MOVEMENTS[self.requested_label_id].display_name
        return f"{name}  {self._rep_cursor + 1}/{self.repetitions}"

    def _begin_repetition(self) -> None:
        self._current = RepetitionRecord(
            requested_label_id=self.requested_label_id,
            repetition_index=self._rep_cursor + 1,
        )
        self.records.append(self._current)
        self._pending_correction = None

    def _enter_execute(self, timestamp_ms: int) -> None:
        self.phase = "execute"
        self._phase_started_ms = timestamp_ms
        if self._current is not None:
            self._current.window_start_ms = timestamp_ms

    def _enter_return(self, timestamp_ms: int) -> None:
        self.phase = "return_guard"
        self._phase_started_ms = timestamp_ms
        if self._current is not None:
            self._current.window_end_ms = timestamp_ms

    def _enter_review(self, timestamp_ms: int) -> None:
        self.phase = "review"
        self._phase_started_ms = timestamp_ms

    def _enter_next_or_finish(self, timestamp_ms: int) -> None:
        if self._class_cursor >= len(self.classes) - 1 and self._rep_cursor >= self.repetitions - 1:
            self.phase = "finished"
            self._current = None
            return
        self.phase = "next"
        self._phase_started_ms = timestamp_ms

    def _advance_schedule(self, timestamp_ms: int) -> None:
        self._rep_cursor += 1
        if self._rep_cursor >= self.repetitions:
            self._rep_cursor = 0
            self._class_cursor += 1
        self.phase = "prepare"
        self._phase_started_ms = timestamp_ms
        self._begin_repetition()
