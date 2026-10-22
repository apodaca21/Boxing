"""Clasificador en vivo que reutiliza el genoma y el vector guardados por NEAT."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from app.config import ORTHODOX_STANCE, STANCE_DESCRIPTION
from app.dataset_manager import (
    MOVEMENTS,
    CameraMetadata,
    FeatureRecord,
    FrameRecord,
    MovementSample,
    PlaneCoordinate,
    assemble_sample,
)
from app.evolution.features import extract_raw_features
from app.evolution.metrics import predict_label
from app.evolution.model_store import EvolvedModel
from app.geometry import BiomechanicalFeatures, RelativePoint
from app.trainer.results import (
    STATUS_EXTENDING,
    ClassificationResult,
    PunchEvent,
    neutral_result,
    no_detection_result,
)


class ClassifierUnavailable(RuntimeError):
    """El clasificador pedido no puede usarse con los archivos actuales."""


class NeatClassifier:
    """Traduce una ventana reciente con el mismo preprocesamiento del entrenamiento.

    La clase es el argmax de las cinco salidas. Esas activaciones no son
    probabilidades. Un margen pequeño queda como movimiento no confirmado.
    El mismo brazo no repite el conteo hasta que la activación baja o hay cooldown.
    """

    name = "neat"
    provisional = False

    def __init__(
        self,
        transform: Callable[[tuple[float | None, ...]], tuple[float, ...]],
        activate: Callable[[tuple[float, ...]], list[float] | tuple[float, ...]],
        time_steps: int,
        *,
        window_ms: int = 1600,
        neutral_frames_to_release: int = 3,
        cooldown_ms: int = 280,
        min_margin: float = 0.08,
        retrigger_dip: float = 0.35,
    ) -> None:
        if time_steps < 2:
            raise ValueError("La ventana NEAT necesita al menos 2 puntos temporales.")
        self._transform = transform
        self._activate = activate
        self._time_steps = time_steps
        self._window_ms = window_ms
        self._neutral_frames_to_release = neutral_frames_to_release
        self._cooldown_ms = cooldown_ms
        self._min_margin = min_margin
        self._retrigger_dip = retrigger_dip
        self._frames: list[FrameRecord] = []
        self._latched_label: int | None = None
        self._neutral_streak = 0
        self._released = True
        self._peak = 0.0
        self._arm_ready_ms = {"left": -1, "right": -1}
        self._event_id: str | None = None
        self._cycle_started_ms: int | None = None

    @classmethod
    def from_model(cls, model: EvolvedModel) -> NeatClassifier:
        import neat

        config = model.neat_config()
        network = neat.nn.FeedForwardNetwork.create(model.genome, config)
        return cls(
            transform=model.scaler.transform,
            activate=network.activate,
            time_steps=model.scaler.time_steps,
        )

    def reset(self) -> None:
        self._frames.clear()
        self._latched_label = None
        self._neutral_streak = 0
        self._released = True
        self._peak = 0.0
        self._arm_ready_ms = {"left": -1, "right": -1}
        self._event_id: str | None = None
        self._cycle_started_ms: int | None = None

    def update(
        self,
        features: BiomechanicalFeatures,
        *,
        pose_detected: bool,
        pose: object | None = None,
        frame: object | None = None,
    ) -> ClassificationResult:
        if not pose_detected:
            self.reset()
            return no_detection_result(self.name)

        self._frames.append(_frame_record(features, pose_detected))
        newest = features.timestamp_ms
        self._frames = [
            frame
            for frame in self._frames
            if newest - frame.timestamp_ms <= self._window_ms
        ]
        if len(self._frames) < 2 or self._frames[-1].timestamp_ms == self._frames[0].timestamp_ms:
            return no_detection_result(self.name)

        sample = _window_sample(self._frames)
        raw = extract_raw_features(sample, self._time_steps)
        vector = self._transform(raw)
        outputs = tuple(float(value) for value in self._activate(vector))
        label = predict_label(outputs)
        return self._event_from_label(label, newest, outputs)

    def _event_from_label(
        self,
        label: int,
        timestamp_ms: int,
        outputs: tuple[float, ...],
    ) -> ClassificationResult:
        score_labels = tuple(MOVEMENTS[index].name for index in range(len(outputs)))
        ordered = sorted(outputs, reverse=True)
        margin = ordered[0] - ordered[1]
        if label != 0 and margin < self._min_margin:
            return ClassificationResult(
                source=self.name,
                status="incierto",
                label_id=None,
                display_name="MOVIMIENTO NO CONFIRMADO",
                rejection="margen insuficiente",
                score_labels=score_labels,
                scores=outputs,
                diagnostics={"margin": margin, "min_margin": self._min_margin},
            )

        if label == 0:
            self._neutral_streak += 1
            self._peak = 0.0
            if self._neutral_streak >= self._neutral_frames_to_release:
                self._latched_label = None
                self._released = True
                self._event_id = None
                self._cycle_started_ms = None
            return replace(
                neutral_result(self.name),
                score_labels=score_labels,
                scores=outputs,
            )

        self._neutral_streak = 0
        movement = MOVEMENTS[label]
        arm = _arm_for_label(label)
        activation = outputs[label]
        if self._latched_label is not None and not self._released:
            dipped = activation < self._peak - self._retrigger_dip
            if self._latched_label == label and dipped:
                self._released = True
            self._peak = max(self._peak, activation)
            if not self._released or timestamp_ms < self._arm_ready_ms[arm]:
                return ClassificationResult(
                    source=self.name,
                    status=STATUS_EXTENDING,
                    label_id=None,
                    display_name=movement.display_name,
                    score_labels=score_labels,
                    scores=outputs,
                    diagnostics={
                        "margin": margin,
                        "cooldown_until_ms": self._arm_ready_ms[arm],
                        "arm": arm,
                        "event_id": self._event_id,
                    },
                )
        elif timestamp_ms < self._arm_ready_ms[arm]:
            return ClassificationResult(
                source=self.name,
                status=STATUS_EXTENDING,
                label_id=None,
                display_name=movement.display_name,
                score_labels=score_labels,
                scores=outputs,
                diagnostics={
                    "margin": margin,
                    "cooldown_until_ms": self._arm_ready_ms[arm],
                    "arm": arm,
                },
            )

        self._latched_label = label
        self._released = False
        self._peak = activation
        self._arm_ready_ms[arm] = timestamp_ms + self._cooldown_ms
        self._cycle_started_ms = timestamp_ms
        self._event_id = f"neat:{timestamp_ms}:{arm}"
        event = PunchEvent(
            movement.label_id,
            movement.display_name,
            arm,
            event_id=self._event_id,
            started_ms=timestamp_ms,
            ended_ms=timestamp_ms,
            confirmed=True,
            source=self.name,
        )
        return ClassificationResult(
            source=self.name,
            status="completado",
            label_id=movement.label_id,
            display_name=movement.display_name,
            events=(event,),
            active_arm=event.arm,
            score_labels=score_labels,
            scores=outputs,
            diagnostics={
                "margin": margin,
                "cooldown_until_ms": self._arm_ready_ms[arm],
                "arm": arm,
            },
        )


def resolve_neat_model(explicit: Path | None, results_dir: Path) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not (path / "winner.pkl").is_file():
            raise ClassifierUnavailable(
                "El modelo NEAT indicado no está disponible en "
                f"'{path}'. Entrena con python main.py --mode train o elige "
                "--classifier heuristic."
            )
        return path

    if results_dir.is_dir():
        candidates = [
            path.parent
            for path in results_dir.glob("**/winner.pkl")
            if path.is_file()
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)

    raise ClassifierUnavailable(
        "No hay un modelo NEAT entrenado. El directorio de resultados no "
        "contiene winner.pkl. El entrenador no sustituye este clasificador "
        "por el detector provisional."
    )


def _plane(point: RelativePoint | None) -> PlaneCoordinate | None:
    if point is None:
        return None
    return PlaneCoordinate(point.x, point.y)


def _frame_record(features: BiomechanicalFeatures, pose_detected: bool) -> FrameRecord:
    return FrameRecord(
        timestamp_ms=features.timestamp_ms,
        elapsed_ms=0,
        pose_detected=pose_detected,
        landmarks={},
        features=FeatureRecord(
            left_elbow_angle=features.left_elbow_angle,
            right_elbow_angle=features.right_elbow_angle,
            left_shoulder_wrist_distance=features.left_shoulder_wrist_distance,
            right_shoulder_wrist_distance=features.right_shoulder_wrist_distance,
            left_wrist_relative=_plane(features.left_wrist_relative),
            right_wrist_relative=_plane(features.right_wrist_relative),
            wrist_separation=features.wrist_separation,
        ),
    )


def _window_sample(frames: list[FrameRecord]) -> MovementSample:
    return assemble_sample(
        session_id="00000000-0000-4000-8000-000000000001",
        label=MOVEMENTS[0],
        frames=tuple(frames),
        stance=ORTHODOX_STANCE,
        stance_description=STANCE_DESCRIPTION,
        camera=CameraMetadata(
            index=0,
            requested_width=0,
            requested_height=0,
            actual_width=None,
            actual_height=None,
            mirror_view=True,
        ),
        created_at="1970-01-01T00:00:00+00:00",
        sample_id="00000000-0000-4000-8000-000000000002",
    )


def _arm_for_label(label: int) -> str:
    if label in (1, 3):
        return "left"
    if label in (2, 4):
        return "right"
    return "left"
