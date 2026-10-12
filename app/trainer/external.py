"""Adaptador YOLO Pose + LSTM de ACM40960/Boxing.

Copyright (c) 2025 Jayaganeshan Thanga Kumar, Kabilesh Sekar. Licencia MIT.
El modelo de pose YOLO11 se usa mediante Ultralytics, licencia AGPL-3.0.
No se instaló CUDA ni TensorRT.

La cámara de EvoBox se abre una sola vez. El mismo fotograma alimenta a
MediaPipe, que dibuja el esqueleto, y a YOLO, que prepara la secuencia del
LSTM. La inferencia corre en un hilo para no bloquear la ventana.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import queue
import threading
import time

import numpy as np

from app.config import PROJECT_ROOT
from app.trainer.acm_boxing import (
    BLOCK_ACTIONS,
    INPUT_SIZE,
    LSTM_HIDDEN_SIZE,
    LSTM_LAYERS,
    PUNCH_ACTIONS,
    SEQUENCE_LENGTH,
    ExternalSettings,
    SequenceRecognizer,
)
from app.trainer.neat_classifier import ClassifierUnavailable
from app.trainer.results import ClassificationResult, neutral_result


DEFAULT_EXTERNAL_DIR = PROJECT_ROOT / "models" / "external"
PUNCH_WEIGHTS = "punch_classifier.pth"
BLOCK_WEIGHTS = "block_classifier.pth"
POSE_WEIGHTS = "yolo11m-pose.pt"


def missing_weights_message(model_dir: Path) -> str:
    return (
        "El clasificador externo no está disponible. Faltan pesos en "
        f"'{model_dir}'. Se necesitan {PUNCH_WEIGHTS} y {POSE_WEIGHTS} "
        "publicados en https://github.com/ACM40960/Boxing."
    )


def select_torch_device(torch_module) -> str:
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_lstm(path: Path, num_classes: int, device: str):
    """Carga un state_dict con deserialización restringida."""

    import torch
    from torch import nn

    class ActionClassifierLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                INPUT_SIZE,
                LSTM_HIDDEN_SIZE,
                LSTM_LAYERS,
                batch_first=True,
                dropout=0.5,
            )
            self.fc = nn.Linear(LSTM_HIDDEN_SIZE, num_classes)
            self.hidden_size = LSTM_HIDDEN_SIZE
            self.num_layers = LSTM_LAYERS

        def forward(self, sequence):
            batch = sequence.size(0)
            hidden = torch.zeros(
                self.num_layers,
                batch,
                self.hidden_size,
                device=sequence.device,
                dtype=sequence.dtype,
            )
            cell = torch.zeros(
                self.num_layers,
                batch,
                self.hidden_size,
                device=sequence.device,
                dtype=sequence.dtype,
            )
            output, _ = self.lstm(sequence, (hidden, cell))
            return self.fc(output[:, -1, :])

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ClassifierUnavailable(
            f"No se pudo cargar '{path.name}' con weights_only=True: {error}"
        ) from error
    if not isinstance(state, dict):
        raise ClassifierUnavailable(
            f"'{path.name}' no es un state_dict. No se deserializará como un objeto arbitrario."
        )
    model = ActionClassifierLSTM()
    try:
        model.load_state_dict(state)
    except Exception as error:
        raise ClassifierUnavailable(
            f"La arquitectura no coincide con '{path.name}': {error}"
        ) from error
    model.to(device)
    model.eval()
    return model


class ExternalClassifier:
    """Clasificador externo. Sus resultados no se presentan como NEAT ni como heurístico."""

    name = "external"
    provisional = False

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str | None = None,
        warmup: bool = True,
    ) -> None:
        import torch

        self.model_dir = model_dir
        self.device = device or select_torch_device(torch)
        punch_path = model_dir / PUNCH_WEIGHTS
        pose_path = model_dir / POSE_WEIGHTS
        if not punch_path.is_file() or not pose_path.is_file():
            raise ClassifierUnavailable(missing_weights_message(model_dir))
        self._torch = torch
        self.punch_model = load_lstm(punch_path, len(PUNCH_ACTIONS), self.device)
        block_path = model_dir / BLOCK_WEIGHTS
        self.block_model = (
            load_lstm(block_path, len(BLOCK_ACTIONS), self.device)
            if block_path.is_file()
            else None
        )
        self.pose_model = self._load_pose(pose_path)
        self._recognizer = SequenceRecognizer(
            self._predict_punch,
            predict_block=self._predict_block if self.block_model is not None else None,
        )
        self._latest = neutral_result(self.name)
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.sensitivity = "normal"
        self.last_error: str | None = None
        self.yolo_latency_ms: float | None = None
        self.lstm_latency_ms: float | None = None
        self.warmup_latency_ms: float | None = None
        if warmup:
            self._warmup()
        self._thread = threading.Thread(target=self._worker, name="evobox-yolo-lstm", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def set_sensitivity(self, name: str) -> None:
        """Ajusta la separación entre eventos. No cambia la red ni su entrada."""

        table = {
            "normal": (20, 2.0),
            "high": (10, 1.2),
            "maximum": (5, 0.6),
        }
        key = name.strip().lower()
        if key not in table:
            raise ValueError("La sensibilidad debe ser normal, high o maximum.")
        recovery, speed = table[key]
        self.sensitivity = key
        self._recognizer.settings = replace(
            self._recognizer.settings,
            recovery_frames=recovery,
            min_punch_speed=speed,
        )

    def reset(self) -> None:
        with self._lock:
            self._recognizer.reset()
            self._latest = neutral_result(self.name)

    def update(
        self,
        features: object,
        *,
        pose_detected: bool,
        pose: object | None = None,
        frame: np.ndarray | None = None,
    ) -> ClassificationResult:
        del features, pose_detected, pose
        if frame is not None and not self._stop.is_set():
            snapshot = np.array(frame, copy=True)
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(snapshot)
                except queue.Full:
                    pass
        with self._lock:
            return self._latest

    def analyze_frame(self, frame: np.ndarray) -> ClassificationResult:
        """Inferencia síncrona. La usa el hilo de trabajo y las mediciones."""

        started = time.perf_counter()
        try:
            detection = self._detect_people(frame)
        except Exception as error:
            self.last_error = str(error)
            result = ClassificationResult(
                source=self.name,
                status="no_detection",
                label_id=None,
                display_name="ERROR DE MODELO",
            )
            self.yolo_latency_ms = (time.perf_counter() - started) * 1000.0
            return result
        self.yolo_latency_ms = (time.perf_counter() - started) * 1000.0
        lstm_started = time.perf_counter()
        if detection is None:
            result = self._recognizer.lost()
        else:
            keypoints, confidences = detection
            result = self._recognizer.push(keypoints, confidences)
        self.lstm_latency_ms = (time.perf_counter() - lstm_started) * 1000.0
        return replace(
            result,
            latency_ms=(self.yolo_latency_ms or 0.0) + (self.lstm_latency_ms or 0.0),
            sensitivity=self.sensitivity,
        )

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None or self._stop.is_set():
                break
            result = self.analyze_frame(frame)
            with self._lock:
                self._latest = result

    def _warmup(self) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        started = time.perf_counter()
        try:
            self.pose_model(blank, verbose=False, device=self.device)
            self.pose_model(blank, verbose=False, device=self.device)
        except Exception:
            if self.device != "cpu":
                self.device = "cpu"
                self.punch_model.to("cpu")
                if self.block_model is not None:
                    self.block_model.to("cpu")
                self.pose_model(blank, verbose=False, device="cpu")
            else:
                raise
        self.warmup_latency_ms = (time.perf_counter() - started) * 1000.0

    def _load_pose(self, path: Path):
        os.environ["ULTRALYTICS_SAFE_LOAD"] = "1"
        os.environ["YOLO_AUTOINSTALL"] = "0"
        from ultralytics import YOLO
        from ultralytics.utils import SAFE_LOAD

        if not SAFE_LOAD:
            raise ClassifierUnavailable(
                "Ultralytics no quedó en carga restringida. No se abrirá "
                f"'{path.name}' con deserialización abierta."
            )
        return YOLO(str(path))

    def _detect_people(self, frame: np.ndarray):
        results = self.pose_model(frame, verbose=False, device=self.device)
        if not results:
            return None
        result = results[0]
        if result.keypoints is None or result.boxes is None or result.boxes.conf is None:
            return None
        data = result.keypoints.data
        if data is None or data.shape[0] == 0:
            return None
        index = int(self._torch.argmax(result.boxes.conf).item())
        person = data[index].detach().cpu().numpy()
        if person.shape[0] < 17 or person.shape[1] < 3:
            return None
        return person[:, :2], person[:, 2]

    def _logits(self, model, sequence: np.ndarray) -> np.ndarray:
        tensor = self._torch.tensor(sequence, dtype=self._torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0)
        with self._torch.no_grad():
            output = model(tensor)
        return output.detach().cpu().numpy().reshape(-1)

    def _predict_punch(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
            raise ValueError(
                "El LSTM externo espera "
                f"{SEQUENCE_LENGTH}x{INPUT_SIZE}, no {sequence.shape}."
            )
        return self._logits(self.punch_model, sequence)

    def _predict_block(self, sequence: np.ndarray) -> np.ndarray:
        if self.block_model is None:
            raise RuntimeError("El modelo defensivo no está cargado.")
        return self._logits(self.block_model, sequence)


def load_external_classifier(model_dir: Path | None = None) -> ExternalClassifier:
    directory = DEFAULT_EXTERNAL_DIR if model_dir is None else model_dir.expanduser().resolve()
    if not directory.is_dir():
        raise ClassifierUnavailable(missing_weights_message(directory))
    return ExternalClassifier(directory)
