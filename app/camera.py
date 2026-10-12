"""Adquisición segura de fotogramas mediante OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import time
from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraError(RuntimeError):
    """Error comprensible relacionado con el dispositivo de cámara."""


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Fotograma BGR y metadatos necesarios por el pipeline."""

    frame: NDArray[np.uint8]
    frame_index: int
    timestamp_ms: int
    width: int
    height: int


class Camera:
    """Gestiona VideoCapture y garantiza timestamps crecientes."""

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        probe_attempts: int = 5,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.probe_attempts = probe_attempts
        self._clock_ns = clock_ns
        self._capture: cv2.VideoCapture | None = None
        self._pending_frame: NDArray[np.uint8] | None = None
        self._frame_index = 0
        self._last_timestamp_ms = -1
        self.backend_name: str | None = None

    def _candidate_backends(self) -> list[tuple[int, str]]:
        candidates: list[tuple[int, str]] = []
        if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            candidates.append((cv2.CAP_AVFOUNDATION, "AVFoundation"))
        candidates.append((cv2.CAP_ANY, "backend automático"))
        return candidates

    @staticmethod
    def _valid_frame(frame: object) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.size > 0
            and frame.shape[2] >= 3
        )

    def open(self) -> None:
        """Abre la cámara y comprueba que realmente entregue imágenes."""

        self.release()
        failures: list[str] = []

        for backend, name in self._candidate_backends():
            capture = cv2.VideoCapture(self.index, backend)
            if not capture.isOpened():
                failures.append(f"{name}: no pudo abrirse")
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            first_frame: NDArray[np.uint8] | None = None
            for _ in range(self.probe_attempts):
                ok, candidate = capture.read()
                if ok and self._valid_frame(candidate):
                    first_frame = candidate
                    break

            if first_frame is None:
                failures.append(f"{name}: abrió, pero no entregó fotogramas")
                capture.release()
                continue

            self._capture = capture
            self._pending_frame = first_frame
            self._frame_index = 0
            self._last_timestamp_ms = -1
            self.backend_name = name
            return

        details = "; ".join(failures) if failures else "sin detalles del backend"
        raise CameraError(
            f"No se pudo usar la cámara con índice {self.index} ({details}). "
            "Comprueba que no esté ocupada o desconectada y permite el acceso a "
            "la cámara para Terminal/Python en Ajustes del Sistema > Privacidad "
            "y seguridad > Cámara."
        )

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = self._clock_ns() // 1_000_000
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def read(self) -> FramePacket:
        if self._capture is None or not self._capture.isOpened():
            raise CameraError("La cámara no está abierta.")

        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
        else:
            ok, frame = self._capture.read()
            if not ok or not self._valid_frame(frame):
                raise CameraError(
                    "La cámara dejó de entregar imágenes. Puede haberse "
                    "desconectado, estar ocupada o haber perdido permisos."
                )

        height, width = frame.shape[:2]
        packet = FramePacket(
            frame=frame,
            frame_index=self._frame_index,
            timestamp_ms=self._next_timestamp_ms(),
            width=width,
            height=height,
        )
        self._frame_index += 1
        return packet

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._pending_frame = None
        self.backend_name = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
