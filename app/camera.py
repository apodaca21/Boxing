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


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """Resultado de una prueba de cámara sin conservar el dispositivo abierto."""

    index: int
    available: bool
    width: int | None = None
    height: int | None = None
    backend: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RigFrame:
    """Fotogramas del frontal y de las cámaras auxiliares en un instante."""

    primary: FramePacket
    by_role: dict[str, FramePacket]

    @property
    def auxiliary(self) -> dict[str, FramePacket]:
        return {role: packet for role, packet in self.by_role.items() if role != "front"}


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


def probe_cameras(
    *,
    max_index: int = 8,
    width: int = 640,
    height: int = 480,
    probe_attempts: int = 2,
) -> tuple[CameraInfo, ...]:
    """Enumera cámaras que realmente entregan un fotograma.

    Los índices de OpenCV no son nombres estables de hardware. Esta función es
    deliberadamente una prueba corta: sirve para elegir roles en el arranque,
    pero la selección final debe guardarse por índice en la configuración.
    """

    if max_index < 0:
        raise ValueError("El índice máximo de búsqueda no puede ser negativo.")
    found: list[CameraInfo] = []
    for index in range(max_index + 1):
        camera = Camera(
            index=index,
            width=width,
            height=height,
            probe_attempts=probe_attempts,
        )
        try:
            camera.open()
        except CameraError as error:
            found.append(CameraInfo(index=index, available=False, error=str(error)))
        else:
            try:
                packet = camera.read()
            except CameraError as error:
                found.append(CameraInfo(index=index, available=False, error=str(error)))
            else:
                found.append(
                    CameraInfo(
                        index=index,
                        available=True,
                        width=packet.width,
                        height=packet.height,
                        backend=camera.backend_name,
                    )
                )
        finally:
            camera.release()
    return tuple(found)


class CameraRig:
    """Captura frontal y hasta tres vistas con una API compatible con Camera.

    ``front`` siempre alimenta el detector actual. Las vistas ``left``,
    ``right`` y ``depth`` se conservan en ``last_frame.auxiliary`` para que un
    módulo de fusión pueda usarlas sin cambiar el contrato de los modos
    existentes. Las cámaras auxiliares no se mezclan geométricamente de forma
    automática: para eso hace falta calibración extrínseca real.
    """

    ROLES = ("front", "left", "right", "depth")

    def __init__(
        self,
        *,
        front_index: int = 0,
        left_index: int | None = None,
        right_index: int | None = None,
        depth_index: int | None = None,
        setup: str = "single",
        max_scan_index: int = 8,
        width: int = 640,
        height: int = 480,
        probe_attempts: int = 5,
    ) -> None:
        self.front_index = front_index
        self.requested_indices = {
            "front": front_index,
            "left": left_index,
            "right": right_index,
            "depth": depth_index,
        }
        self.setup = setup
        self.max_scan_index = max_scan_index
        self.width = width
        self.height = height
        self.probe_attempts = probe_attempts
        self._cameras: dict[str, Camera] = {}
        self._last_frame: RigFrame | None = None
        self._resolved_indices: dict[str, int] = {}

    @property
    def last_frame(self) -> RigFrame | None:
        return self._last_frame

    @property
    def role_indices(self) -> dict[str, int]:
        return dict(self._resolved_indices)

    @property
    def backend_name(self) -> str | None:
        camera = self._cameras.get("front")
        return None if camera is None else camera.backend_name

    def _resolve_indices(self) -> dict[str, int]:
        requested = {
            role: index
            for role, index in self.requested_indices.items()
            if index is not None
        }
        if self.setup == "single":
            return {"front": int(self.front_index)}

        available = {
            info.index
            for info in probe_cameras(
                max_index=self.max_scan_index,
                width=self.width,
                height=self.height,
                probe_attempts=min(2, self.probe_attempts),
            )
            if info.available
        }
        available.discard(self.front_index)
        used = set(requested.values())
        candidates = sorted(available - used)

        def choose(role: str) -> None:
            if role in requested:
                return
            if candidates:
                requested[role] = candidates.pop(0)

        if self.setup in {"stereo", "triple", "auto"}:
            choose("left")
        if self.setup in {"triple", "auto"}:
            choose("right")
        if self.setup == "depth":
            choose("depth")

        if self.setup == "depth" and "depth" not in requested:
            raise CameraError(
                "El modo depth necesita una cámara auxiliar disponible. "
                "Indícala con --camera-depth-index o conecta otra webcam."
            )
        if self.setup in {"stereo", "triple"}:
            required = ("left",) if self.setup == "stereo" else ("left", "right")
            missing = [role for role in required if role not in requested]
            if missing:
                raise CameraError(
                    "No se pudieron asignar las cámaras "
                    + ", ".join(missing)
                    + ". Usa --camera-left-index/--camera-right-index."
                )

        resolved = {"front": int(self.front_index)}
        resolved.update({role: int(index) for role, index in requested.items() if role != "front"})
        if len(set(resolved.values())) != len(resolved):
            raise CameraError("Cada rol de cámara debe usar un índice distinto.")
        return resolved

    def open(self) -> None:
        self.release()
        self._resolved_indices = self._resolve_indices()
        failures: list[str] = []
        opened: dict[str, Camera] = {}
        try:
            for role, index in self._resolved_indices.items():
                camera = Camera(
                    index=index,
                    width=self.width,
                    height=self.height,
                    probe_attempts=self.probe_attempts,
                )
                try:
                    camera.open()
                except CameraError as error:
                    failures.append(f"{role}={index}: {error}")
                    camera.release()
                    continue
                opened[role] = camera
            if "front" not in opened:
                raise CameraError("La cámara frontal no pudo abrirse.")
            if failures:
                raise CameraError("No se pudieron abrir todas las cámaras: " + "; ".join(failures))
            self._cameras = opened
            self._last_frame = None
        except Exception:
            for camera in opened.values():
                camera.release()
            self._cameras.clear()
            raise

    def read(self) -> FramePacket:
        """Lee el frontal y actualiza las auxiliares; devuelve el frontal."""

        if not self._cameras:
            raise CameraError("El rig de cámaras no está abierto.")
        packets = {role: camera.read() for role, camera in self._cameras.items()}
        primary = packets["front"]
        self._last_frame = RigFrame(primary=primary, by_role=packets)
        return primary

    def read_rig(self) -> RigFrame:
        self.read()
        assert self._last_frame is not None
        return self._last_frame

    def release(self) -> None:
        for camera in self._cameras.values():
            camera.release()
        self._cameras.clear()
        self._last_frame = None

    def swap_front_depth(self) -> tuple[int, int]:
        """Intercambia los índices frontal y depth, reabriendo el rig."""

        front = self._resolved_indices.get("front", self.front_index)
        depth = self._resolved_indices.get("depth", self.requested_indices.get("depth"))
        if depth is None:
            raise CameraError("No hay cámara depth para intercambiar.")
        self.front_index = depth
        self.requested_indices = {
            **self.requested_indices,
            "front": depth,
            "depth": front,
        }
        self.setup = "depth"
        self.open()
        return depth, front

    def __enter__(self) -> CameraRig:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
