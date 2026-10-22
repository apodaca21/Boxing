"""Configuración central de EvoBox AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
ORTHODOX_STANCE = "orthodox"
STANCE_DESCRIPTION = (
    "Postura convencional diestra: pie y brazo izquierdos adelantados."
)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Parámetros ajustables de captura, seguimiento y visualización."""

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    mirror_view: bool = True
    model_path: Path = DEFAULT_MODEL_PATH

    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_landmark_visibility: float = 0.55
    min_landmark_presence: float = 0.5

    fps_window_size: int = 30
    camera_probe_attempts: int = 5
    mode: str = "track"
    data_dir: Path = DEFAULT_DATA_DIR
    stance: str = ORTHODOX_STANCE
    countdown_ms: int = 3000
    capture_duration_ms: int = 2000
    min_sample_frames: int = 12
    min_sample_duration_ms: int = 1500
    max_sample_duration_ms: int = 2800
    min_detected_ratio: float = 0.80
    min_reliable_ratio: float = 0.70

    # Colores BGR para OpenCV.
    color_left: tuple[int, int, int] = (50, 210, 255)
    color_right: tuple[int, int, int] = (255, 105, 70)
    color_torso: tuple[int, int, int] = (150, 160, 175)
    color_joint: tuple[int, int, int] = (235, 240, 245)
    color_success: tuple[int, int, int] = (80, 220, 120)
    color_searching: tuple[int, int, int] = (70, 175, 255)

    def validate(self) -> None:
        if self.camera_index < 0:
            raise ValueError("El índice de cámara no puede ser negativo.")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("La resolución de cámara debe ser positiva.")
        if self.fps_window_size < 2:
            raise ValueError("La ventana de FPS debe contener al menos 2 muestras.")
        if self.camera_probe_attempts < 1:
            raise ValueError("Debe realizarse al menos un intento de lectura de cámara.")
        if self.mode not in {
            "track",
            "collect",
            "train",
            "live",
            "benchmark",
            "calibrate",
            "dataset-report",
            "targets",
            "game",
        }:
            raise ValueError(
                "El modo debe ser 'track', 'collect', 'train', 'live', 'benchmark', "
                "'calibrate', 'dataset-report', 'targets' o 'game'."
            )
        if self.stance != ORTHODOX_STANCE:
            raise ValueError("Esta fase solamente admite la postura orthodox.")
        if self.countdown_ms <= 0 or self.capture_duration_ms <= 0:
            raise ValueError("La cuenta regresiva y la captura deben durar más de 0 ms.")
        if self.min_sample_frames < 2:
            raise ValueError("Una muestra requiere al menos 2 fotogramas.")
        if not (
            0 < self.min_sample_duration_ms
            <= self.capture_duration_ms
            <= self.max_sample_duration_ms
        ):
            raise ValueError(
                "La duración objetivo debe estar dentro del rango de captura válido."
            )
        if not 0.0 < self.min_detected_ratio <= 1.0:
            raise ValueError("La proporción mínima de detección debe estar en (0, 1].")
        if not 0.0 < self.min_reliable_ratio <= 1.0:
            raise ValueError("La proporción mínima de visibilidad debe estar en (0, 1].")

        thresholds = {
            "detección": self.min_pose_detection_confidence,
            "presencia": self.min_pose_presence_confidence,
            "seguimiento": self.min_tracking_confidence,
            "visibilidad": self.min_landmark_visibility,
            "presencia de articulación": self.min_landmark_presence,
        }
        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"El umbral de {name} debe estar entre 0 y 1; se recibió {value}."
                )

        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise FileNotFoundError(
                "No se encontró un modelo Pose Landmarker válido en "
                f"'{self.model_path}'. Consulta README.md para descargarlo."
            )
