"""Herramientas auxiliares de visión para la cámara RGB externa.

Este paquete no participa en la decisión del clasificador ni en el ledger de
golpes. Sus salidas son señales de observabilidad que pueden consumirse más
adelante por biomecánica, sesiones y mapas de calor.
"""

from app.vision_lab.background_model import BackgroundModel
from app.vision_lab.biomechanics import calculate_elbow_angle, calculate_kinematics
from app.vision_lab.heatmaps import HeatmapAccumulator, accumulate_motion
from app.vision_lab.motion_detector import MotionDetector
from app.vision_lab.segmentation import ForegroundSegmenter, segment_foreground
from app.vision_lab.session_recorder import SessionRecorder, SessionReport

__all__ = [
    "BackgroundModel",
    "ForegroundSegmenter",
    "HeatmapAccumulator",
    "MotionDetector",
    "SessionRecorder",
    "SessionReport",
    "accumulate_motion",
    "calculate_elbow_angle",
    "calculate_kinematics",
    "segment_foreground",
]
