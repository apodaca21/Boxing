"""Operaciones biomecánicas puras y tolerantes a datos incompletos."""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation


EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class RelativePoint:
    """Posición bidimensional normalizada por la escala corporal."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class BiomechanicalFeatures:
    """Características escalares preparadas para futuras ventanas temporales."""

    timestamp_ms: int
    left_elbow_angle: float | None
    right_elbow_angle: float | None
    left_shoulder_wrist_distance: float | None
    right_shoulder_wrist_distance: float | None
    left_wrist_relative: RelativePoint | None = None
    right_wrist_relative: RelativePoint | None = None
    wrist_separation: float | None = None


def _is_usable(
    point: LandmarkObservation | None,
    min_visibility: float,
    min_presence: float,
) -> bool:
    return point is not None and point.is_reliable(min_visibility, min_presence)


def _xy(
    point: LandmarkObservation,
    image_size: tuple[int, int] | None,
) -> tuple[float, float]:
    if image_size is None:
        return point.x, point.y
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Las dimensiones de imagen deben ser positivas.")
    return point.x * width, point.y * height


def calculate_angle(
    first: LandmarkObservation | None,
    vertex: LandmarkObservation | None,
    third: LandmarkObservation | None,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    """Calcula el ángulo first-vertex-third en grados, entre 0 y 180."""

    points = (first, vertex, third)
    if not all(
        _is_usable(point, min_visibility, min_presence) for point in points
    ):
        return None

    # El chequeo anterior garantiza que estos valores no son None.
    first_xy = _xy(first, image_size)  # type: ignore[arg-type]
    vertex_xy = _xy(vertex, image_size)  # type: ignore[arg-type]
    third_xy = _xy(third, image_size)  # type: ignore[arg-type]

    vector_a = (first_xy[0] - vertex_xy[0], first_xy[1] - vertex_xy[1])
    vector_b = (third_xy[0] - vertex_xy[0], third_xy[1] - vertex_xy[1])
    norm_a = math.hypot(*vector_a)
    norm_b = math.hypot(*vector_b)
    if norm_a <= EPSILON or norm_b <= EPSILON:
        return None

    cosine = (
        vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
    ) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    return angle if math.isfinite(angle) else None


def normalized_distance(
    first: LandmarkObservation | None,
    second: LandmarkObservation | None,
    scale_first: LandmarkObservation | None,
    scale_second: LandmarkObservation | None,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    """Distancia first-second dividida por la distancia de referencia."""

    points = (first, second, scale_first, scale_second)
    if not all(
        _is_usable(point, min_visibility, min_presence) for point in points
    ):
        return None

    first_xy = _xy(first, image_size)  # type: ignore[arg-type]
    second_xy = _xy(second, image_size)  # type: ignore[arg-type]
    scale_first_xy = _xy(scale_first, image_size)  # type: ignore[arg-type]
    scale_second_xy = _xy(scale_second, image_size)  # type: ignore[arg-type]

    distance = math.dist(first_xy, second_xy)
    scale = math.dist(scale_first_xy, scale_second_xy)
    if scale <= EPSILON:
        return None

    value = distance / scale
    return value if math.isfinite(value) else None


def left_elbow_angle(
    pose: PoseObservation,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    return calculate_angle(
        pose.landmark(LandmarkId.LEFT_SHOULDER),
        pose.landmark(LandmarkId.LEFT_ELBOW),
        pose.landmark(LandmarkId.LEFT_WRIST),
        min_visibility=min_visibility,
        min_presence=min_presence,
        image_size=image_size,
    )


def right_elbow_angle(
    pose: PoseObservation,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    return calculate_angle(
        pose.landmark(LandmarkId.RIGHT_SHOULDER),
        pose.landmark(LandmarkId.RIGHT_ELBOW),
        pose.landmark(LandmarkId.RIGHT_WRIST),
        min_visibility=min_visibility,
        min_presence=min_presence,
        image_size=image_size,
    )


def left_shoulder_wrist_distance(
    pose: PoseObservation,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    return normalized_distance(
        pose.landmark(LandmarkId.LEFT_SHOULDER),
        pose.landmark(LandmarkId.LEFT_WRIST),
        pose.landmark(LandmarkId.LEFT_SHOULDER),
        pose.landmark(LandmarkId.RIGHT_SHOULDER),
        min_visibility=min_visibility,
        min_presence=min_presence,
        image_size=image_size,
    )


def normalized_relative_position(
    point: LandmarkObservation | None,
    origin_first: LandmarkObservation | None,
    origin_second: LandmarkObservation | None,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> RelativePoint | None:
    """Posición de un punto respecto al punto medio, dividida por la escala."""

    points = (point, origin_first, origin_second)
    if not all(
        _is_usable(candidate, min_visibility, min_presence) for candidate in points
    ):
        return None

    point_xy = _xy(point, image_size)  # type: ignore[arg-type]
    origin_first_xy = _xy(origin_first, image_size)  # type: ignore[arg-type]
    origin_second_xy = _xy(origin_second, image_size)  # type: ignore[arg-type]
    scale = math.dist(origin_first_xy, origin_second_xy)
    if scale <= EPSILON:
        return None

    midpoint = (
        (origin_first_xy[0] + origin_second_xy[0]) / 2,
        (origin_first_xy[1] + origin_second_xy[1]) / 2,
    )
    x = (point_xy[0] - midpoint[0]) / scale
    y = (point_xy[1] - midpoint[1]) / scale
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return RelativePoint(x, y)


def left_wrist_relative(
    pose: PoseObservation,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> RelativePoint | None:
    return normalized_relative_position(
        pose.landmark(LandmarkId.LEFT_WRIST),
        pose.landmark(LandmarkId.LEFT_SHOULDER),
        pose.landmark(LandmarkId.RIGHT_SHOULDER),
        min_visibility=min_visibility,
        min_presence=min_presence,
        image_size=image_size,
    )


def right_wrist_relative(
    pose: PoseObservation,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> RelativePoint | None:
    return normalized_relative_position(
        pose.landmark(LandmarkId.RIGHT_WRIST),
        pose.landmark(LandmarkId.LEFT_SHOULDER),
        pose.landmark(LandmarkId.RIGHT_SHOULDER),
        min_visibility=min_visibility,
        min_presence=min_presence,
        image_size=image_size,
    )


def wrist_separation(
    pose: PoseObservation,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    return normalized_distance(
        pose.landmark(LandmarkId.LEFT_WRIST),
        pose.landmark(LandmarkId.RIGHT_WRIST),
        pose.landmark(LandmarkId.LEFT_SHOULDER),
        pose.landmark(LandmarkId.RIGHT_SHOULDER),
        min_visibility=min_visibility,
        min_presence=min_presence,
        image_size=image_size,
    )


def right_shoulder_wrist_distance(
    pose: PoseObservation,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    return normalized_distance(
        pose.landmark(LandmarkId.RIGHT_SHOULDER),
        pose.landmark(LandmarkId.RIGHT_WRIST),
        pose.landmark(LandmarkId.LEFT_SHOULDER),
        pose.landmark(LandmarkId.RIGHT_SHOULDER),
        min_visibility=min_visibility,
        min_presence=min_presence,
        image_size=image_size,
    )


def extract_biomechanical_features(
    pose: PoseObservation,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
    image_size: tuple[int, int] | None = None,
) -> BiomechanicalFeatures:
    """Construye un vector semántico sin aplicar reglas de clasificación."""

    common = {
        "min_visibility": min_visibility,
        "min_presence": min_presence,
        "image_size": image_size,
    }
    return BiomechanicalFeatures(
        timestamp_ms=pose.timestamp_ms,
        left_elbow_angle=left_elbow_angle(pose, **common),
        right_elbow_angle=right_elbow_angle(pose, **common),
        left_shoulder_wrist_distance=left_shoulder_wrist_distance(pose, **common),
        right_shoulder_wrist_distance=right_shoulder_wrist_distance(pose, **common),
        left_wrist_relative=left_wrist_relative(pose, **common),
        right_wrist_relative=right_wrist_relative(pose, **common),
        wrist_separation=wrist_separation(pose, **common),
    )
