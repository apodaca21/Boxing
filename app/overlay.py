"""Dibujo del esqueleto y del HUD deportivo."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import AppConfig
from app.benchmark.protocol import BenchmarkView
from app.data_collector import CollectionView
from app.geometry import BiomechanicalFeatures
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from app.targets.view import TargetView
from app.trainer.session import LiveView


LEFT_ARM_CONNECTIONS = (
    (LandmarkId.LEFT_SHOULDER, LandmarkId.LEFT_ELBOW),
    (LandmarkId.LEFT_ELBOW, LandmarkId.LEFT_WRIST),
)
RIGHT_ARM_CONNECTIONS = (
    (LandmarkId.RIGHT_SHOULDER, LandmarkId.RIGHT_ELBOW),
    (LandmarkId.RIGHT_ELBOW, LandmarkId.RIGHT_WRIST),
)
TORSO_CONNECTIONS = (
    (LandmarkId.LEFT_SHOULDER, LandmarkId.RIGHT_SHOULDER),
    (LandmarkId.LEFT_SHOULDER, LandmarkId.LEFT_HIP),
    (LandmarkId.RIGHT_SHOULDER, LandmarkId.RIGHT_HIP),
    (LandmarkId.LEFT_HIP, LandmarkId.RIGHT_HIP),
)
HEAD_CONNECTIONS = (
    (LandmarkId.LEFT_EAR, LandmarkId.NOSE),
    (LandmarkId.NOSE, LandmarkId.RIGHT_EAR),
)
DISPLAYED_LANDMARKS = tuple(
    dict.fromkeys(
        landmark_id
        for connection in (
            LEFT_ARM_CONNECTIONS
            + RIGHT_ARM_CONNECTIONS
            + TORSO_CONNECTIONS
            + HEAD_CONNECTIONS
        )
        for landmark_id in connection
    )
)


def _visible(
    landmark: LandmarkObservation | None,
    config: AppConfig,
) -> bool:
    return (
        landmark is not None
        and landmark.is_reliable(
            config.min_landmark_visibility,
            config.min_landmark_presence,
        )
    )


def _to_pixel(
    landmark: LandmarkObservation,
    width: int,
    height: int,
    mirrored: bool,
) -> tuple[int, int]:
    normalized_x = 1.0 - landmark.x if mirrored else landmark.x
    x = int(round(np.clip(normalized_x, 0.0, 1.0) * (width - 1)))
    y = int(round(np.clip(landmark.y, 0.0, 1.0) * (height - 1)))
    return x, y


def _draw_connections(
    image: NDArray[np.uint8],
    pose: PoseObservation,
    connections: tuple[tuple[LandmarkId, LandmarkId], ...],
    color: tuple[int, int, int],
    config: AppConfig,
    thickness: int = 4,
) -> None:
    height, width = image.shape[:2]
    for start_id, end_id in connections:
        start = pose.landmark(start_id)
        end = pose.landmark(end_id)
        if not _visible(start, config) or not _visible(end, config):
            continue
        cv2.line(
            image,
            _to_pixel(start, width, height, config.mirror_view),
            _to_pixel(end, width, height, config.mirror_view),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _joint_color(
    landmark_id: LandmarkId,
    config: AppConfig,
) -> tuple[int, int, int]:
    if landmark_id in {
        LandmarkId.LEFT_SHOULDER,
        LandmarkId.LEFT_ELBOW,
        LandmarkId.LEFT_WRIST,
    }:
        return config.color_left
    if landmark_id in {
        LandmarkId.RIGHT_SHOULDER,
        LandmarkId.RIGHT_ELBOW,
        LandmarkId.RIGHT_WRIST,
    }:
        return config.color_right
    return config.color_joint


def _draw_skeleton(
    image: NDArray[np.uint8],
    pose: PoseObservation,
    config: AppConfig,
    highlight_arm: str | None = None,
) -> None:
    _draw_connections(image, pose, TORSO_CONNECTIONS, config.color_torso, config)
    _draw_connections(image, pose, HEAD_CONNECTIONS, config.color_torso, config)
    left_thickness = 8 if highlight_arm == "left" else 4
    right_thickness = 8 if highlight_arm == "right" else 4
    _draw_connections(
        image,
        pose,
        LEFT_ARM_CONNECTIONS,
        config.color_left,
        config,
        left_thickness,
    )
    _draw_connections(
        image,
        pose,
        RIGHT_ARM_CONNECTIONS,
        config.color_right,
        config,
        right_thickness,
    )

    height, width = image.shape[:2]
    for landmark_id in DISPLAYED_LANDMARKS:
        landmark = pose.landmark(landmark_id)
        if not _visible(landmark, config):
            continue
        point = _to_pixel(landmark, width, height, config.mirror_view)
        cv2.circle(image, point, 6, _joint_color(landmark_id, config), -1, cv2.LINE_AA)
        cv2.circle(image, point, 8, (25, 28, 34), 2, cv2.LINE_AA)


def _format_angle(angle: float | None) -> str:
    return "--" if angle is None else f"{angle:5.1f} deg"


def _draw_hud(
    image: NDArray[np.uint8],
    pose: PoseObservation,
    features: BiomechanicalFeatures,
    fps: float | None,
    config: AppConfig,
) -> None:
    height, width = image.shape[:2]
    hud_width = min(330, width)
    hud_height = min(190, height)
    roi = image[:hud_height, :hud_width]
    dark_panel = np.full_like(roi, (17, 20, 27))
    cv2.addWeighted(dark_panel, 0.78, roi, 0.22, 0.0, dst=roi)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(
        image,
        "EvoBox AI",
        (18, 31),
        font,
        0.78,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.line(image, (18, 42), (hud_width - 18, 42), (66, 72, 82), 1, cv2.LINE_AA)

    status = "CUERPO DETECTADO" if pose.detected else "BUSCANDO PERSONA"
    status_color = config.color_success if pose.detected else config.color_searching
    cv2.circle(image, (24, 62), 5, status_color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        status,
        (38, 68),
        font,
        0.46,
        status_color,
        1,
        cv2.LINE_AA,
    )

    fps_text = "--" if fps is None else f"{fps:5.1f}"
    rows = (
        f"FPS          {fps_text}",
        f"Codo izq.    {_format_angle(features.left_elbow_angle)}",
        f"Codo der.    {_format_angle(features.right_elbow_angle)}",
    )
    for row_index, text in enumerate(rows):
        cv2.putText(
            image,
            text,
            (18, 94 + row_index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (222, 226, 232),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        "Presiona Q para salir",
        (18, hud_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (170, 177, 188),
        1,
        cv2.LINE_AA,
    )


def _opencv_text(text: str) -> str:
    """Hershey fonts no dibujan acentos; conserva el texto legible."""

    return text.translate(str.maketrans("áéíóúÁÉÍÓÚñÑüÜ", "aeiouAEIOUnNuU"))


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = _opencv_text(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) > 3:
        lines = lines[:3]
        lines[-1] = lines[-1][: max(0, max_chars - 3)] + "..."
    return lines


def _draw_collection(
    image: NDArray[np.uint8],
    view: CollectionView,
    config: AppConfig,
) -> None:
    height, width = image.shape[:2]
    panel_height = min(156, height)
    panel_top = height - panel_height
    roi = image[panel_top:, :]
    dark_panel = np.full_like(roi, (17, 20, 27))
    cv2.addWeighted(dark_panel, 0.82, roi, 0.18, 0.0, dst=roi)

    if view.outcome == "accepted":
        accent = config.color_success
    elif view.outcome == "rejected":
        accent = (70, 80, 235)
    else:
        accent = (210, 216, 224)
    cv2.rectangle(image, (0, panel_top), (width - 1, height - 1), accent, 1)

    cv2.putText(
        image,
        _opencv_text(view.movement_name),
        (18, panel_top + 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{view.state_label}    Muestras: {view.saved_count}",
        (18, panel_top + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        accent,
        1,
        cv2.LINE_AA,
    )
    max_chars = max(24, (width - 36) // 9)
    for line_index, line in enumerate(_wrap_text(view.message, max_chars)):
        cv2.putText(
            image,
            line,
            (18, panel_top + 80 + line_index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (222, 226, 232),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        view.instructions,
        (18, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (170, 177, 188),
        1,
        cv2.LINE_AA,
    )

    if view.countdown_seconds is not None:
        _draw_centered_status(image, str(view.countdown_seconds), (245, 247, 250), 3.0)
    elif view.state_label == "GRABANDO":
        _draw_centered_status(image, "GRABANDO", (70, 80, 235), 1.1)


def _draw_benchmark(
    image: NDArray[np.uint8],
    view: BenchmarkView,
    config: AppConfig,
) -> None:
    height, width = image.shape[:2]
    panel_height = min(168, height)
    panel_top = height - panel_height
    roi = image[panel_top:, :]
    dark_panel = np.full_like(roi, (17, 20, 27))
    cv2.addWeighted(dark_panel, 0.84, roi, 0.16, 0.0, dst=roi)
    accent = (40, 180, 255) if view.review else config.color_success
    cv2.rectangle(image, (0, panel_top), (width - 1, height - 1), accent, 1)
    cv2.putText(
        image,
        _opencv_text(view.instruction),
        (18, panel_top + 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        _opencv_text(view.progress),
        (18, panel_top + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        accent,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        _opencv_text(f"{view.classifier_name}  {view.sensitivity}  det {view.detected_count}"),
        (18, panel_top + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 177, 188),
        1,
        cv2.LINE_AA,
    )
    max_chars = max(28, (width - 36) // 8)
    for line_index, line in enumerate(_wrap_text(view.detail, max_chars)[:3]):
        cv2.putText(
            image,
            line,
            (18, panel_top + 104 + line_index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (222, 226, 232),
            1,
            cv2.LINE_AA,
        )


def _draw_centered_status(
    image: NDArray[np.uint8],
    text: str,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 6 if scale >= 2 else 2
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    origin = ((width - text_size[0]) // 2, height // 2)
    cv2.putText(image, text, origin, font, scale, (12, 14, 18), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _format_clock(elapsed_ms: int) -> str:
    total_seconds = max(0, elapsed_ms) // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _draw_live(
    image: NDArray[np.uint8],
    pose: PoseObservation,
    fps: float | None,
    view: LiveView,
    config: AppConfig,
) -> None:
    height, width = image.shape[:2]
    panel_width = min(250, max(180, width // 3))
    roi = image[:, :panel_width]
    dark_panel = np.full_like(roi, (12, 14, 18))
    cv2.addWeighted(dark_panel, 0.84, roi, 0.16, 0.0, dst=roi)
    cv2.line(
        image,
        (panel_width, 0),
        (panel_width, height - 1),
        (64, 70, 82),
        1,
        cv2.LINE_AA,
    )

    title_font = cv2.FONT_HERSHEY_DUPLEX
    body_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        image,
        "EVObox AI",
        (16, 32),
        title_font,
        0.72,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )
    if view.source == "external":
        badge = "MODELO: YOLO + LSTM"
        badge_color = (80, 220, 120)
    elif view.source == "neat":
        badge = "MODELO: NEAT"
        badge_color = (80, 220, 120)
    else:
        badge = "MODELO: HEURISTICO"
        badge_color = (70, 175, 255)
    cv2.putText(image, _opencv_text(badge), (16, 54), body_font, 0.38, badge_color, 1, cv2.LINE_AA)
    style = "SPARRING" if view.training_style == "sparring" else "LIBRE"
    cv2.putText(
        image,
        _opencv_text(f"{style}  {view.sensitivity}"),
        (16, 72),
        body_font,
        0.40,
        (80, 220, 120) if view.training_style == "sparring" else (170, 177, 188),
        1,
        cv2.LINE_AA,
    )

    person_detected = pose.detected
    status = "PERSONA DETECTADA" if person_detected else "BUSCANDO PERSONA"
    status_color = config.color_success if person_detected else config.color_searching
    cv2.circle(image, (22, 94), 5, status_color, -1, cv2.LINE_AA)
    cv2.putText(image, status, (34, 99), body_font, 0.42, status_color, 1, cv2.LINE_AA)

    cv2.putText(
        image,
        "CANDIDATO" if view.moving else "GOLPE",
        (16, 124),
        body_font,
        0.40,
        (40, 180, 255) if view.moving else (150, 158, 170),
        1,
        cv2.LINE_AA,
    )
    shown = view.candidate if view.moving else view.recognition
    cv2.putText(
        image,
        _opencv_text(shown),
        (16, 148),
        title_font,
        0.48,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        _opencv_text(f"Ultimo: {view.last_punch}"),
        (16, 170),
        body_font,
        0.42,
        (210, 216, 224),
        1,
        cv2.LINE_AA,
    )

    row_step = 18 if len(view.rows) > 4 else 22
    for index, (name, count, side) in enumerate(view.rows):
        y = 196 + index * row_step
        if side == "left":
            color = config.color_left
        elif side == "right":
            color = config.color_right
        else:
            color = config.color_joint
        cv2.circle(image, (22, y - 4), 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            _opencv_text(name),
            (34, y),
            body_font,
            0.45,
            (226, 230, 236),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(count),
            (panel_width - 42, y),
            title_font,
            0.50,
            (245, 247, 250),
            1,
            cv2.LINE_AA,
        )

    pace = "--" if view.punches_per_minute is None else f"{view.punches_per_minute:4.1f}"
    fps_text = "--" if fps is None else f"{fps:4.1f}"
    latency = "--" if view.latency_ms is None else f"{view.latency_ms:.0f} ms"
    summary = (
        f"Total {view.total}",
        f"Golpes/min {pace}",
        f"Tiempo {_format_clock(view.elapsed_ms)}",
        f"FPS {fps_text}  {latency}",
    )
    summary_y = 196 + len(view.rows) * row_step + 16
    for index, text in enumerate(summary):
        cv2.putText(
            image,
            text,
            (16, summary_y + index * 20),
            body_font,
            0.46,
            (222, 226, 232),
            1,
            cv2.LINE_AA,
        )
    state = "EN MARCHA" if view.running else "EN PAUSA"
    cv2.putText(
        image,
        state,
        (16, height - 36),
        body_font,
        0.42,
        config.color_success if view.running else (170, 177, 188),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "1-3 sens  D debug  C calibra",
        (16, height - 16),
        body_font,
        0.40,
        (170, 177, 188),
        1,
        cv2.LINE_AA,
    )

    if view.moving:
        cv2.circle(image, (panel_width + 18, 28), 7, (40, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            "MOVIMIENTO",
            (panel_width + 32, 33),
            body_font,
            0.45,
            (40, 180, 255),
            1,
            cv2.LINE_AA,
        )
    if view.debug and view.debug_lines:
        for index, line in enumerate(view.debug_lines):
            cv2.putText(
                image,
                _opencv_text(line),
                (panel_width + 12, 120 + index * 18),
                body_font,
                0.40,
                (210, 220, 230),
                1,
                cv2.LINE_AA,
            )
    if view.highlight_arm is not None:
        text = _opencv_text(view.last_punch)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.85
        thickness = 2
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        area_width = max(1, width - panel_width)
        origin = (
            panel_width + max(12, (area_width - text_size[0]) // 2),
            72,
        )
        cv2.putText(image, text, origin, font, scale, (12, 14, 18), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, font, scale, config.color_success, thickness, cv2.LINE_AA)


def _draw_stored_skeleton(image: NDArray[np.uint8], view: TargetView) -> None:
    """Dibuja solo articulaciones presentes. No completa las que faltan."""

    if not view.skeleton_points:
        return
    height, width = image.shape[:2]
    origin_x, origin_y = 250, 48
    box_w, box_h = max(40, width - origin_x - 16), max(40, height - 80)
    cv2.rectangle(image, (origin_x, origin_y), (origin_x + box_w, origin_y + box_h), (22, 26, 32), -1)
    pixels = []
    for _name, x, y in view.skeleton_points:
        pixels.append((
            int(origin_x + 12 + min(max(x, 0.0), 1.0) * (box_w - 24)),
            int(origin_y + 12 + min(max(y, 0.0), 1.0) * (box_h - 24)),
        ))
    for start, end in view.skeleton_links:
        if 0 <= start < len(pixels) and 0 <= end < len(pixels):
            cv2.line(image, pixels[start], pixels[end], (180, 190, 200), 2, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(image, point, 4, (80, 220, 120), -1, cv2.LINE_AA)
    if view.playback_line:
        cv2.putText(
            image,
            _opencv_text(view.playback_line),
            (origin_x + 8, origin_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (230, 234, 238),
            1,
            cv2.LINE_AA,
        )


def _draw_targets(image: NDArray[np.uint8], view: TargetView, fps: float | None) -> None:
    """HUD oscuro del entrenamiento con objetivos, encima de la webcam."""

    height, width = image.shape[:2]
    if view.phase in {"review", "history", "summary"}:
        shade = image.copy()
        cv2.rectangle(shade, (0, 0), (width, height), (10, 12, 16), -1)
        cv2.addWeighted(shade, 0.9, image, 0.1, 0, image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, _opencv_text(view.title), (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (80, 220, 120), 1, cv2.LINE_AA)
        y = 54
        limit = 90 if view.phase == "summary" else 42
        for line in view.review_lines:
            cv2.putText(image, _opencv_text(line)[:limit], (16, y), font, 0.40, (230, 234, 238), 1, cv2.LINE_AA)
            y += 20
            if y > height - 40:
                break
        cv2.putText(
            image,
            _opencv_text(view.status_line)[:90],
            (16, height - 36),
            font,
            0.42,
            (255, 210, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(image, _opencv_text(view.controls)[:110], (12, height - 16), font, 0.38, (168, 174, 182), 1, cv2.LINE_AA)
        _draw_stored_skeleton(image, view)
        return
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (250, height), (12, 14, 18), -1)
    cv2.addWeighted(panel, 0.72, image, 0.28, 0, image)
    cv2.rectangle(image, (0, height - 34), (width, height), (12, 14, 18), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "EVOBOX", (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (80, 220, 120), 1, cv2.LINE_AA)
    cv2.putText(image, _opencv_text(view.title), (16, 52), font, 0.42, (220, 224, 230), 1, cv2.LINE_AA)
    cv2.putText(image, f"Puntos {view.score}", (16, 78), font, 0.55, (245, 248, 252), 1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"Aciertos {view.hits}  Fallos {view.misses}",
        (16, 100),
        font,
        0.42,
        (190, 196, 204),
        1,
        cv2.LINE_AA,
    )
    if view.remaining_ms is not None:
        cv2.putText(
            image,
            f"Tiempo {view.remaining_ms / 1000:.1f}s",
            (16, 122),
            font,
            0.42,
            (190, 196, 204),
            1,
            cv2.LINE_AA,
        )
    combo = _opencv_text(view.combo_text) if view.combo_text else "Combinacion --"
    cv2.putText(image, combo[:28], (16, 148), font, 0.38, (255, 196, 90), 1, cv2.LINE_AA)
    y = 176
    for line in view.progress_lines:
        cv2.putText(image, _opencv_text(line), (16, y), font, 0.38, (210, 214, 220), 1, cv2.LINE_AA)
        y += 18
    track = "Cuerpo OK" if view.tracking else "Sin cuerpo"
    color = (80, 220, 120) if view.tracking else (70, 175, 255)
    cv2.circle(image, (22, y + 4), 5, color, -1, cv2.LINE_AA)
    cv2.putText(image, track, (34, y + 8), font, 0.40, color, 1, cv2.LINE_AA)
    y += 28
    cv2.putText(image, _opencv_text(view.status_line)[:32], (16, y), font, 0.40, (245, 248, 252), 1, cv2.LINE_AA)
    for index, line in enumerate(view.review_lines):
        cv2.putText(
            image,
            _opencv_text(line)[:40],
            (16, min(height - 52, y + 24 + index * 18)),
            font,
            0.36,
            (255, 210, 120),
            1,
            cv2.LINE_AA,
        )
    fps_text = "-- FPS" if fps is None else f"{fps:.1f} FPS"
    cv2.putText(image, fps_text, (16, height - 46), font, 0.40, (170, 176, 184), 1, cv2.LINE_AA)
    cv2.putText(image, _opencv_text(view.controls), (12, height - 12), font, 0.38, (168, 174, 182), 1, cv2.LINE_AA)

    if view.show_ring and view.screen_xy is not None and view.screen_radius > 0:
        center = view.screen_xy
        pulse = view.screen_radius + int(5 * (1.0 - view.fade))
        ring = view.target_color
        cv2.circle(image, center, pulse + 8, ring, 1, cv2.LINE_AA)
        cv2.circle(image, center, max(8, pulse), ring, 2, cv2.LINE_AA)
        cv2.circle(image, center, 4, (245, 248, 252), -1, cv2.LINE_AA)
        if view.target_name:
            cv2.putText(
                image,
                _opencv_text(view.target_name),
                (center[0] + 12, center[1] - 12),
                font,
                0.50,
                ring,
                1,
                cv2.LINE_AA,
            )
    if view.countdown_s is not None:
        text = str(view.countdown_s)
        cv2.putText(image, text, (width // 2 - 12, height // 2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (12, 14, 18), 6, cv2.LINE_AA)
        cv2.putText(image, text, (width // 2 - 12, height // 2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (245, 248, 252), 2, cv2.LINE_AA)
    if view.phase == "idle":
        cv2.putText(
            image,
            "ESPACIO PARA EMPEZAR",
            (width // 2 - 150, height // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (245, 248, 252),
            1,
            cv2.LINE_AA,
        )
    if view.phase == "ready":
        _draw_next_technique(image, view)


def _draw_next_technique(image: NDArray[np.uint8], view: TargetView) -> None:
    height, width = image.shape[:2]
    shade = image.copy()
    cv2.rectangle(shade, (40, height // 2 - 90), (width - 40, height // 2 + 90), (12, 14, 18), -1)
    cv2.addWeighted(shade, 0.82, image, 0.18, 0, image)
    font = cv2.FONT_HERSHEY_DUPLEX
    title = _opencv_text(view.combo_text) or "SIGUE"
    action = "ESPACIO PARA CONTINUAR"
    cv2.putText(image, title[:28], (width // 2 - 180, height // 2 - 16), font, 0.9, view.target_color, 2, cv2.LINE_AA)
    cv2.putText(image, action, (width // 2 - 190, height // 2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 248, 252), 2, cv2.LINE_AA)


def render_frame(
    frame_bgr: NDArray[np.uint8],
    pose: PoseObservation,
    features: BiomechanicalFeatures,
    fps: float | None,
    config: AppConfig,
    collection: CollectionView | None = None,
    live: LiveView | None = None,
    benchmark: BenchmarkView | None = None,
    targets: TargetView | None = None,
) -> NDArray[np.uint8]:
    """Devuelve una vista anotada sin modificar el fotograma de captura."""

    output = (
        cv2.flip(frame_bgr, 1)
        if config.mirror_view
        else np.array(frame_bgr, copy=True)
    )
    if pose.detected:
        highlight = None if live is None else live.highlight_arm
        _draw_skeleton(output, pose, config, highlight)
    if targets is not None:
        _draw_targets(output, targets, fps)
    elif live is not None:
        _draw_live(output, pose, fps, live, config)
    else:
        _draw_hud(output, pose, features, fps, config)
    if collection is not None:
        _draw_collection(output, collection, config)
    if benchmark is not None:
        _draw_benchmark(output, benchmark, config)
    return output
