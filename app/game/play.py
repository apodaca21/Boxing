"""Bucle de cámara de los dos juegos. Reutiliza el clasificador ya existente."""

from __future__ import annotations

import argparse

import cv2

from app.camera import Camera
from app.config import AppConfig
from app.game.challenge import TargetChallenge
from app.game.draw import draw_challenge, draw_free, draw_menu, shown_point, target_screen
from app.game.free_round import FreeRound
from app.game.store import latest_rounds, save_round
from app.geometry import extract_biomechanical_features
from app.overlay import render_frame
from app.pose_tracker import LandmarkId, PoseTracker
from app.targets.placement import PointSmoother, body_offset
from app.trainer.factory import create_classifier
from main import FrameRateMeter


GAME_WINDOW = "EvoBox AI"
# La cámara puede ser 640x480; la ventana se muestra más grande.
MIN_DISPLAY_WIDTH = 1280


def run_game(config: AppConfig, args: argparse.Namespace) -> int:
    kind = args.classifier
    if kind == "external":
        print("El juego usa NEAT o el detector heurístico.")
        return 2
    config.validate()
    mode = args.game_mode
    while mode is None:
        chosen = _menu_session(config, args)
        if chosen is None:
            return 0
        if chosen == "capture":
            from main import run_targets

            run_targets(config, args)
            continue
        mode = chosen
        args.game_mode = mode
    sensitivity = args.sensitivity or ("play" if kind == "heuristic" else "high")
    classifier = create_classifier(
        kind,
        neat_model=args.neat_model,
        results_dir=args.results_dir,
        sensitivity=sensitivity,
    )
    camera = Camera(
        index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        probe_attempts=config.camera_probe_attempts,
    )
    fps_meter = FrameRateMeter(config.fps_window_size)
    try:
        with PoseTracker(
            model_path=config.model_path,
            min_detection_confidence=config.min_pose_detection_confidence,
            min_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as tracker, camera:
            _open_window(config.frame_width, config.frame_height)
            print(f"Juego {args.game_mode}. Clasificador activo: {classifier.name}.")
            if classifier.name == "neat":
                print("NEAT está activo. Su fitness de entrenamiento no es la precisión de esta partida.")
            else:
                print(
                    "Detector heurístico en sensibilidad PLAY. "
                    "Cuenta al frenar el golpe. NEAT sigue siendo el modelo académico."
                )
            print("ESPACIO inicia la ronda. Q sale. R repite el resumen.")
            duration_ms = int(args.round_seconds * 1000)
            if args.game_mode == "free":
                game: FreeRound | TargetChallenge = FreeRound(classifier.name, duration_ms=duration_ms)
                return _play_loop(config, args, classifier, camera, tracker, fps_meter, game)
            if args.game_mode == "targets":
                game = TargetChallenge(
                    classifier.name,
                    difficulty=args.difficulty,
                    duration_ms=duration_ms,
                )
                return _play_loop(config, args, classifier, camera, tracker, fps_meter, game)
            print("El modo de juego debe ser free o targets.")
            return 2
    finally:
        close = getattr(classifier, "close", None)
        if close is not None:
            close()
        cv2.destroyWindow(GAME_WINDOW)
        cv2.waitKey(1)
    return 0


def _display_size(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return MIN_DISPLAY_WIDTH, int(MIN_DISPLAY_WIDTH * 0.75)
    scale = max(1.0, MIN_DISPLAY_WIDTH / float(width))
    return int(round(width * scale)), int(round(height * scale))


def _open_window(width: int, height: int) -> None:
    display_width, display_height = _display_size(width, height)
    cv2.namedWindow(GAME_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(GAME_WINDOW, display_width, display_height)


def _show(frame: object, width: int, height: int) -> None:
    display_width, display_height = _display_size(width, height)
    if frame.shape[1] != display_width or frame.shape[0] != display_height:
        frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(GAME_WINDOW, frame)


def _menu_session(config: AppConfig, args: argparse.Namespace) -> str | None:
    camera = Camera(
        index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        probe_attempts=config.camera_probe_attempts,
    )
    fps_meter = FrameRateMeter(config.fps_window_size)
    show_stats = False
    choice: str | None = None
    print("Menú: 1 libre, 2 objetivos, 3 captura, 4 estadísticas, Q sale.")
    try:
        with PoseTracker(
            model_path=config.model_path,
            min_detection_confidence=config.min_pose_detection_confidence,
            min_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as tracker, camera:
            _open_window(config.frame_width, config.frame_height)
            while True:
                packet = camera.read()
                pose = tracker.detect(packet.frame, packet.timestamp_ms)
                features = extract_biomechanical_features(
                    pose,
                    min_visibility=config.min_landmark_visibility,
                    min_presence=config.min_landmark_presence,
                    image_size=(packet.width, packet.height),
                )
                display = render_frame(
                    packet.frame, pose, features, fps_meter.tick(), config, overlay=False
                )
                stats: tuple[str, ...] = ()
                if show_stats:
                    rounds = latest_rounds(args.results_dir)
                    stats = tuple(
                        f"{item.get('game')} {item.get('classifier')} total {item.get('total', item.get('score', 0))}"
                        for item in rounds
                    ) or ("No hay partidas guardadas.",)
                draw_menu(display, stats)
                _show(display, packet.width, packet.height)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    choice = None
                    break
                if key == ord("1"):
                    choice = "free"
                    break
                if key == ord("2"):
                    choice = "targets"
                    break
                if key == ord("3"):
                    choice = "capture"
                    break
                if key == ord("4"):
                    show_stats = not show_stats
    finally:
        # Solo cierra la ventana al salir o ir a captura. 1 y 2 la reutilizan.
        if choice not in {"free", "targets"}:
            cv2.destroyWindow(GAME_WINDOW)
            cv2.waitKey(1)
    return choice


def _play_loop(
    config: AppConfig,
    args: argparse.Namespace,
    classifier: object,
    camera: Camera,
    tracker: PoseTracker,
    fps_meter: FrameRateMeter,
    game: FreeRound | TargetChallenge,
) -> int:
    left_smoother = PointSmoother()
    right_smoother = PointSmoother()
    saved = False
    while True:
        packet = camera.read()
        pose = tracker.detect(packet.frame, packet.timestamp_ms)
        features = extract_biomechanical_features(
            pose,
            min_visibility=config.min_landmark_visibility,
            min_presence=config.min_landmark_presence,
            image_size=(packet.width, packet.height),
        )
        result = classifier.update(  # type: ignore[attr-defined]
            features,
            pose_detected=pose.detected,
            pose=pose,
            frame=packet.frame,
        )
        fps = fps_meter.tick()
        display = render_frame(packet.frame, pose, features, fps, config, overlay=False)
        tracking = _tracking(pose)
        if isinstance(game, FreeRound):
            game.tick(packet.timestamp_ms, result)
            draw_free(display, game.snapshot(), fps=fps, tracking=tracking)
            if game.phase == "summary" and not saved:
                path = save_round(args.results_dir, game.record())
                print(f"Resumen de ronda en {path}")
                saved = True
        else:
            wrist, center, radius = _target_geometry(
                pose,
                packet.width,
                packet.height,
                config.mirror_view,
                game,
                left_smoother,
                right_smoother,
                tracking,
            )
            event = result.events[0] if result.events else None
            game.tick(
                packet.timestamp_ms,
                tracking=tracking,
                event_present=event is not None and result.status != "incierto",
                event_label=None if event is None else event.label_id,
                event_key=None if event is None else (event.event_id or f"{packet.timestamp_ms}:{event.label_id}:{event.arm}"),
                wrist=wrist,
                uncertain=result.status == "incierto",
                event_id=None if event is None else event.event_id,
            )
            draw_challenge(
                display,
                game.snapshot(packet.timestamp_ms),
                center=center,
                radius=radius,
                fps=fps,
                tracking=tracking,
            )
            if game.phase == "summary" and not saved:
                path = save_round(args.results_dir, game.record())
                print(f"Resumen de desafío en {path}")
                saved = True
        _show(display, packet.width, packet.height)
        key = cv2.waitKeyEx(1)
        if key >= 0 and game.handle_key(key):
            break
        if game.phase == "idle":
            saved = False
    return 0


def _tracking(pose: object) -> bool:
    left = pose.landmark(LandmarkId.LEFT_SHOULDER)  # type: ignore[attr-defined]
    right = pose.landmark(LandmarkId.RIGHT_SHOULDER)  # type: ignore[attr-defined]
    return bool(
        pose.detected  # type: ignore[attr-defined]
        and left is not None
        and right is not None
        and left.is_finite
        and right.is_finite
    )


def _target_geometry(
    pose: object,
    width: int,
    height: int,
    mirror: bool,
    game: TargetChallenge,
    left_smoother: PointSmoother,
    right_smoother: PointSmoother,
    tracking: bool,
) -> tuple[tuple[float, float] | None, tuple[int, int] | None, int]:
    label = game.label_id
    if not tracking or label is None:
        left_smoother.reset()
        right_smoother.reset()
        return None, None, 0
    left = pose.landmark(LandmarkId.LEFT_SHOULDER)  # type: ignore[attr-defined]
    right = pose.landmark(LandmarkId.RIGHT_SHOULDER)  # type: ignore[attr-defined]
    left_point = left_smoother.update(shown_point(left.x, left.y, width, height, mirror=mirror))
    right_point = right_smoother.update(shown_point(right.x, right.y, width, height, mirror=mirror))
    screen = target_screen(left_point, right_point, label, game.scale, width, height)
    wrist_id = LandmarkId.LEFT_WRIST if label in (1, 3) else LandmarkId.RIGHT_WRIST
    wrist = pose.landmark(wrist_id)  # type: ignore[attr-defined]
    offset = None
    if wrist is not None and wrist.is_finite:
        offset = body_offset(
            shown_point(wrist.x, wrist.y, width, height, mirror=mirror),
            left_point,
            right_point,
        )
    if screen is None:
        return offset, None, 0
    return offset, screen[0], screen[1]
