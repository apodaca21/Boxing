"""Punto de entrada de EvoBox AI."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from collections.abc import Callable
from pathlib import Path
import sys
import time

import cv2

from app.camera import Camera, CameraError
from app.config import DEFAULT_DATA_DIR, PROJECT_ROOT, AppConfig, DEFAULT_MODEL_PATH
from app.data_collector import DataCollector
from app.dataset_manager import DatasetError, DatasetManager
from app.geometry import extract_biomechanical_features
from app.overlay import render_frame
from app.pose_tracker import PoseTracker
from app.trainer.factory import CLASSIFIER_NAMES
from app.trainer.neat_classifier import ClassifierUnavailable
from app.trainer.session import TrainingSession


WINDOW_TITLE = "EvoBox AI — Pose Tracking"
LIVE_WINDOW_TITLE = "EvoBox AI — Entrenador"


class FrameRateMeter:
    """FPS real calculado sobre instantes monotónicos recientes."""

    def __init__(
        self,
        window_size: int = 30,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if window_size < 2:
            raise ValueError("La ventana de FPS debe tener al menos 2 muestras.")
        self._samples: deque[int] = deque(maxlen=window_size)
        self._clock_ns = clock_ns

    def tick(self) -> float | None:
        now = self._clock_ns()
        self._samples.append(now)
        if len(self._samples) < 2:
            return None
        elapsed_ns = self._samples[-1] - self._samples[0]
        if elapsed_ns <= 0:
            return None
        return (len(self._samples) - 1) * 1_000_000_000 / elapsed_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seguimiento corporal local en tiempo real para EvoBox AI."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Índice de VideoCapture (predeterminado: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Ancho solicitado a la cámara (predeterminado: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Alto solicitado a la cámara (predeterminado: 480).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Muestra la imagen sin reflejarla horizontalmente.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta a un modelo Pose Landmarker .task.",
    )
    parser.add_argument(
        "--mode",
        choices=("track", "collect", "train", "live", "benchmark", "dataset-report", "targets", "game"),
        default="track",
        help="track, collect, train, live, benchmark, dataset-report, targets o game.",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_NAMES,
        default="heuristic",
        help="Clasificador del modo live: heuristic, neat o external.",
    )
    parser.add_argument(
        "--neat-model",
        type=Path,
        help="Directorio del genoma ganador. Solo se usa con --classifier neat.",
    )
    parser.add_argument(
        "--external-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "external",
        help="Directorio con punch_classifier.pth y yolo11m-pose.pt.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directorio local del dataset (predeterminado: data/).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directorio local para auditorías y resultados de entrenamiento.",
    )
    parser.add_argument(
        "--experiment",
        choices=("quick", "initial"),
        default="quick",
        help="quick usa 30 redes y 5 generaciones; initial usa 100 y 50.",
    )
    parser.add_argument("--population", type=int, help="Reemplaza la población del experimento.")
    parser.add_argument("--generations", type=int, help="Reemplaza las generaciones del experimento.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument(
        "--time-steps",
        type=int,
        default=8,
        help="Puntos temporales fijos de cada muestra.",
    )
    parser.add_argument(
        "--split",
        choices=("session", "person"),
        default="session",
        help="Unidad de división. person solo está disponible si los datos la contienen.",
    )
    parser.add_argument(
        "--game-mode",
        choices=("free", "targets"),
        help="free o targets. Sin este argumento, --mode game abre el menú.",
    )
    parser.add_argument(
        "--difficulty",
        choices=("easy", "normal", "hard"),
        default="normal",
        help="Tamaño y tiempo de los objetivos. No cambia el criterio de detección.",
    )
    parser.add_argument(
        "--round-seconds",
        type=float,
        default=90,
        help="Duración de la ronda de juego, en segundos.",
    )
    parser.add_argument(
        "--review-flagged",
        action="store_true",
        help="Abre en el reproductor las muestras alineadas al bloque sin revisión visual.",
    )
    parser.add_argument(
        "--training-style",
        choices=("libre", "sparring"),
        default="libre",
        help="sparring prioriza combinaciones y usa sensibilidad high si no se indica otra.",
    )
    parser.add_argument(
        "--sensitivity",
        choices=("normal", "high", "maximum"),
        help="Perfil del detector. En sparring, el predeterminado es high. En benchmark, normal.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Repeticiones por clase en la evaluación.",
    )
    parser.add_argument("--prepare-ms", type=int, default=1500, help="Pausa de preparación.")
    parser.add_argument(
        "--active-ms",
        type=int,
        default=2000,
        help="Duración del intervalo en el que se evalúa y se puede guardar la ventana.",
    )
    parser.add_argument("--return-ms", type=int, default=1000, help="Pausa para volver a guardia.")
    parser.add_argument("--next-ms", type=int, default=800, help="Aviso antes de la siguiente repetición.")
    parser.add_argument(
        "--match-tolerance-ms",
        type=int,
        default=400,
        help="Holgura temporal para emparejar una predicción con un golpe verificado.",
    )
    parser.add_argument(
        "--no-save-samples",
        action="store_true",
        help="No guarda ventanas biomecánicas aunque la persona verifique la repetición.",
    )
    parser.add_argument(
        "--training-order",
        choices=("mixed", "blocked"),
        default="mixed",
        help="mixed alterna las clases. blocked practica un solo golpe por bloque.",
    )
    parser.add_argument(
        "--only-class",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=None,
        help="En blocked, practica solo esa clase: 0 guardia, 1 jab, 2 cross, 3 gancho izquierdo, 4 gancho derecho.",
    )
    parser.add_argument(
        "--target-goal",
        type=int,
        default=10,
        help="Intentos por clase en blocked, o meta verificada por clase en mixed.",
    )
    parser.add_argument(
        "--round-size",
        type=int,
        default=8,
        help="Objetivos de una ronda antes de la revisión.",
    )
    parser.add_argument(
        "--review-verified",
        action="store_true",
        help="Reabre muestras ya guardadas cuya etiqueta no coincide con la pedida.",
    )
    parser.add_argument(
        "--review-session",
        help="Abre una sesión ya guardada en data/review sin repetir los golpes.",
    )
    return parser.parse_args()


def run(config: AppConfig) -> None:
    config.validate()
    fps_meter = FrameRateMeter(config.fps_window_size)

    camera = Camera(
        index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        probe_attempts=config.camera_probe_attempts,
    )

    try:
        with PoseTracker(
            model_path=config.model_path,
            min_detection_confidence=config.min_pose_detection_confidence,
            min_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as tracker, camera:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

            while True:
                packet = camera.read()
                pose = tracker.detect(packet.frame, packet.timestamp_ms)
                image_size = (packet.width, packet.height)
                features = extract_biomechanical_features(
                    pose,
                    min_visibility=config.min_landmark_visibility,
                    min_presence=config.min_landmark_presence,
                    image_size=image_size,
                )
                fps = fps_meter.tick()
                display = render_frame(packet.frame, pose, features, fps, config)
                cv2.imshow(WINDOW_TITLE, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def run_collection(config: AppConfig) -> None:
    config.validate()
    fps_meter = FrameRateMeter(config.fps_window_size)
    dataset = DatasetManager(config.data_dir)
    collector = DataCollector(config, dataset)
    camera = Camera(
        index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        probe_attempts=config.camera_probe_attempts,
    )
    print(f"Recopilación local en {config.data_dir}")
    print("Postura orthodox. Elige la clase con 0-4 y graba con Espacio.")

    try:
        with PoseTracker(
            model_path=config.model_path,
            min_detection_confidence=config.min_pose_detection_confidence,
            min_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as tracker, camera:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

            while True:
                packet = camera.read()
                pose = tracker.detect(packet.frame, packet.timestamp_ms)
                image_size = (packet.width, packet.height)
                features = extract_biomechanical_features(
                    pose,
                    min_visibility=config.min_landmark_visibility,
                    min_presence=config.min_landmark_presence,
                    image_size=image_size,
                )
                message = collector.advance(
                    packet.timestamp_ms,
                    pose,
                    features,
                    packet.width,
                    packet.height,
                )
                if message is not None:
                    print(message)
                fps = fps_meter.tick()
                display = render_frame(
                    packet.frame,
                    pose,
                    features,
                    fps,
                    config,
                    collector.view(),
                )
                cv2.imshow(WINDOW_TITLE, display)

                key = cv2.waitKey(1) & 0xFF
                if collector.handle_key(key, packet.timestamp_ms):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"Muestras guardadas en esta sesión: {collector.saved_count}")


def run_live(config: AppConfig, args: argparse.Namespace) -> int:
    from app.trainer.calibration import MotionCalibrator
    from app.trainer.diagnostics import DiagnosticLog
    from app.trainer.factory import create_classifier

    sensitivity = args.sensitivity
    if sensitivity is None:
        sensitivity = "high" if args.training_style == "sparring" else "normal"
    classifier = create_classifier(
        args.classifier,
        neat_model=args.neat_model,
        results_dir=args.results_dir,
        external_dir=args.external_model,
        sensitivity=sensitivity,
    )
    tune = getattr(classifier, "set_sensitivity", None)
    if tune is not None and classifier.name != "heuristic":
        tune(sensitivity)
    config.validate()
    session = TrainingSession()
    session.source = classifier.name
    session.provisional = classifier.provisional
    session.training_style = args.training_style
    session.sensitivity = sensitivity.upper()
    calibrator = MotionCalibrator()
    diagnostics = DiagnosticLog()
    default_sensitivity = sensitivity
    fps_meter = FrameRateMeter(config.fps_window_size)
    camera = Camera(
        index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        probe_attempts=config.camera_probe_attempts,
    )
    if classifier.name == "external":
        print("Entrenador en vivo con YOLO Pose y el LSTM preentrenado.")
        print("Las activaciones no se muestran como probabilidades calibradas.")
    elif classifier.name == "neat":
        print("Entrenador en vivo con el clasificador NEAT guardado.")
    else:
        print("Entrenador en vivo con detector provisional.")
        print("Los umbrales son heurísticos y todavía necesitan calibración.")
    if args.training_style == "sparring":
        print("Modo sparring. Cada brazo confirma al frenar, sin esperar la guardia exacta.")
    print(f"Sensibilidad inicial: {sensitivity}.")
    print("1 normal, 2 alta, 3 máxima, 0 restaura, D diagnóstico, C calibración.")
    print("P pausa o reanuda, R reinicia y Q cierra. La sesión ya está contando.")

    try:
        with PoseTracker(
            model_path=config.model_path,
            min_detection_confidence=config.min_pose_detection_confidence,
            min_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as tracker, camera:
            cv2.namedWindow(LIVE_WINDOW_TITLE, cv2.WINDOW_NORMAL)
            while True:
                packet = camera.read()
                pose = tracker.detect(packet.frame, packet.timestamp_ms)
                features = extract_biomechanical_features(
                    pose,
                    min_visibility=config.min_landmark_visibility,
                    min_presence=config.min_landmark_presence,
                    image_size=(packet.width, packet.height),
                )
                result = classifier.update(
                    features,
                    pose_detected=pose.detected,
                    pose=pose,
                    frame=packet.frame,
                )
                if calibrator.active and classifier.name == "heuristic":
                    calibrator.observe(packet.timestamp_ms, classifier.telemetry)
                    if calibrator.finished(packet.timestamp_ms):
                        proposed = calibrator.propose(classifier.settings)
                        classifier.apply_settings(proposed)
                        calibrator.save(proposed)
                        print("Calibración guardada en results/. No se crearon muestras para NEAT.")
                    else:
                        result = replace(
                            result,
                            events=(),
                            display_name=calibrator.prompt(packet.timestamp_ms),
                            candidate="CALIBRANDO",
                            moving=True,
                        )
                session.advance(packet.timestamp_ms, result)
                fps = fps_meter.tick()
                if result.counted or result.status == "unknown" or result.rejection:
                    kind = "confirmado" if result.counted else (
                        "no_identificado" if result.status == "unknown" else "rechazo"
                    )
                    diagnostics.write(
                        {
                            "timestamp_ms": packet.timestamp_ms,
                            "kind": kind,
                            "source": result.source,
                            "display_name": result.display_name,
                            "candidate": result.candidate,
                            "rejection": result.rejection,
                            "sensitivity": session.sensitivity,
                            "latency_ms": None if result.latency_ms is None else round(result.latency_ms, 2),
                            "fps": None if fps is None else round(fps, 2),
                            "contains_video": False,
                        }
                    )
                display = render_frame(
                    packet.frame,
                    pose,
                    features,
                    fps,
                    config,
                    live=session.view(packet.timestamp_ms),
                )
                cv2.imshow(LIVE_WINDOW_TITLE, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("1"), ord("2"), ord("3"), ord("0")):
                    chosen = {"1": "normal", "2": "high", "3": "maximum", "0": default_sensitivity}[chr(key)]
                    tuner = getattr(classifier, "set_sensitivity", None)
                    if tuner is None:
                        print("Este clasificador no tiene perfiles de sensibilidad.")
                    else:
                        tuner(chosen)
                        session.sensitivity = chosen.upper()
                elif key in (ord("d"), ord("D")):
                    session.debug = not session.debug
                elif key in (ord("c"), ord("C")):
                    if classifier.name == "heuristic":
                        calibrator.start(packet.timestamp_ms)
                        print("Calibración de 14 s. Los golpes de esta rutina no se guardan para NEAT.")
                    else:
                        print("La calibración de umbrales aplica al detector heurístico.")
                if session.handle_key(key):
                    break
    finally:
        close = getattr(classifier, "close", None)
        if close is not None:
            close()
        camera.release()
        cv2.destroyAllWindows()
    return 0


def run_benchmark(config: AppConfig, args: argparse.Namespace) -> int:
    from datetime import datetime

    from app.benchmark.protocol import BenchmarkProtocol
    from app.benchmark.report import summary_text, update_comparison, write_session_report
    from app.benchmark.samples import VerifiedSampleStore
    from app.data_collector import build_frame_record
    from app.dataset_manager import DatasetManager
    from app.trainer.factory import create_classifier

    if args.classifier == "neat":
        print(
            "La evaluación usa el detector heurístico o el modelo externo. "
            "NEAT sigue disponible en --mode live y --mode train.",
            file=sys.stderr,
        )
        return 2
    sensitivity = args.sensitivity or "normal"
    classifier = create_classifier(
        args.classifier,
        neat_model=args.neat_model,
        results_dir=args.results_dir,
        external_dir=args.external_model,
        sensitivity=sensitivity,
    )
    tune = getattr(classifier, "set_sensitivity", None)
    if tune is not None and classifier.name != "heuristic":
        tune(sensitivity)
    config.validate()
    if not (
        config.min_sample_duration_ms
        <= args.active_ms
        <= config.max_sample_duration_ms
    ):
        print(
            "El intervalo activo está fuera de 1.5 a 2.8 s. "
            "La evaluación continúa, pero esas ventanas no cumplen el formato de NEAT."
        )
    protocol = BenchmarkProtocol(
        repetitions=args.repetitions,
        prepare_ms=args.prepare_ms,
        active_ms=args.active_ms,
        return_ms=args.return_ms,
        next_ms=args.next_ms,
    )
    store = None
    if not args.no_save_samples:
        store = VerifiedSampleStore(config, DatasetManager(config.data_dir))
    frames: list = []
    fps_meter = FrameRateMeter(config.fps_window_size)
    camera = Camera(
        index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        probe_attempts=config.camera_probe_attempts,
    )
    print("Evaluación guiada. La consigna no se toma como un acierto.")
    print("Y confirma la ejecución, 0-4 corrige la etiqueta, X excluye y Q termina.")
    print(f"Sensibilidad {sensitivity}. Repeticiones por clase: {args.repetitions}.")

    try:
        with PoseTracker(
            model_path=config.model_path,
            min_detection_confidence=config.min_pose_detection_confidence,
            min_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as tracker, camera:
            cv2.namedWindow(LIVE_WINDOW_TITLE, cv2.WINDOW_NORMAL)
            while True:
                packet = camera.read()
                pose = tracker.detect(packet.frame, packet.timestamp_ms)
                features = extract_biomechanical_features(
                    pose,
                    min_visibility=config.min_landmark_visibility,
                    min_presence=config.min_landmark_presence,
                    image_size=(packet.width, packet.height),
                )
                previous = protocol.phase
                protocol.advance(packet.timestamp_ms)
                if protocol.phase == "execute" and previous != "execute":
                    frames = []
                result = classifier.update(
                    features,
                    pose_detected=pose.detected,
                    pose=pose,
                    frame=packet.frame,
                )
                fps = fps_meter.tick()
                if protocol.phase == "execute":
                    frames.append(
                        build_frame_record(config, packet.timestamp_ms, pose, features)
                    )
                    protocol.observe(
                        packet.timestamp_ms,
                        result.events,
                        latency_ms=result.latency_ms,
                        fps=fps,
                    )
                display = render_frame(
                    packet.frame,
                    pose,
                    features,
                    fps,
                    config,
                    benchmark=protocol.view(
                        sensitivity=sensitivity,
                        classifier_name=classifier.name,
                    ),
                )
                cv2.imshow(LIVE_WINDOW_TITLE, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if protocol.phase == "review" and key in (ord("y"), ord("Y"), ord(" "), 13):
                    verified = protocol.confirm()
                    if verified is not None and store is not None:
                        outcome = store.save(
                            verified,
                            frames,
                            width=packet.width,
                            height=packet.height,
                        )
                        if outcome.saved:
                            print(f"Ventana verificada guardada: {outcome.sample_id}")
                        else:
                            print("La ventana no se guardó: " + "; ".join(outcome.reasons))
                    frames = []
                elif protocol.phase == "review" and ord("0") <= key <= ord("4"):
                    protocol.correct(key - ord("0"))
                elif protocol.phase == "review" and key in (ord("x"), ord("X")):
                    protocol.exclude()
                    frames = []
                if protocol.finished:
                    break
    finally:
        close = getattr(classifier, "close", None)
        if close is not None:
            close()
        camera.release()
        cv2.destroyAllWindows()

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    directory = args.results_dir / "benchmarks" / sensitivity / stamp
    saved_ids = () if store is None else tuple(store.saved_ids)
    rejections = () if store is None else tuple(store.rejected)
    metrics = write_session_report(
        protocol,
        directory,
        classifier_name=classifier.name,
        sensitivity=sensitivity,
        tolerance_ms=args.match_tolerance_ms,
        saved_sample_ids=saved_ids,
        sample_rejections=rejections,
    )
    comparison = update_comparison(args.results_dir / "benchmarks")
    print(summary_text(metrics, sensitivity, classifier.name))
    print(f"Informe guardado en {directory}")
    if comparison is not None:
        print(f"Comparación de perfiles en {comparison}")
        print("Ningún perfil se marca como ganador por haber contado más golpes.")
    return 0


def run_targets(config: AppConfig, args: argparse.Namespace) -> int:
    """Objetivos sobre la webcam. Las muestras quedan provisionales hasta la revisión."""

    from app.data_collector import build_frame_record
    from app.pose_tracker import LandmarkId
    from app.targets.placement import (
        PointSmoother,
        body_offset,
        display_point,
        offset_for,
        project_target,
        radius_for,
    )
    from app.targets.review import ReviewStore
    from app.targets.training import TargetTraining
    from app.trainer.factory import create_classifier

    config.validate()
    if args.review_session:
        return _review_saved_targets(
            config,
            args.review_session,
            review_verified=args.review_verified,
            review_flagged=args.review_flagged,
        )
    if args.only_class is not None and args.training_order != "blocked":
        print("--only-class solo funciona con --training-order blocked.")
        return 2
    if not (
        config.min_sample_duration_ms
        <= args.active_ms
        <= config.max_sample_duration_ms
    ):
        print("La ventana de Target Training debe durar entre 1.5 y 2.8 segundos.")
        return 2
    sensitivity = args.sensitivity or "high"
    classifier = create_classifier("heuristic", sensitivity=sensitivity)
    review = ReviewStore(config.data_dir)
    session = TargetTraining(
        review=review,
        config=config,
        goal=args.target_goal,
        round_size=args.round_size,
        countdown_ms=config.countdown_ms,
        active_ms=args.active_ms,
        seed=args.seed,
        order=args.training_order,
        only_class=args.only_class,
    )
    left_smoother = PointSmoother()
    right_smoother = PointSmoother()
    fps_meter = FrameRateMeter(config.fps_window_size)
    camera = Camera(
        index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        probe_attempts=config.camera_probe_attempts,
    )
    print("Target Training. El detector heurístico solo propone eventos.")
    print("La etiqueta del objetivo no queda verificada hasta la revisión.")
    print("La puntuación del juego no es la precisión de NEAT.")
    print(f"Sesión {review.session_id}.")
    if args.training_order == "blocked":
        from app.dataset_manager import MOVEMENTS

        if args.only_class is None:
            print(f"Cinco bloques, {args.target_goal} intentos. La guardia es el ultimo. Total {args.target_goal * 5}.")
        else:
            print(f"Solo {MOVEMENTS[args.only_class].display_name}: {args.target_goal} intentos.")
        print("Entre cada intento espera. La pantalla dice la tecnica que sigue.")
        print("ESPACIO continua. Durante el golpe, ESPACIO pausa. S salta el intento.")
        print("Q no sale a la primera si faltan bloques. Al terminar, P repite lo incompleto.")
        print("0-4 elige la etiqueta y G guarda solo esa captura. La clase pedida no se verifica sola.")
    else:
        print(f"Meta: {args.target_goal} verificadas por clase.")
        print("ESPACIO inicia o pausa, H abre sesiones guardadas, R reinicia, V revisa, Q sale.")
    print("En el modo mezclado, un intento no entra a NEAT hasta que elijas 0-4 y pulses G.")
    try:
        with PoseTracker(
            model_path=config.model_path,
            min_detection_confidence=config.min_pose_detection_confidence,
            min_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as tracker, camera:
            cv2.namedWindow("EvoBox AI — Objetivos", cv2.WINDOW_NORMAL)

            def _on_click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    session.handle_click(x, y, session.width, session.height)

            cv2.setMouseCallback("EvoBox AI — Objetivos", _on_click)
            while True:
                packet = camera.read()
                pose = tracker.detect(packet.frame, packet.timestamp_ms)
                features = extract_biomechanical_features(
                    pose,
                    min_visibility=config.min_landmark_visibility,
                    min_presence=config.min_landmark_presence,
                    image_size=(packet.width, packet.height),
                )
                result = classifier.update(
                    features,
                    pose_detected=pose.detected,
                    pose=pose,
                    frame=packet.frame,
                )
                left = pose.landmark(LandmarkId.LEFT_SHOULDER)
                right = pose.landmark(LandmarkId.RIGHT_SHOULDER)
                tracking = bool(
                    pose.detected
                    and left is not None
                    and right is not None
                    and left.is_finite
                    and right.is_finite
                )
                screen = None
                radius = 0
                wrist_offset = None
                if tracking and left is not None and right is not None:
                    left_point = left_smoother.update(
                        display_point(
                            left.x, left.y, packet.width, packet.height, mirror=config.mirror_view
                        )
                    )
                    right_point = right_smoother.update(
                        display_point(
                            right.x, right.y, packet.width, packet.height, mirror=config.mirror_view
                        )
                    )
                    label = session.current_label
                    if label not in (None, 0):
                        projected = project_target(
                            left_point,
                            right_point,
                            offset_for(label),
                            radius_for(label),
                            packet.width,
                            packet.height,
                        )
                        if projected is not None:
                            screen = (projected[0], projected[1])
                            radius = projected[2]
                        wrist_id = LandmarkId.LEFT_WRIST if label in (1, 3) else LandmarkId.RIGHT_WRIST
                        wrist = pose.landmark(wrist_id)
                        if wrist is not None and wrist.is_finite:
                            wrist_offset = body_offset(
                                display_point(
                                    wrist.x,
                                    wrist.y,
                                    packet.width,
                                    packet.height,
                                    mirror=config.mirror_view,
                                ),
                                left_point,
                                right_point,
                            )
                else:
                    left_smoother.reset()
                    right_smoother.reset()
                event = result.events[0] if result.events else None
                session.width = packet.width
                session.height = packet.height
                session.tick(
                    packet.timestamp_ms,
                    tracking=tracking,
                    event_present=event is not None,
                    event_label=None if event is None else event.label_id,
                    event_key=None if event is None else f"{packet.timestamp_ms}:{event.label_id}:{event.arm}",
                    event_name="" if event is None else event.display_name,
                    candidate=result.candidate,
                    wrist=wrist_offset,
                    frame=build_frame_record(config, packet.timestamp_ms, pose, features),
                )
                display = render_frame(
                    packet.frame,
                    pose,
                    features,
                    fps_meter.tick(),
                    config,
                    targets=session.view(screen, radius),
                )
                session.advance_playback(33)
                cv2.imshow("EvoBox AI — Objetivos", display)
                key = cv2.waitKeyEx(1)
                if key >= 0 and session.handle_key(key, packet.timestamp_ms):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    counts = review.counts()
    verified = review.verified_counts()
    print(f"Sesión {review.session_id}")
    print(
        "Revisión: "
        f"provisionales {counts['provisional']}, "
        f"dudosas {counts['needs_review']}, "
        f"inválidas {counts['invalid']}, "
        f"verificadas {counts['verified']}."
    )
    print("Verificadas por clase: " + ", ".join(f"{label}={count}" for label, count in verified.items()))
    from app.targets.review import dataset_gap_lines

    for line in dataset_gap_lines(config.data_dir, args.target_goal):
        print(line)
    print("NEAT solo entrena con lo que hayas guardado desde la revisión.")
    return 0


def _review_saved_targets(
    config: AppConfig,
    session_id: str,
    review_verified: bool = False,
    review_flagged: bool = False,
) -> int:
    """Revisión de una sesión previa, sin cámara y sin borrar capturas."""

    import numpy as np

    from app.geometry import BiomechanicalFeatures
    from app.pose_tracker import PoseObservation
    from app.targets.review import dataset_gap_lines, list_review_sessions, open_review_session
    from app.targets.training import TargetTraining

    if session_id == "list":
        sessions = list_review_sessions(config.data_dir)
        if not sessions:
            print("No hay sesiones con capturas en data/review/.")
            return 2
        for item in sessions:
            print(
                f"{item['session_id']} | capturas {item['captures']} | "
                f"provisionales {item['provisional']} | dudosas {item['needs_review']} | "
                f"verificadas {item['verified']}"
            )
        print("Ábrelas con --review-session y el identificador. No incluyen video.")
        return 0
    try:
        review = open_review_session(config.data_dir, session_id)
    except FileNotFoundError as error:
        print(error)
        print("Sesiones disponibles:")
        for item in list_review_sessions(config.data_dir):
            print(item["session_id"])
        return 2
    review_ids: frozenset[str] = frozenset()
    if review_flagged:
        import json

        note = config.data_dir / "review_notes" / "bfda2f15_block_alignment.json"
        if not note.is_file():
            print(f"No está la lista de revisión en {note}.")
            return 2
        payload = json.loads(note.read_text(encoding="utf-8"))
        review_ids = frozenset(str(item) for item in payload["sample_ids"])
        print(f"{len(review_ids)} muestras alineadas al bloque, sin verificación visual.")
        print("Mira el esqueleto. 0-4 y G corrigen. G solo no cambia nada.")
    session = TargetTraining(
        review=review,
        config=config,
        goal=10,
        review_verified=review_verified,
        review_ids=review_ids,
    )
    session.phase = "review"
    session._load_player()
    disagreements = sum(
        1
        for _path, payload in review.entries()
        if payload.get("verification_status") == "verified"
        and payload.get("approved_label_id") != payload.get("requested_label_id")
    )
    session.message = (
        f"{disagreements} guardadas no coinciden con lo pedido. Elige 0-4 y pulsa G. Nada cambia solo."
        if review_verified
        else "Sesión recuperada. Elige 0-4 y pulsa G. Nada se aprueba solo."
    )
    print(f"Revisando {session_id} sin cámara.")
    print("No hay video de los golpes. Puedes ver duración, ángulos y la predicción original.")
    print("0-4 elige la etiqueta, G guarda, X descarta, ESPACIO reproduce, coma y punto avanzan un fotograma.")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    pose = PoseObservation(timestamp_ms=0)
    features = BiomechanicalFeatures(0, None, None, None, None)
    cv2.namedWindow("EvoBox AI — Revisión", cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = render_frame(blank, pose, features, None, config, targets=session.view(None, 0))
            session.advance_playback(30)
            cv2.imshow("EvoBox AI — Revisión", frame)
            key = cv2.waitKeyEx(30)
            if key < 0:
                continue
            if session.handle_key(key, 0):
                break
    finally:
        cv2.destroyAllWindows()
    for line in dataset_gap_lines(config.data_dir, 10):
        print(line)
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "train":
        from app.evolution.experiment import run_training

        try:
            return run_training(args)
        except (DatasetError, ValueError, RuntimeError) as error:
            print(f"Error de EvoBox AI: {error}", file=sys.stderr)
            return 1
    if args.mode == "dataset-report":
        from app.evolution.academic import build_academic_package

        try:
            return build_academic_package(args)
        except (DatasetError, ValueError, RuntimeError) as error:
            print(f"Error de EvoBox AI: {error}", file=sys.stderr)
            return 1

    config = AppConfig(
        camera_index=args.camera_index,
        frame_width=args.width,
        frame_height=args.height,
        mirror_view=not args.no_mirror,
        model_path=args.model.expanduser().resolve(),
        mode=args.mode,
        data_dir=args.data_dir.expanduser().resolve(),
    )

    try:
        if config.mode == "live":
            return run_live(config, args)
        if config.mode == "benchmark":
            return run_benchmark(config, args)
        if config.mode == "targets":
            return run_targets(config, args)
        if config.mode == "game":
            from app.game.play import run_game

            return run_game(config, args)
        if config.mode == "collect":
            run_collection(config)
        else:
            run(config)
    except KeyboardInterrupt:
        print("\nEvoBox AI cerrado por el usuario.")
        return 0
    except ClassifierUnavailable as error:
        print(f"Clasificador no disponible: {error}", file=sys.stderr)
        return 2
    except (
        CameraError,
        DatasetError,
        FileNotFoundError,
        ValueError,
        RuntimeError,
        cv2.error,
    ) as error:
        print(f"Error de EvoBox AI: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
