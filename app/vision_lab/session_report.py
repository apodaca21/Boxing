"""Vista final de análisis sin reabrir ni almacenar video crudo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.vision_lab.session_recorder import SessionReport


ANALYSIS_WINDOW = "EvoBox AI — Análisis de sesión"


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    *,
    bold: bool = False,
) -> None:
    """Dibuja UTF-8 real; OpenCV Hershey no soporta acentos."""

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    draw.text(origin, text, fill=(color[2], color[1], color[0]), font=_font(size, bold=bold))
    image[:] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _placeholder(title: str, message: str, *, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), (19, 22, 29), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (82, 91, 105), 1)
    _put_text(image, title, (16, 22), 20, (245, 247, 250), bold=True)
    _put_text(image, message, (16, height // 2), 18, (180, 190, 205))
    return image


def _read_map(
    path: Path | None,
    title: str,
    *,
    width: int = 360,
    height: int = 270,
) -> np.ndarray:
    if path is None or not path.is_file():
        return _placeholder(title, "Datos insuficientes", width=width, height=height)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return _placeholder(title, "No se pudo leer el PNG", width=width, height=height)
    panel = _fit(image, width, height)
    cv2.rectangle(panel, (0, 0), (width - 1, 30), (12, 14, 20), -1)
    _put_text(panel, title, (10, 5), 16, (245, 247, 250))
    return panel


def show_session_analysis(report: SessionReport) -> None:
    """Muestra Resumen, Trayectorias/Biomecánica, Mapas y Exportar con teclas."""

    print("Análisis: 1 resumen | 2 trayectorias/biomecánica | 3 mapas | 4 exportar | Q cerrar")
    summary = report.summary
    maps = {
        "Actividad general (Steren)": Path(report.exports["activity"]) if "activity" in report.exports else None,
        "Mano izquierda (MacBook)": Path(report.exports["left"]) if "left" in report.exports else None,
        "Mano derecha (MacBook)": Path(report.exports["right"]) if "right" in report.exports else None,
    }
    tab = "summary"
    try:
        cv2.namedWindow(ANALYSIS_WINDOW, cv2.WINDOW_NORMAL)
        while True:
            if tab == "maps":
                image = _maps_view(maps)
            elif tab == "biomechanics":
                image = _trajectory_view(report)
            else:
                image = np.zeros((620, 980, 3), dtype=np.uint8)
                _text_view(image, report, tab)
            cv2.imshow(ANALYSIS_WINDOW, image)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), ord("Q"), 27, 13, 10):
                break
            if key == ord("1"):
                tab = "summary"
            elif key == ord("2"):
                tab = "biomechanics"
            elif key == ord("3"):
                tab = "maps"
            elif key == ord("4"):
                tab = "export"
    except cv2.error:
        # La persistencia sigue siendo válida aunque el proceso se ejecute sin
        # un display (CI, SSH o una sesión sin permiso de ventana).
        return
    finally:
        try:
            cv2.destroyWindow(ANALYSIS_WINDOW)
        except cv2.error:
            pass


def _maps_view(maps: dict[str, Path | None]) -> np.ndarray:
    width, height = 360, 270
    panels = [
        _read_map(path, title, width=width, height=height)
        for title, path in maps.items()
    ]
    footer = 56
    canvas = np.full((height + footer, width * len(panels), 3), (10, 12, 17), dtype=np.uint8)
    for index, panel in enumerate(panels):
        canvas[:height, index * width : (index + 1) * width] = panel
    _put_text(canvas, "3 Mapas de calor  ·  1 Resumen  ·  2 Biomecánica  ·  4 Exportar  ·  Q cerrar", (18, height + 16), 18, (235, 240, 245))
    _put_text(canvas, "Escala compartida entre manos; Steren sólo si hubo señal sincronizada", (18, height + 36), 15, (170, 180, 195))
    return canvas


def _trajectory_view(report: SessionReport) -> np.ndarray:
    path = report.directory / "exports" / "trajectory_arms.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((620, 980, 3), dtype=np.uint8)
    else:
        image = _fit(image, 760, 500)
        canvas = np.zeros((620, 980, 3), dtype=np.uint8)
        canvas[40:540, 110:870] = image
        image = canvas
    _put_text(image, "2 Trayectorias y biomecánica  ·  1 Resumen  ·  3 Mapas  ·  4 Exportar  ·  Q cerrar", (18, 600), 18, (235, 240, 245))
    return image


def _text_view(image: np.ndarray, report: SessionReport, tab: str) -> None:
    summary = report.summary
    title = {
        "summary": "RESUMEN DE SESIÓN",
        "biomechanics": "TRAYECTORIAS Y BIOMECÁNICA",
        "export": "EXPORTAR / ARCHIVOS",
    }.get(tab, "ANÁLISIS")
    _put_text(image, title, (30, 30), 32, (245, 247, 250), bold=True)
    lines: list[str]
    if tab == "summary":
        lines = [
            f"Sesión: {report.session_id}",
            f"Duración activa: {summary.get('duration_active_s', 0):.2f} s",
            f"Golpes válidos: {summary.get('total_punches', 0)}",
            f"Por mano: {summary.get('by_hand', {})}",
            f"Por tipo: {summary.get('by_type', {})}",
            f"Tracking: {summary.get('tracking_quality', 0):.0%} | muñecas: {summary.get('wrist_tracking_quality', 0):.0%}",
            f"Actividad Steren: {summary.get('movement_time_s', 0):.2f} s",
            "3D métrico: no disponible (Steren RGB)",
        ]
    elif tab == "biomechanics":
        lines = [
            "Las métricas están expresadas en px/s, px/s², grados y anchos de hombro.",
            "Las pérdidas de tracking se marcan; no se interpolan como observaciones reales.",
            "Los detalles por golpe están en biomechanics.json y punch_events.csv.",
            "Mapas: 3 | Resumen: 1 | Exportar: 4 | Cerrar: Q",
        ]
    else:
        lines = [
            f"Directorio: {report.directory}",
            "Archivos exportados:",
            *[f"- {path}" for path in report.exports.values()],
            "3 muestra los mapas; Q cierra el análisis.",
        ]
    for index, line in enumerate(lines):
        _put_text(image, str(line)[:120], (34, 90 + index * 34), 20, (220, 228, 236))
    _put_text(image, "1 Resumen   2 Biomecánica   3 Mapas   4 Exportar   Q cerrar", (30, image.shape[0] - 34), 18, (120, 210, 240))
