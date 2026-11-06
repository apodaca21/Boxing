"""Identidad de webcams en macOS y asignación frontal/profundidad."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from app.camera import probe_cameras


FRONT_NAME_HINTS = ("macbook", "facetime", "built-in", "integrated", "isight")
DEPTH_NAME_HINTS = ("steren", "usb", "com-", "external", "hd webcam", "logitech", "c9")
AVOID_NAME_HINTS = ("iphone", "continuity", "ipad", "desk view")


@dataclass(frozen=True, slots=True)
class NamedCamera:
    index: int
    name: str
    available: bool
    mean_brightness: float | None = None
    width: int | None = None
    height: int | None = None
    backend: str | None = None


@dataclass(frozen=True, slots=True)
class DepthPair:
    front_index: int
    depth_index: int
    front_name: str
    depth_name: str
    notes: tuple[str, ...] = ()


def list_avfoundation_names() -> tuple[str, ...]:
    """Nombres en el orden histórico de AVFoundation."""

    if platform.system() != "Darwin":
        return ()
    script = (
        "import AVFoundation\n"
        "import Foundation\n"
        "let devices = AVCaptureDevice.devices(for: .video)\n"
        "for d in devices {\n"
        "  print(d.localizedName)\n"
        "}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as handle:
        handle.write(script)
        path = handle.name
    try:
        completed = subprocess.run(
            ["swift", path],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    finally:
        Path(path).unlink(missing_ok=True)
    if completed.returncode != 0:
        return ()
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _capture_named_thumbnails(directory: Path) -> dict[str, np.ndarray]:
    """Captura un JPEG por dispositivo AVFoundation para emparejar con OpenCV."""

    if platform.system() != "Darwin":
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    script = f"""
import AVFoundation
import AppKit
import Foundation

final class Grab: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {{
  var done = false
  var image: NSImage?
  let lock = NSLock()
  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {{
    lock.lock(); defer {{ lock.unlock() }}
    if done {{ return }}
    guard let buf = CMSampleBufferGetImageBuffer(sampleBuffer) else {{ return }}
    let ci = CIImage(cvImageBuffer: buf)
    let rep = NSCIImageRep(ciImage: ci)
    let ns = NSImage(size: rep.size); ns.addRepresentation(rep)
    image = ns
    done = true
  }}
}}

func save(_ device: AVCaptureDevice, to path: String) -> Bool {{
  let session = AVCaptureSession()
  session.sessionPreset = .vga640x480
  guard let input = try? AVCaptureDeviceInput(device: device) else {{ return false }}
  let output = AVCaptureVideoDataOutput()
  output.alwaysDiscardsLateVideoFrames = true
  let grab = Grab()
  let q = DispatchQueue(label: "grab")
  output.setSampleBufferDelegate(grab, queue: q)
  session.beginConfiguration()
  guard session.canAddInput(input), session.canAddOutput(output) else {{ return false }}
  session.addInput(input)
  session.addOutput(output)
  session.commitConfiguration()
  session.startRunning()
  let start = Date()
  while Date().timeIntervalSince(start) < 1.8 {{
    grab.lock.lock(); let ready = grab.done; let img = grab.image; grab.lock.unlock()
    if ready, let img = img, let tiff = img.tiffRepresentation,
       let rep = NSBitmapImageRep(data: tiff),
       let data = rep.representation(using: .jpeg, properties: [:]) {{
      try? data.write(to: URL(fileURLWithPath: path))
      session.stopRunning()
      return true
    }}
    Thread.sleep(forTimeInterval: 0.05)
  }}
  session.stopRunning()
  return false
}}

let devices = AVCaptureDevice.devices(for: .video)
for (i, d) in devices.enumerated() {{
  let safe = d.localizedName.replacingOccurrences(of: "/", with: "_")
  let path = "{directory.as_posix()}/named_\\(i).jpg"
  let ok = save(d, to: path)
  print("\\(d.localizedName)|\\(ok)|\\(path)")
}}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as handle:
        handle.write(script)
        path = handle.name
    try:
        completed = subprocess.run(
            ["swift", path],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    finally:
        Path(path).unlink(missing_ok=True)
    if completed.returncode != 0:
        return {}
    thumbs: dict[str, np.ndarray] = {}
    for line in completed.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3 or parts[1] != "true":
            continue
        name, _ok, image_path = parts
        image = cv2.imread(image_path)
        if image is None:
            continue
        thumbs[name] = cv2.resize(image, (160, 120)).astype(np.float32)
    return thumbs


def _opencv_thumbnail(index: int, width: int, height: int, attempts: int) -> tuple[np.ndarray | None, float | None, tuple[int, int] | None, str | None]:
    backends = []
    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append((cv2.CAP_AVFOUNDATION, "AVFoundation"))
    backends.append((cv2.CAP_ANY, "backend automático"))
    for backend, label in backends:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        best = None
        # Algunas cámaras (MacBook/Continuity) entregan 1-2 fotogramas negros al abrir.
        for _ in range(max(6, attempts + 3)):
            ok, candidate = capture.read()
            if ok and isinstance(candidate, np.ndarray) and candidate.size > 0:
                brightness = float(np.mean(candidate))
                if best is None or brightness > best[0]:
                    best = (brightness, candidate)
                if brightness >= 8.0 and best[0] >= 8.0:
                    # Conserva el más brillante visto tras estabilizar un poco.
                    pass
            time.sleep(0.03)
        capture.release()
        if best is None:
            continue
        brightness, frame = best
        thumb = cv2.resize(frame, (160, 120)).astype(np.float32)
        return thumb, brightness, (int(frame.shape[1]), int(frame.shape[0])), label
    return None, None, None, None


def _match_name(thumb: np.ndarray, named: dict[str, np.ndarray], used: set[str]) -> str | None:
    best_name = None
    best_error = None
    for name, reference in named.items():
        if name in used:
            continue
        error = float(np.mean((thumb - reference) ** 2))
        if best_error is None or error < best_error:
            best_error = error
            best_name = name
    if best_name is None or best_error is None or best_error > 8000:
        return None
    used.add(best_name)
    return best_name


def inventory(
    *,
    max_index: int = 8,
    width: int = 640,
    height: int = 480,
    probe_attempts: int = 3,
) -> tuple[NamedCamera, ...]:
    """Empareja índices OpenCV con nombres reales cuando es posible."""

    names = list_avfoundation_names()
    named_thumbs: dict[str, np.ndarray] = {}
    if names:
        with tempfile.TemporaryDirectory(prefix="evobox-cams-") as temporary:
            named_thumbs = _capture_named_thumbnails(Path(temporary))
    probed = {
        info.index: info
        for info in probe_cameras(
            max_index=max_index,
            width=width,
            height=height,
            probe_attempts=probe_attempts,
        )
    }
    used_names: set[str] = set()
    rows: list[NamedCamera] = []
    for index in range(max_index + 1):
        info = probed.get(index)
        thumb, brightness, size, backend = _opencv_thumbnail(
            index, width, height, probe_attempts
        )
        available = bool(
            info
            and info.available
            and brightness is not None
            and brightness >= 8.0
            and thumb is not None
        )
        matched = None
        if thumb is not None and named_thumbs:
            matched = _match_name(thumb, named_thumbs, used_names)
        fallback = names[index] if index < len(names) else f"camara_{index}"
        name = matched or (fallback if matched is None and index < len(names) and fallback not in used_names else f"opencv_{index}")
        if matched is None and name.startswith("opencv_"):
            pass
        elif matched is None and name in used_names:
            name = f"opencv_{index}"
        rows.append(
            NamedCamera(
                index=index,
                name=name,
                available=available,
                mean_brightness=brightness,
                width=None if size is None else size[0],
                height=None if size is None else size[1],
                backend=backend or (None if info is None else info.backend),
            )
        )
    return tuple(rows)


def _score_name(name: str, hints: tuple[str, ...]) -> int:
    lowered = name.casefold()
    return sum(1 for hint in hints if hint in lowered)


def suggest_depth_pair(
    cameras: tuple[NamedCamera, ...] | None = None,
    *,
    preferred_front: int | None = None,
    preferred_depth: int | None = None,
) -> DepthPair | None:
    """Elige laptop como frontal y USB/Steren como profundidad cuando sea posible."""

    rows = cameras if cameras is not None else inventory()
    usable = [row for row in rows if row.available]
    if len(usable) < 2:
        return None

    def front_key(row: NamedCamera) -> tuple[int, int, float]:
        avoid = _score_name(row.name, AVOID_NAME_HINTS)
        prefer = _score_name(row.name, FRONT_NAME_HINTS)
        brightness = row.mean_brightness or 0.0
        return (prefer, -avoid, brightness)

    def depth_key(row: NamedCamera) -> tuple[int, int, float]:
        avoid = _score_name(row.name, AVOID_NAME_HINTS)
        prefer = _score_name(row.name, DEPTH_NAME_HINTS)
        brightness = row.mean_brightness or 0.0
        return (prefer, -avoid, brightness)

    notes: list[str] = []
    if preferred_front is not None and preferred_depth is not None:
        by_index = {row.index: row for row in usable}
        if preferred_front in by_index and preferred_depth in by_index and preferred_front != preferred_depth:
            front = by_index[preferred_front]
            depth = by_index[preferred_depth]
            return DepthPair(front.index, depth.index, front.name, depth.name, notes=tuple(notes))

    ranked_front = sorted(usable, key=front_key, reverse=True)
    front = ranked_front[0]
    depth_candidates = [row for row in usable if row.index != front.index]
    depth = sorted(depth_candidates, key=depth_key, reverse=True)[0]

    if _score_name(front.name, AVOID_NAME_HINTS):
        notes.append(
            f"La frontal candidata '{front.name}' parece Continuity/iPhone; "
            "abre la tapa de la MacBook o pulsa F para intercambiar."
        )
    if _score_name(front.name, FRONT_NAME_HINTS) == 0:
        notes.append(
            f"Frontal en '{front.name}'. Si no es la laptop, pulsa F para intercambiar."
        )
    if _score_name(depth.name, DEPTH_NAME_HINTS) == 0:
        notes.append(
            f"Profundidad en '{depth.name}'. Colócala al lado para ver el brazo."
        )
    return DepthPair(front.index, depth.index, front.name, depth.name, notes=tuple(notes))
