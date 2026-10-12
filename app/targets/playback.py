"""Reproducción 2D con las articulaciones que sí se guardaron."""

from __future__ import annotations

from dataclasses import dataclass


STORED_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)


@dataclass(frozen=True, slots=True)
class SkeletonSnapshot:
    points: tuple[tuple[str, float, float], ...]
    connections: tuple[tuple[int, int], ...]
    index: int
    count: int
    playing: bool
    elapsed_ms: int
    angle_lines: tuple[str, ...]
    notice: str


class SkeletonPlayer:
    """Avanza con los timestamps guardados. No inventa articulaciones."""

    def __init__(self, frames: list[object], *, mirror: bool = False) -> None:
        self.frames = frames
        self.mirror = mirror
        self.index = 0
        self.playing = False
        self.error = ""
        self._carry_ms = 0

    def toggle(self) -> None:
        if len(self.frames) < 2 or self.error:
            self.playing = False
            return
        self.playing = not self.playing

    def step(self, delta: int) -> None:
        self.playing = False
        self._carry_ms = 0
        if not self.frames:
            return
        self.index = min(max(self.index + delta, 0), len(self.frames) - 1)

    def advance(self, dt_ms: int) -> None:
        if not self.playing or dt_ms <= 0 or len(self.frames) < 2:
            return
        self._carry_ms += dt_ms
        while self.playing and self.index < len(self.frames) - 1:
            current = _timestamp(self.frames[self.index])
            nxt = _timestamp(self.frames[self.index + 1])
            if current is None or nxt is None or nxt <= current:
                self.playing = False
                self.error = "Los timestamps guardados no permiten continuar la reproducción."
                return
            gap = nxt - current
            if self._carry_ms < gap:
                return
            self._carry_ms -= gap
            self.index += 1
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self._carry_ms = 0

    def snapshot(self) -> SkeletonSnapshot:
        frame = self.frames[self.index] if self.frames else None
        points = _points(frame, self.mirror)
        by_name = {name: index for index, (name, _x, _y) in enumerate(points)}
        connections = tuple(
            (by_name[start], by_name[end])
            for start, end in CONNECTIONS
            if start in by_name and end in by_name
        )
        elapsed = _elapsed(frame)
        notice = self.error or (
            "Solo se dibujan las articulaciones guardadas. No hay cabeza ni imagen de cámara."
        )
        return SkeletonSnapshot(
            points=tuple(points),
            connections=connections,
            index=self.index,
            count=len(self.frames),
            playing=self.playing,
            elapsed_ms=0 if elapsed is None else elapsed,
            angle_lines=_angles(frame),
            notice=notice,
        )


def _timestamp(frame: object) -> int | None:
    if not isinstance(frame, dict):
        return None
    value = frame.get("timestamp_ms")
    if isinstance(value, int):
        return value
    return None


def _elapsed(frame: object) -> int | None:
    if not isinstance(frame, dict):
        return None
    value = frame.get("elapsed_ms")
    if isinstance(value, int):
        return value
    return None


def _points(frame: object, mirror: bool) -> list[tuple[str, float, float]]:
    if not isinstance(frame, dict):
        return []
    raw = frame.get("landmarks")
    if not isinstance(raw, dict):
        return []
    points: list[tuple[str, float, float]] = []
    for name in STORED_JOINTS:
        record = raw.get(name)
        if not isinstance(record, dict):
            continue
        x = record.get("x")
        y = record.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        shown_x = 1.0 - float(x) if mirror else float(x)
        points.append((name, shown_x, float(y)))
    return points


def _angles(frame: object) -> tuple[str, ...]:
    if not isinstance(frame, dict):
        return ("Ángulos: sin fotograma.",)
    features = frame.get("features")
    if not isinstance(features, dict):
        return ("Ángulos: no se guardaron.",)
    return (
        f"Codo izquierdo: {_number(features.get('left_elbow_angle'))}",
        f"Codo derecho: {_number(features.get('right_elbow_angle'))}",
        f"Alcance izquierdo: {_number(features.get('left_shoulder_wrist_distance'))}",
        f"Alcance derecho: {_number(features.get('right_shoulder_wrist_distance'))}",
    )


def _number(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "sin dato"
    return f"{float(value):.0f}"
