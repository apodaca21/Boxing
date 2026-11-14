"""Sesión de entrenamiento libre: tiempo activo, conteos y ritmo."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.dataset_manager import MOVEMENTS
from app.trainer.counting import PunchLedger, event_identity, rows_for
from app.trainer.results import ClassificationResult


HIGHLIGHT_MS = 700


@dataclass(frozen=True, slots=True)
class LiveView:
    """Estado visual del entrenador. No incluye porcentajes de confianza."""

    running: bool
    recognition: str
    last_punch: str
    counts: dict[int, int]
    total: int
    punches_per_minute: float | None
    elapsed_ms: int
    source: str
    provisional: bool
    highlight_arm: str | None
    instructions: str
    rows: tuple[tuple[str, int, str], ...]
    candidate: str = "—"
    sensitivity: str = "NORMAL"
    latency_ms: float | None = None
    moving: bool = False
    debug: bool = False
    debug_lines: tuple[str, ...] = ()
    training_style: str = "libre"
    arm_only: bool = False
    rejection: str | None = None
    camera_line: str = ""
    depth_status: str = ""
    depth_ready: bool = False
    awaiting_setup: bool = False
    setup_ready: bool = False
    setup_status: str = ""
    front_ready: bool = False
    left_arm_ok: bool = False
    right_arm_ok: bool = False
    arm_hint: str = ""


class TrainingSession:
    """El contador avanza solo con eventos completos y la sesión en marcha."""

    def __init__(
        self,
        *,
        require_camera_setup: bool = False,
        arm_only: bool = False,
    ) -> None:
        self.require_camera_setup = require_camera_setup
        self.arm_only = arm_only
        self.awaiting_setup = require_camera_setup
        self.setup_ready = False
        self.setup_status = (
            "AJUSTA CAMARA PARA MEDIR ANGULOS"
            if require_camera_setup
            else ""
        )
        self.front_ready = False
        self.running = not require_camera_setup
        self._ledger = PunchLedger()
        self.counts = self._ledger.counts
        self.named_counts = self._ledger.named_counts
        self.punch_total = 0
        self.last_punch = "—"
        self.recognition = MOVEMENTS[0].display_name
        self.source = "heuristic"
        self.provisional = True
        self.elapsed_ms = 0
        self._previous_timestamp: int | None = None
        self._highlight_arm: str | None = None
        self._highlight_until_ms = -1
        self.candidate = "—"
        self.sensitivity = "NORMAL"
        self.latency_ms: float | None = None
        self.moving = False
        self.debug = False
        self.debug_lines: tuple[str, ...] = ()
        self.training_style = "libre"
        self.rejection: str | None = None
        self.camera_line = ""
        self.depth_status = ""
        self.depth_ready = False
        self.left_arm_ok = False
        self.right_arm_ok = False
        self.arm_hint = ""

    def reset(self) -> None:
        self._ledger.reset()
        self.counts = self._ledger.counts
        self.named_counts = self._ledger.named_counts
        self.punch_total = 0
        self.last_punch = "—"
        self.recognition = MOVEMENTS[0].display_name
        self.elapsed_ms = 0
        self._previous_timestamp = None
        self._highlight_arm = None
        self._highlight_until_ms = -1
        self.candidate = "—"
        self.moving = False
        self.rejection = None
        self.debug_lines = ()
        if self.require_camera_setup:
            self.awaiting_setup = True
            self.setup_ready = False
            self.running = False
            self.setup_status = "AJUSTA CAMARA PARA MEDIR ANGULOS"
        else:
            self.running = True
        # camera_line y depth_status se mantienen: describen el rig, no el conteo.
        # depth_ready también describe el rig y no cambia al reiniciar el contador.

    def update_camera_setup(
        self,
        *,
        front_ready: bool,
        front_status: str,
        depth_ready: bool,
        depth_required: bool,
        depth_status: str = "",
    ) -> None:
        """Actualiza el bloqueo de inicio según la calidad de las cámaras."""

        self.front_ready = front_ready
        self.depth_ready = depth_ready if depth_required else True
        if not front_ready:
            self.setup_ready = False
            self.setup_status = front_status or "AJUSTA CAMARA FRONTAL"
            return
        if depth_required and not depth_ready:
            self.setup_ready = False
            self.setup_status = depth_status or "AJUSTA CAMARA DEPTH"
            return
        self.setup_ready = True
        self.setup_status = "LISTO  ENTER PARA EMPEZAR"

    def update_arm_tracking(
        self,
        *,
        left_arm_ok: bool,
        right_arm_ok: bool,
        too_close: bool = False,
    ) -> None:
        """Guía en vivo para no perder izquierda/derecha a mitad de sesión."""

        self.left_arm_ok = left_arm_ok
        self.right_arm_ok = right_arm_ok
        if too_close:
            self.arm_hint = "ALEJATE: pecho y brazos en cuadro"
        elif not left_arm_ok and not right_arm_ok:
            self.arm_hint = "MUESTRA AMBOS BRAZOS"
        elif not left_arm_ok:
            self.arm_hint = "BRAZO IZQUIERDO FUERA DE CUADRO"
        elif not right_arm_ok:
            self.arm_hint = "BRAZO DERECHO FUERA DE CUADRO"
        else:
            self.arm_hint = ""

    def try_start(self) -> bool:
        """Libera el bloqueo solo si las cámaras ya permiten medir ángulos."""

        if not self.awaiting_setup:
            return False
        if not self.setup_ready:
            return False
        self.awaiting_setup = False
        self.running = True
        self._previous_timestamp = None
        self.setup_status = "CAMARAS LISTAS"
        return True

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            return True
        if self.awaiting_setup:
            if key in (13, ord(" "), ord("p"), ord("P")):
                self.try_start()
            elif key in (ord("r"), ord("R")):
                self.reset()
            return False
        if key in (ord("p"), ord("P"), ord(" ")):
            self.running = not self.running
            self._previous_timestamp = None
        elif key in (ord("r"), ord("R")):
            self.reset()
        return False

    def advance(self, timestamp_ms: int, result: ClassificationResult) -> None:
        if self.awaiting_setup:
            self.source = result.source
            self.provisional = result.source == "heuristic"
            self.candidate = self.setup_status or "AJUSTA CAMARA"
            self.moving = False
            self.rejection = "camara no lista"
            self.latency_ms = result.latency_ms
            if result.sensitivity:
                self.sensitivity = result.sensitivity.upper()
            self.debug_lines = result.debug_lines
            self.recognition = self.setup_status or "AJUSTA CAMARA"
            return
        if self.running and self._previous_timestamp is not None:
            delta = timestamp_ms - self._previous_timestamp
            if 0 < delta <= 5000:
                self.elapsed_ms += delta
        self._previous_timestamp = timestamp_ms
        self.source = result.source
        self.provisional = result.source == "heuristic"
        self.candidate = result.candidate or "—"
        self.moving = result.moving
        self.rejection = result.rejection
        self.latency_ms = result.latency_ms
        if result.sensitivity:
            self.sensitivity = result.sensitivity.upper()
        self.debug_lines = result.debug_lines
        if result.events or not result.moving:
            self.recognition = result.display_name
        if not self.running:
            return
        if not result.events:
            self._ledger.release()
        for event in result.events:
            if event.event_id is None:
                key, arm = event_identity(event)
                event = replace(event, event_id=f"session:{timestamp_ms}:{arm}:{key}")
            if not self._ledger.commit(event):
                continue
            self.counts = self._ledger.counts
            self.named_counts = self._ledger.named_counts
            self.punch_total = self._ledger.total
            self.last_punch = self._ledger.last_punch
            self._highlight_arm = event.arm
            self._highlight_until_ms = timestamp_ms + HIGHLIGHT_MS

    def view(self, timestamp_ms: int) -> LiveView:
        total = self.punch_total
        pace = None
        if self.elapsed_ms >= 1000 and total >= 0:
            pace = total * 60_000 / self.elapsed_ms
        highlight = (
            self._highlight_arm
            if timestamp_ms <= self._highlight_until_ms
            else None
        )
        if self.awaiting_setup:
            state = "AJUSTA CAMARA" if not self.setup_ready else "LISTO PARA EMPEZAR"
            instructions = (
                "ENTER o ESPACIO empieza cuando las camaras midan bien los angulos"
                if self.setup_ready
                else "Centrate, muestra brazos y espera FRONT/DEPTH LISTA"
            )
        else:
            state = "EN MARCHA" if self.running else "EN PAUSA"
            instructions = f"{state}   P pausa   R reinicia   Q sale"
        return LiveView(
            running=self.running,
            recognition=self.recognition,
            last_punch=self.last_punch,
            counts=dict(self.counts),
            total=total,
            punches_per_minute=pace,
            elapsed_ms=self.elapsed_ms,
            source=self.source,
            provisional=self.provisional,
            highlight_arm=highlight,
            instructions=instructions,
            rows=self._rows(),
            candidate=self.candidate,
            sensitivity=self.sensitivity,
            latency_ms=self.latency_ms,
            moving=self.moving,
            debug=self.debug,
            debug_lines=self.debug_lines,
            training_style=self.training_style,
            arm_only=self.arm_only,
            rejection=self.rejection,
            camera_line=self.camera_line,
            depth_status=self.depth_status,
            depth_ready=self.depth_ready,
            awaiting_setup=self.awaiting_setup,
            setup_ready=self.setup_ready,
            setup_status=self.setup_status,
            front_ready=self.front_ready,
            left_arm_ok=self.left_arm_ok,
            right_arm_ok=self.right_arm_ok,
            arm_hint=self.arm_hint,
        )

    def _rows(self) -> tuple[tuple[str, int, str], ...]:
        return rows_for(
            self.source,
            self.counts,
            self.named_counts,
            arm_only=self.arm_only,
        )
