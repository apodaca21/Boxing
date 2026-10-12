"""Ronda de objetivos. La puntuación del juego no es una métrica de NEAT."""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.config import AppConfig
from app.dataset_manager import MOVEMENTS, FrameRecord
from app.targets.generator import BLOCK_LABELS, BlockPlan, TargetGenerator
from app.targets.impact import judge_target
from app.targets.placement import offset_for, radius_for
from app.targets.playback import SkeletonPlayer
from app.targets.review import (
    CaptureDraft,
    ReviewStore,
    capture_details,
    dataset_gap_lines,
    list_review_sessions,
    open_review_session,
)
from app.targets.view import TargetView


COLORS = {
    0: (176, 184, 176),
    1: (50, 210, 255),
    2: (255, 105, 70),
    3: (70, 170, 255),
    4: (90, 90, 255),
}
SUCCESS = {"acierto", "guardia"}


@dataclass
class TargetTraining:
    """Controla tiempos, aciertos y la cola de revisión."""

    review: ReviewStore
    config: AppConfig
    goal: int = 10
    round_size: int = 8
    countdown_ms: int = 3000
    active_ms: int = 2000
    gap_ms: int = 450
    seed: int = 42
    order: str = "mixed"
    only_class: int | None = None
    review_verified: bool = False
    review_ids: frozenset[str] = frozenset()
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        if not (
            self.config.min_sample_duration_ms
            <= self.active_ms
            <= self.config.max_sample_duration_ms
        ):
            raise ValueError("La ventana del objetivo debe durar entre 1.5 y 2.8 segundos.")
        if self.order not in {"mixed", "blocked"}:
            raise ValueError("El orden debe ser 'mixed' o 'blocked'.")
        if self.only_class is not None and self.only_class not in MOVEMENTS:
            raise ValueError("La clase sola debe ser 0, 1, 2, 3 o 4.")
        self.generator = TargetGenerator(goal=self.goal, seed=self.seed)
        sequence = BLOCK_LABELS if self.only_class is None else (self.only_class,)
        self.scheduled_classes = sequence
        self.block = BlockPlan(self.goal, sequence=sequence)
        self.blocks_finished = False
        self._quit_armed_until = 0
        self.attempts = 0
        self.phase = "idle"
        self.paused = False
        self.tracking = False
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.resolved_in_round = 0
        self.current_label: int | None = None
        self.outcome = ""
        self.candidate = ""
        self.detector_label: int | None = None
        self.detector_name = ""
        self._elapsed = 0
        self._last_ms: int | None = None
        self._window_start: int | None = None
        self._frames: list[FrameRecord] = []
        self._consumed: str | None = None
        self._locked = False
        self._reset_armed_until = 0
        self.selected = 0
        self._label_choice: int | None = None
        self._wrist_distance: float | None = None
        self._history: list[dict[str, object]] = []
        self._history_index = 0
        self.player: SkeletonPlayer | None = None
        self.message = (
            f"{self.block.header()}. PROGRESO: 0/{self.goal}. ESPACIO para empezar."
            if self.order == "blocked"
            else "ESPACIO inicia. H abre sesiones guardadas."
        )

    def tick(
        self,
        now_ms: int,
        *,
        tracking: bool,
        event_present: bool,
        event_label: int | None,
        event_key: str | None,
        event_name: str,
        candidate: str | None,
        wrist: tuple[float, float] | None,
        frame: FrameRecord | None,
    ) -> None:
        self.tracking = tracking
        self.candidate = candidate or ""
        delta = 0
        if self._last_ms is not None and tracking and not self.paused and self.phase in {
            "countdown",
            "active",
            "tail",
            "gap",
            "transition",
        }:
            delta = max(0, now_ms - self._last_ms)
        self._last_ms = now_ms
        if frame is not None and tracking and not self.paused and self.phase in {"active", "tail"}:
            self._frames.append(frame)
        self._elapsed += delta
        if self.phase == "countdown" and self._elapsed >= self.countdown_ms:
            self._open_target(now_ms)
        elif self.phase == "active" and self.current_label is not None:
            judgement = judge_target(
                requested=self.current_label,
                event_present=event_present,
                event_label=event_label,
                event_key=event_key,
                consumed_key=self._consumed,
                wrist=wrist,
                target=offset_for(self.current_label),
                radius=radius_for(self.current_label),
                tracking=tracking,
                elapsed_ms=self._elapsed,
                limit_ms=self.active_ms,
            )
            if wrist is not None:
                target = offset_for(self.current_label)
                self._wrist_distance = math.hypot(wrist[0] - target[0], wrist[1] - target[1])
            if event_present and event_key is not None:
                self._consumed = event_key
                self.detector_label = event_label
                self.detector_name = event_name
            if judgement.resolved and not self._locked:
                self._lock(judgement.status, judgement.failure)
            if self._locked and self._elapsed >= self.active_ms:
                self._close_window()
        elif self.phase == "tail" and self._elapsed >= self.active_ms:
            self._close_window()
        elif self.phase == "gap" and self._elapsed >= self.gap_ms:
            self._open_target(now_ms)
        if not tracking and self.phase in {"countdown", "active", "tail", "gap"}:
            self.message = "Sin seguimiento. La ronda esta en pausa."

    def handle_key(self, key: int, now_ms: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            if self.order == "blocked" and not self.blocks_finished:
                if now_ms > self._quit_armed_until:
                    self._quit_armed_until = now_ms + 2500
                    missing = ", ".join(MOVEMENTS[label].display_name for label in self._remaining_labels())
                    self.message = f"Aun falta: {missing}. Q otra vez para salir. La guardia no se guarda sola."
                    return False
            return True
        if key in (ord("h"), ord("H")) and self.phase in {"idle", "review"}:
            self._open_history()
            return False
        if self.phase == "history":
            self._history_key(key)
            return False
        if self.phase == "summary":
            if key == ord(" "):
                self.phase = "review"
                self._label_choice = None
                self._load_player()
                self.message = "Elige 0-4 y pulsa G. La clase pedida no queda verificada sola."
            elif key in (ord("p"), ord("P")):
                self._repeat_short_classes()
            return False
        if self.phase == "review":
            if key == ord(" "):
                if self.player is not None:
                    self.player.toggle()
                return False
            if key == ord(","):
                if self.player is not None:
                    self.player.step(-1)
                return False
            if key == ord("."):
                if self.player is not None:
                    self.player.step(1)
                return False
            if key in (ord("c"), ord("C")):
                self._begin_round()
                return False
            self._review_key(key)
            return False
        if key in (ord("s"), ord("S")) and self.order == "blocked":
            self._skip_attempt(now_ms)
            return False
        if key in (ord("t"), ord("T")) and self.order == "blocked":
            self._repeat_block_attempt(now_ms)
            return False
        if key == ord(" ") and self.order == "blocked" and self.phase == "ready":
            self._open_target(now_ms)
            return False
        if key == ord(" "):
            self._space()
        elif key in (ord("r"), ord("R")) and self.phase not in {"review", "history"}:
            self._reset(now_ms)
        elif key in (ord("v"), ord("V")) and self.phase not in {"idle", "history"}:
            if self.order == "blocked" and not self.blocks_finished:
                self.message = "La revision se abre al terminar los bloques. Aun no llegas al final."
                return False
            self.phase = "review"
            self._label_choice = None
            self._load_player()
            self.message = "Revisión. Elige 0-4 y pulsa G. Nada se aprueba solo."
        return False

    def view(self, screen_xy: tuple[int, int] | None, screen_radius: int) -> TargetView:
        label = self.current_label
        remaining = None
        countdown = None
        if self.phase == "countdown":
            countdown = max(1, int((self.countdown_ms - self._elapsed + 999) / 1000))
        if self.phase in {"active", "tail"}:
            remaining = max(0, self.active_ms - self._elapsed)
        counts = self.review.counts()
        verified = self.review.verified_counts()
        captured = sum(counts.values())
        pending = counts["needs_review"] + counts["provisional"]
        rejected = counts["discarded"] + counts["invalid"]
        valid = counts["needs_review"] + counts["provisional"] + counts["verified"]
        if self.order == "blocked":
            tally, reasons = self._class_tally()
            progress = (
                self.block.header(),
                f"PROGRESO: {self.block.attempts}/{self.block.class_goal()}",
                f"OBJETIVO ACTUAL: {MOVEMENTS[self.block.label].display_name}",
                f"Intentos {self.attempts}. Validos {valid}. Rechazados {rejected}. Pendientes {pending}.",
                " | ".join(
                    f"{MOVEMENTS[label].display_name[:12]} {tally[label]['valid']}/{self.goal}"
                    for label in self.scheduled_classes
                ),
                f"Verificadas por revision humana: {sum(verified.values())}.",
                reasons[-1] if reasons else "Sin rechazos en esta sesion.",
            )
        else:
            progress = (
                f"Juego: aciertos {self.hits}, fallos {self.misses}. No es precision NEAT.",
                f"Capturadas {captured}. Pendientes {pending}. Rechazadas {rejected}.",
                *(
                    f"{MOVEMENTS[label_id].display_name}: {verified[label_id]}/{self.goal} verificadas"
                    for label_id in MOVEMENTS
                ),
                *dataset_gap_lines(self.config.data_dir, self.goal),
                f"Sesiones de revision: {len(list_review_sessions(self.config.data_dir))}.",
            )
        review_lines = ()
        if self.phase == "summary":
            review_lines = self._summary_lines()
        elif self.phase == "review":
            review_lines = self._review_lines(counts)
        elif self.phase == "history":
            review_lines = self._history_lines()
        show_ring = label not in (None, 0) and self.phase in {"active", "tail"} and screen_xy is not None
        fade = 0.0
        if self.phase == "tail" and self.outcome in SUCCESS:
            fade = min(1.0, self._elapsed / max(1, self.active_ms))
        status = self.message
        if self.candidate and self.phase == "active" and not self._locked:
            status = f"Candidato: {self.candidate}"
        snapshot = None if self.player is None or self.phase != "review" else self.player.snapshot()
        return TargetView(
            phase=self.phase,
            title=self._title(),
            combo_text=self._combo_text(),
            countdown_s=countdown,
            remaining_ms=remaining,
            score=self.score,
            hits=self.hits,
            misses=self.misses,
            tracking=self.tracking,
            status_line=status,
            progress_lines=progress,
            review_lines=review_lines,
            controls=self._controls(),
            target_name=None if label is None else MOVEMENTS[label].display_name,
            target_color=COLORS.get(label if label is not None else 0, COLORS[0]),
            screen_xy=screen_xy,
            screen_radius=screen_radius,
            fade=fade,
            show_ring=show_ring,
            skeleton_points=() if snapshot is None else snapshot.points,
            skeleton_links=() if snapshot is None else snapshot.connections,
            playback_line="" if snapshot is None else (
                f"Fotograma {snapshot.index + 1}/{snapshot.count}  "
                f"{'reproduciendo' if snapshot.playing else 'pausa'}  "
                f"{snapshot.elapsed_ms} ms"
            ),
        )

    def advance_playback(self, dt_ms: int) -> None:
        if self.phase == "review" and self.player is not None:
            self.player.advance(dt_ms)

    def _begin_round(self) -> None:
        self.resolved_in_round = 0
        self.phase = "countdown"
        self.paused = False
        self.player = None
        self._elapsed = 0
        self._last_ms = None
        self.message = "Preparate"

    def _space(self) -> None:
        if self.phase == "idle":
            self.resolved_in_round = 0
            self.phase = "countdown"
            self.paused = False
            self._elapsed = 0
            self._last_ms = None
            self.message = "Preparate"
            return
        self.paused = not self.paused
        self._last_ms = None
        self.message = "En pausa" if self.paused else "En marcha"

    def _reset(self, now_ms: int) -> None:
        if now_ms > self._reset_armed_until:
            self._reset_armed_until = now_ms + 1500
            self.message = "Pulsa R otra vez para reiniciar la ronda"
            return
        self.phase = "idle"
        self.paused = False
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.resolved_in_round = 0
        self.current_label = None
        self._locked = False
        self._frames = []
        self._elapsed = 0
        self._last_ms = None
        self.generator = TargetGenerator(goal=self.goal, seed=self.seed)
        sequence = BLOCK_LABELS if self.only_class is None else (self.only_class,)
        self.scheduled_classes = sequence
        self.block = BlockPlan(self.goal, sequence=sequence)
        self.blocks_finished = False
        self.attempts = 0
        self.message = "Ronda reiniciada. ESPACIO para iniciar"

    def _open_target(self, now_ms: int) -> None:
        if self.order == "blocked":
            self.current_label = self.block.label
        else:
            self.current_label = self.generator.next_label(self.review.verified_counts())
        self.phase = "active"
        self._elapsed = 0
        self._last_ms = None
        self._window_start = now_ms
        self._frames = []
        self._consumed = None
        self._locked = False
        self.outcome = ""
        self.detector_label = None
        self.detector_name = ""
        self._wrist_distance = None
        if self.order == "blocked" and self.current_label == 0:
            self.message = "Guardia. No hace falta golpear ni mover la muneca. Manten el cuerpo visible."
        else:
            self.message = MOVEMENTS[self.current_label].display_name

    def _lock(self, outcome: str, failure: bool) -> None:
        self._locked = True
        self.outcome = outcome
        self.phase = "tail"
        if outcome in SUCCESS:
            self.score += 100
            self.hits += 1
            self.message = "Acierto"
        elif failure:
            self.misses += 1
            self.message = _failure_text(outcome)

    def _close_window(self) -> None:
        if self.current_label is None or self._window_start is None:
            return
        self.review.ingest(
            CaptureDraft(
                requested_label=self.current_label,
                outcome=self.outcome or "fallo",
                detector_label=self.detector_label,
                detector_name=self.detector_name,
                frames=tuple(self._frames),
                window_start_ms=self._window_start,
                window_ms=self.active_ms,
                target_offset=offset_for(self.current_label),
                radius=radius_for(self.current_label),
                game_score=self.score,
                wrist_distance=self._wrist_distance,
            ),
            self.config,
            width=self.width,
            height=self.height,
        )
        self.resolved_in_round += 1
        self.attempts += 1
        self._frames = []
        self._locked = False
        self.current_label = None
        if self.order == "blocked":
            self._advance_block()
            return
        if self.resolved_in_round >= self.round_size:
            self.phase = "review"
            self._label_choice = None
            self.selected = 0
            self.message = "Ronda terminada. Elige 0-4 y pulsa G. Puedes dejar capturas sin decidir."
            self._load_player()
            return
        self.phase = "gap"
        self._elapsed = 0
        self._last_ms = None
        self.message = "Siguiente objetivo"

    def _advance_block(self) -> None:
        decision = self.block.record_attempt()
        if decision == "done":
            self.blocks_finished = True
            self.phase = "summary"
            self._label_choice = None
            self.selected = 0
            self.player = None
            self.message = "Bloques terminados. Nada queda verificado hasta que elijas la etiqueta."
            return
        if decision == "next":
            self.block.advance()
        self._hold_for_next()

    def _remaining_labels(self) -> tuple[int, ...]:
        if self.blocks_finished:
            return ()
        pending = []
        if self.block.attempts < self.block.class_goal():
            pending.append(self.block.label)
        pending.extend(self.block.sequence[self.block.index + 1 :])
        return tuple(pending)

    def _class_tally(self) -> tuple[dict[int, dict[str, int]], list[str]]:
        tally = {label: {"attempts": 0, "valid": 0, "rejected": 0, "pending": 0} for label in MOVEMENTS}
        reasons: list[str] = []
        for _path, payload in self.review.entries():
            label = payload.get("requested_label_id")
            if not isinstance(label, int) or label not in tally:
                continue
            tally[label]["attempts"] += 1
            status = str(payload.get("verification_status"))
            if status in {"needs_review", "provisional", "verified"}:
                tally[label]["valid"] += 1
            if status == "invalid":
                tally[label]["rejected"] += 1
                quality = payload.get("quality_reasons")
                if isinstance(quality, list) and quality:
                    reasons.append(f"{MOVEMENTS[label].display_name}: {quality[0]}")
            if status in {"needs_review", "provisional"}:
                tally[label]["pending"] += 1
        return tally, reasons

    def _summary_lines(self) -> tuple[str, ...]:
        tally, reasons = self._class_tally()
        lines = [
            f"Intentos realizados: {self.attempts}",
            f"Validos: {sum(item['valid'] for item in tally.values())}",
            f"Rechazados: {sum(item['rejected'] for item in tally.values())}",
            f"Pendientes de revision: {sum(item['pending'] for item in tally.values())}",
            "Bloque actual: terminado" if self.blocks_finished else self.block.header(),
        ]
        missing: list[str] = []
        for label in self.scheduled_classes:
            item = tally[label]
            lines.append(
                f"{MOVEMENTS[label].display_name}: intentos {item['attempts']}, "
                f"validos {item['valid']}, rechazados {item['rejected']}, pendientes {item['pending']}"
            )
            if item["valid"] < self.goal:
                missing.append(f"{MOVEMENTS[label].display_name} ({self.goal - item['valid']})")
        if reasons:
            lines.append("Motivos de rechazo:")
            lines.extend(reasons[-4:])
        else:
            lines.append("No hubo rechazos del validador.")
        if missing:
            lines.append("Faltan validas: " + ", ".join(missing) + ". P las repite.")
        else:
            lines.append("Cada clase pedida tiene sus capturas validas. Siguen sin estar verificadas.")
        lines.append("ESPACIO abre el esqueleto. 0-4 elige la etiqueta y G guarda solo esa.")
        return tuple(lines)

    def _repeat_short_classes(self) -> None:
        tally, _reasons = self._class_tally()
        sequence: list[int] = []
        quotas: list[int] = []
        for label in self.scheduled_classes:
            missing = self.goal - tally[label]["valid"]
            if missing > 0:
                sequence.append(label)
                quotas.append(missing)
        if not sequence:
            self.message = "No faltan capturas validas. Puedes revisarlas con ESPACIO."
            return
        self.block = BlockPlan(self.goal, sequence=tuple(sequence), quotas=tuple(quotas))
        self.blocks_finished = False
        self._hold_for_next()
        self.message = "Repeticion de lo que falto. " + self.message

    def _hold_for_next(self) -> None:
        self.phase = "ready"
        self.paused = False
        self._elapsed = 0
        self._last_ms = None
        self.current_label = self.block.label
        name = MOVEMENTS[self.block.label].display_name.upper()
        self.message = f"SIGUE: {name}. ESPACIO para continuar."

    def handle_click(self, x: int, y: int, width: int, height: int) -> None:
        """El clic no verifica muestras. La etiqueta la elige el usuario con 0-4 y G."""

        del x, y, width, height

    def _skip_attempt(self, now_ms: int) -> None:
        if self.phase not in {"active", "tail"}:
            self.message = "S salta solo el intento que está en curso."
            return
        self._frames = []
        self._locked = False
        self._open_target(now_ms)
        self.message = "Intento saltado. No se guardó ni se contó."

    def _repeat_block_attempt(self, now_ms: int) -> None:
        if self.phase != "ready" or self.block.index == 0 or self.block.attempts != 0:
            self.message = "T repite un intento cuando ya se muestra la tecnica siguiente."
            return
        self.block.index -= 1
        self.block.attempts = max(0, self.block.class_goal() - 1)
        self._hold_for_next()
        self.message = "Un intento mas de la tecnica anterior. ESPACIO para continuar."

    def open_saved_session(self, session_id: str) -> None:
        """Cambia a una sesión ya escrita, sin borrar sus archivos."""

        self.review = open_review_session(self.config.data_dir, session_id)
        self.phase = "review"
        self.selected = 0
        self._label_choice = None
        self.message = "Sesión recuperada. Elige 0-4 y pulsa G. Nada se aprueba solo."
        self._load_player()

    def _review_key(self, key: int) -> None:
        from app.dataset_manager import DatasetManager

        entries = self._review_entries()
        if not entries:
            self.message = "No quedan capturas por revisar. Puedes dejar la sesión así."
            return
        self.selected %= len(entries)
        path, _payload = entries[self.selected]
        if key in (ord("n"), ord("N")):
            self.selected = (self.selected + 1) % len(entries)
            self._label_choice = None
            self._load_player()
        elif key in (ord("b"), ord("B")):
            self.selected = (self.selected - 1) % len(entries)
            self._label_choice = None
            self._load_player()
        elif key in (ord("m"), ord("M")) and self.order == "blocked":
            current = entries[self.selected][1].get("requested_label_id")
            for offset in range(1, len(entries) + 1):
                candidate = (self.selected + offset) % len(entries)
                if entries[candidate][1].get("requested_label_id") != current:
                    self.selected = candidate
                    self._label_choice = None
                    self._load_player()
                    break
        elif key in (ord("x"), ord("X")):
            if _payload.get("verification_status") == "verified":
                self.message = "X no borra una muestra ya guardada. Elige 0-4 y pulsa G para corregirla."
                return
            self.review.discard(path)
            self._label_choice = None
            self._load_player()
            self.message = "Captura descartada. No entró al dataset."
        elif (label := _label_from_key(key)) is not None:
            # La etiqueta 0 es Guardia. No se puede descartar con un if choice: porque 0 es falso.
            self._label_choice = label
            self.message = f"Etiqueta {label}: {MOVEMENTS[label].display_name}. Pulsa G para guardarla."
        elif key in (ord("g"), ord("G")):
            if self._label_choice is None:
                self.message = "G no guardó nada. Primero elige una etiqueta con 0-4."
                return
            chosen = self._label_choice
            if _payload.get("verification_status") == "verified":
                saved = self.review.relabel_verified(path, chosen, DatasetManager(self.config.data_dir), self.config)
            else:
                self.review.correct(path, chosen)
                saved = self.review.approve(path, DatasetManager(self.config.data_dir), self.config)
            self._label_choice = None
            self._load_player()
            if saved:
                self.message = f"Guardada como {MOVEMENTS[chosen].display_name}."
            else:
                self.message = _save_failure(path)

    def _review_entries(self) -> list[tuple[object, dict[str, object]]]:
        if self.review_ids:
            selected = []
            for item in self.review.entries():
                sample = item[1].get("sample")
                sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
                if sample_id in self.review_ids:
                    selected.append(item)
            return selected
        if not self.review_verified:
            return self._pending_entries()
        selected = []
        for item in self.review.entries():
            status = item[1].get("verification_status")
            if status in {"provisional", "needs_review"}:
                selected.append(item)
            elif status == "verified" and item[1].get("approved_label_id") != item[1].get("requested_label_id"):
                selected.append(item)
        return selected

    def _pending_entries(self) -> list[tuple[object, dict[str, object]]]:
        pending = [
            item
            for item in self.review.entries()
            if item[1].get("verification_status") in {"provisional", "needs_review"}
        ]
        if self.order == "blocked":
            order = {label: index for index, label in enumerate(BLOCK_LABELS)}
            pending.sort(
                key=lambda item: order.get(_label_or_missing(item[1].get("requested_label_id")), 9)
            )
        return pending

    def _review_lines(self, counts: dict[str, int]) -> tuple[str, ...]:
        lines = [
            "0 GUARDIA    1 JAB    2 CROSS",
            "3 GANCHO IZQ    4 GANCHO DER",
            "No hay tecla 5. La guardia es la tecla 0.",
            self.message,
            f"Esta ronda: aciertos {self.hits}, fallos {self.misses}, puntos {self.score}.",
            (
                f"Sesión: provisionales {counts['provisional']}, dudosas {counts['needs_review']}, "
                f"inválidas {counts['invalid']}, verificadas {counts['verified']}, descartadas {counts['discarded']}."
            ),
            *dataset_gap_lines(self.config.data_dir, self.goal),
            "0-4 elige la etiqueta. G la guarda. X descarta. N deja esta captura sin decidir.",
            "ESPACIO reproduce o pausa. Coma y punto avanzan un fotograma.",
        ]
        entries = self._review_entries()
        if not entries:
            lines.append("No hay capturas pendientes.")
            return tuple(lines)
        self.selected %= len(entries)
        _path, payload = entries[self.selected]
        lines.append(f"Pendiente {self.selected + 1} de {len(entries)}. Las demás pueden quedar sin decidir.")
        lines.extend(capture_details(payload))
        if self.player is not None:
            lines.extend(self.player.snapshot().angle_lines)
            lines.append(self.player.snapshot().notice)
        if self._label_choice is not None:
            lines.append(f"Lista para guardar como: {MOVEMENTS[self._label_choice].display_name}")
        return tuple(lines)

    def _load_player(self) -> None:
        entries = self._review_entries()
        if not entries:
            self.player = None
            return
        self.selected %= len(entries)
        payload = entries[self.selected][1]
        sample = payload.get("sample")
        frames = sample.get("frames") if isinstance(sample, dict) else None
        mirror = False
        if isinstance(sample, dict) and isinstance(sample.get("camera"), dict):
            mirror = bool(sample["camera"].get("mirror_view"))
        self.player = SkeletonPlayer(frames if isinstance(frames, list) else [], mirror=mirror)

    def _open_history(self) -> None:
        self._history = list_review_sessions(self.config.data_dir)
        self._history_index = 0
        self.phase = "history"
        self.message = "Sesiones guardadas. G abre la seleccionada."

    def _history_key(self, key: int) -> None:
        if not self._history:
            if key == ord(" "):
                self.phase = "idle"
            return
        if key in (ord("n"), ord("N")):
            self._history_index = (self._history_index + 1) % len(self._history)
        elif key in (ord("b"), ord("B")):
            self._history_index = (self._history_index - 1) % len(self._history)
        elif key in (ord("g"), ord("G"), 13):
            session_id = str(self._history[self._history_index]["session_id"])
            self.open_saved_session(session_id)
        elif key == ord(" "):
            self.phase = "idle"
            self.message = "ESPACIO inicia una ronda nueva."

    def _history_lines(self) -> tuple[str, ...]:
        if not self._history:
            return ("No hay sesiones con capturas.", "ESPACIO vuelve al inicio.")
        lines = ["Elige una sesión guardada. No se repetirán los golpes.", "N siguiente, B anterior, G abre, ESPACIO vuelve."]
        for index, item in enumerate(self._history):
            mark = ">" if index == self._history_index else " "
            lines.append(
                f"{mark} {item['session_id']} | {item['captures']} capturas | "
                f"provisionales {item['provisional']} | dudosas {item['needs_review']} | "
                f"verificadas {item['verified']}"
            )
        lines.append("No hay video de esas ejecuciones. La revisión usa la biomecánica guardada.")
        return tuple(lines)

    def _controls(self) -> str:
        if self.phase == "summary":
            return "ESPACIO abre la revision   P repite las clases incompletas   Q sale"
        if self.phase == "review":
            return "0 GUARDIA  1 JAB  2 CROSS  3 G.IZQ  4 G.DER   G guarda   N siguiente   Q sale"
        if self.order == "blocked" and self.phase == "ready":
            return "ESPACIO continua con la tecnica que se muestra   Q sale"
        if self.order == "blocked" and self.phase not in {"review", "history"}:
            return "ESPACIO pausa   S salta el intento   Q sale"
        if self.phase == "history":
            return "N/B elige  G abre  ESPACIO vuelve  Q sale"
        return "ESPACIO inicia o pausa   H sesiones guardadas   R reinicia   V revisa   Q sale"

    def _title(self) -> str:
        if self.order == "blocked" and self.phase != "review":
            return "EVOBOX AI — TARGET TRAINING"
        if self.phase == "idle":
            return "TARGET TRAINING"
        if self.phase == "review":
            return "REVISION"
        if self.phase == "history":
            return "SESIONES GUARDADAS"
        if self.paused:
            return "EN PAUSA"
        if not self.tracking and self.phase != "idle":
            return "BUSCANDO CUERPO"
        return "EN ENTRENAMIENTO"

    def _combo_text(self) -> str:
        if self.order != "blocked":
            return self.generator.combo_text()
        if self.phase == "ready":
            return f"SIGUE: {MOVEMENTS[self.block.label].display_name.upper()}"
        return self.block.header()


def _label_from_key(key: int) -> int | None:
    """Convierte la tecla en 0-4. El código 0 también es Guardia, no una tecla ausente."""

    if key in range(ord("0"), ord("5")):
        return key - ord("0")
    if key == 0:
        return 0
    if key > 255 and (key & 0xFF) in range(ord("0"), ord("5")):
        return (key & 0xFF) - ord("0")
    return None


def _label_or_missing(value: object) -> int:
    if isinstance(value, int):
        return value
    return -1


def _save_failure(path: object) -> str:
    import json
    from pathlib import Path

    if not isinstance(path, Path):
        return "No se guardó."
    payload = json.loads(path.read_text(encoding="utf-8"))
    reasons = payload.get("quality_reasons")
    if isinstance(reasons, list) and reasons:
        return "No se guardó. " + str(reasons[0])
    if payload.get("verification_status") == "verified":
        return "Esa captura ya estaba guardada. No se duplicó."
    if payload.get("approved_label_id") is None:
        return "No se guardó. Falta elegir una etiqueta. La 0 también vale."
    return "No se guardó. Revisa el estado o un duplicado."


def _failure_text(outcome: str) -> str:
    return {
        "equivocado": "Golpe equivocado",
        "lejos": "Lejos del objetivo",
        "no_identificado": "Movimiento no identificado",
        "fallo": "Objetivo fallado",
        "guardia_rota": "Saliste de guardia",
    }.get(outcome, "Objetivo fallado")
