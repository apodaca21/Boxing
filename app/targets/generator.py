"""Secuencias balanceadas. La clase pedida no sale del detector."""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from app.dataset_manager import MOVEMENTS


BLOCK_LABELS: tuple[int, ...] = (1, 2, 3, 4, 0)


@dataclass
class BlockPlan:
    """Bloques de un solo golpe. El avance cuenta intentos, no verificaciones."""

    goal: int = 10
    index: int = 0
    attempts: int = 0
    sequence: tuple[int, ...] = BLOCK_LABELS
    quotas: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.goal < 1:
            raise ValueError("Cada bloque necesita al menos un intento.")
        if not self.sequence or any(label not in MOVEMENTS for label in self.sequence):
            raise ValueError("La secuencia de bloques no es válida.")
        if self.quotas is not None and len(self.quotas) != len(self.sequence):
            raise ValueError("Cada bloque necesita su propia cantidad de intentos.")

    @property
    def label(self) -> int:
        return self.sequence[min(self.index, len(self.sequence) - 1)]

    @property
    def complete(self) -> bool:
        return self.index >= len(self.sequence)

    def class_goal(self) -> int:
        if self.quotas is None:
            return self.goal
        return self.quotas[min(self.index, len(self.quotas) - 1)]

    def header(self) -> str:
        movement = MOVEMENTS[self.label].display_name
        return f"BLOQUE {self.index + 1}/{len(self.sequence)} — {movement.upper()}"

    def record_attempt(self) -> str:
        self.attempts += 1
        if self.attempts < self.class_goal():
            return "continue"
        if self.index + 1 >= len(self.sequence):
            return "done"
        return "next"

    def advance(self) -> None:
        self.index += 1
        self.attempts = 0

    def labels(self) -> tuple[int, ...]:
        counts = self.quotas or (self.goal,) * len(self.sequence)
        return tuple(label for label, count in zip(self.sequence, counts, strict=True) for _ in range(count))


COMBOS: tuple[tuple[int, ...], ...] = (
    (1,),
    (2,),
    (3,),
    (4,),
    (0,),
    (1, 2),
    (1, 2, 3),
    (2, 4),
    (4, 3),
    (0,),
)


@dataclass
class TargetGenerator:
    """Prioriza las clases que todavía no llegan a la meta verificada."""

    goal: int = 10
    seed: int = 42
    combo: tuple[int, ...] = ()
    combo_index: int = 0
    _queue: list[int] = field(default_factory=list)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        if self.goal < 1:
            raise ValueError("La meta por clase debe ser al menos 1.")
        self._rng = random.Random(self.seed)

    def next_label(self, verified_counts: dict[int, int]) -> int:
        if not self._queue:
            self.combo = self.choose_combo(verified_counts)
            self.combo_index = 0
            self._queue = list(self.combo)
        label = self._queue.pop(0)
        self.combo_index = len(self.combo) - len(self._queue) - 1
        return label

    def choose_combo(self, verified_counts: dict[int, int]) -> tuple[int, ...]:
        """Una pasada con las cinco clases, o solo las que aún no llegan a la meta."""

        needed = [
            label
            for label in range(5)
            if int(verified_counts.get(label, 0)) < self.goal
        ]
        if not needed:
            needed = [0, 1, 2, 3, 4]
        return tuple(needed)

    def combo_text(self) -> str:
        if not self.combo:
            return ""
        names = [MOVEMENTS[label].display_name for label in self.combo]
        return " > ".join(names)

    def _weight(self, combo: tuple[int, ...], verified_counts: dict[int, int]) -> int:
        return sum(
            max(0, self.goal - int(verified_counts.get(label, 0)))
            for label in set(combo)
        )
