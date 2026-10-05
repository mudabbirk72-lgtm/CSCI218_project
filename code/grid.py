
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]

SQRT2 = 2 ** 0.5

WALL_CHARS = frozenset("#@")

STRAIGHT_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_MOVES = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class SearchResult:

    path: list | None
    cost: float
    expanded: int
    generated: int
    runtime_s: float
    success: bool
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    start: Cell
    goal: Cell
    optimal_cost: float
    id: int = 0


class Grid:

    def __init__(self, blocked: np.ndarray, diagonal: bool = True) -> None:
        blocked = np.asarray(blocked)
        if blocked.ndim != 2:
            raise ValueError(f"blocked must be 2-dimensional, got {blocked.ndim} dimension(s)")
        if blocked.dtype != np.bool_:
            raise ValueError(f"blocked must have dtype bool, got {blocked.dtype}")
        self.blocked = blocked
        self.height, self.width = blocked.shape
        self.diagonal = diagonal


    @classmethod
    def from_strings(cls, rows: list[str], diagonal: bool = True) -> Grid:
        if not rows:
            raise ValueError("rows is empty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("all rows must have the same length")
        blocked = np.array([[ch in WALL_CHARS for ch in row] for row in rows], dtype=bool)
        return cls(blocked, diagonal)


    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width
    

    def is_free(self, cell: Cell) -> bool:
        r, c = cell
        # Bounds are checked first: a negative index would otherwise wrap around in NumPy.
        return 0 <= r < self.height and 0 <= c < self.width and not self.blocked[r, c]

    def neighbours(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        r, c = cell
        blocked, h, w = self.blocked, self.height, self.width
        for dr, dc in STRAIGHT_MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc]:
                yield (nr, nc), 1.0
        if self.diagonal:
            for dr, dc in DIAGONAL_MOVES:
                nr, nc = r + dr, c + dc
                # No corner cutting: the destination AND both side cells must be open.
                # (If the destination is in bounds, both side cells are in bounds too.)
                if (0 <= nr < h and 0 <= nc < w and not blocked[nr, nc]
                        and not blocked[nr, c] and not blocked[r, nc]):
                    yield (nr, nc), SQRT2


    def restricted_neighbours(self, labels: np.ndarray,
                              allowed: set[int]) -> Callable[[Cell], Iterator[tuple[Cell, float]]]:
        raise NotImplementedError
    

    def step_cost(self, a: Cell, b: Cell) -> float:
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        if dr + dc == 1:
            return 1.0
        if dr == 1 and dc == 1 and self.diagonal:
            return SQRT2
        raise ValueError(f"cells {a} and {b} are not adjacent")


    def path_cost(self, path: list[Cell]) -> float:
        raise NotImplementedError


    def is_valid_path(self, path: list[Cell]) -> bool:
        raise NotImplementedError


    def block(self, cell: Cell) -> None:
        raise NotImplementedError


    def unblock(self, cell: Cell) -> None:
        raise NotImplementedError


    def copy(self) -> Grid:
        raise NotImplementedError
