
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]



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

        raise NotImplementedError

    @classmethod
    def from_strings(cls, rows: list[str], diagonal: bool = True) -> Grid:
        raise NotImplementedError

    def in_bounds(self, cell: Cell) -> bool:
        raise NotImplementedError

    def is_free(self, cell: Cell) -> bool:
        raise NotImplementedError

    def neighbours(self, cell: Cell) -> Iterator[tuple[Cell, float]]:
        raise NotImplementedError

    def restricted_neighbours(self, labels: np.ndarray,
                              allowed: set[int]) -> Callable[[Cell], Iterator[tuple[Cell, float]]]:
        raise NotImplementedError

    def step_cost(self, a: Cell, b: Cell) -> float:
        raise NotImplementedError

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
