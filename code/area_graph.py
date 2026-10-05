
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from hierpath.grid import Cell


@dataclass
class Gap:

    a: int
    b: int
    pairs: list[tuple[Cell, Cell]]


def find_crossings(blocked: np.ndarray,
                   labels: np.ndarray) -> dict[tuple[int, int], list[tuple[Cell, Cell]]]:
    raise NotImplementedError


def find_gaps(crossings: dict[tuple[int, int], list[tuple[Cell, Cell]]]) -> list[Gap]:
    raise NotImplementedError


class AreaGraph:

    def __init__(self, blocked: np.ndarray, labels: np.ndarray, cost_mode: str = "centroid") -> None:
        raise NotImplementedError

    def area_of(self, cell: Cell) -> int:
        raise NotImplementedError

    def neighbours(self, area: int) -> Iterator[tuple[int, float]]:
        raise NotImplementedError

    def heuristic(self, goal_area: int) -> Callable[[int], float]:
        raise NotImplementedError
