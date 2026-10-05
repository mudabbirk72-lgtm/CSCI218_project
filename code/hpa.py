

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from hierpath.area_graph import Gap
from hierpath.grid import Cell, Grid, SearchResult


def place_doorways(gap: Gap, max_gap: int = 6) -> list[tuple[Cell, Cell]]:
    raise NotImplementedError


class DoorwayGraph:

    def __init__(self, grid: Grid, labels: np.ndarray, max_gap: int = 6) -> None:
        raise NotImplementedError

    def neighbours(self, node: int) -> Iterator[tuple[int, float]]:
        raise NotImplementedError

    def heuristic(self, goal_cell: Cell, name: str = "octile") -> Callable[[int], float]:
        raise NotImplementedError

    def segment(self, u: int, v: int) -> list[Cell]:
        raise NotImplementedError

    def insert(self, cell: Cell) -> int:
        raise NotImplementedError

    def remove_temporary(self) -> None:
        raise NotImplementedError

    def rebuild_areas(self, areas: set[int]) -> int:
        raise NotImplementedError


class HPAPlanner:

    def __init__(self, partition: str = "watershed", partition_params: dict | None = None,
                 max_gap: int = 6, heuristic: str = "octile") -> None:
        raise NotImplementedError

    def preprocess(self, grid: Grid) -> float:
        raise NotImplementedError

    def query(self, start: Cell, goal: Cell,
              on_expand: Callable[[Cell], None] | None = None) -> SearchResult:
        raise NotImplementedError

    def update(self, cell: Cell, blocked: bool) -> float:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError
