from __future__ import annotations
from collections.abc import Callable
from grid import Cell, Grid, SearchResult


class TwoTierPlanner:

    def __init__(self, partition: str = "watershed", refine: str = "nearest",
                 partition_params: dict | None = None, cost_mode: str = "centroid",
                 heuristic: str = "octile") -> None:
        raise NotImplementedError

    def preprocess(self, grid: Grid) -> float:
        raise NotImplementedError

    def find_route(self, start: Cell, goal: Cell) -> SearchResult:
        raise NotImplementedError

    def refine_nearest(self, route: list[int], start: Cell, goal: Cell,
                       on_expand: Callable[[Cell], None] | None = None) -> SearchResult:
        raise NotImplementedError

    def refine_corridor(self, route: list[int], start: Cell, goal: Cell,
                        on_expand: Callable[[Cell], None] | None = None) -> SearchResult:
        raise NotImplementedError

    def query(self, start: Cell, goal: Cell,
              on_expand: Callable[[Cell], None] | None = None) -> SearchResult:
        raise NotImplementedError

    def update(self, cell: Cell, blocked: bool) -> float:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError
