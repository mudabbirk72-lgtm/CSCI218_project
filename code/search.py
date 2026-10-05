

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Iterable

from hierpath.grid import Cell, Grid, SearchResult


def manhattan(a: Cell, b: Cell) -> float:
    raise NotImplementedError


def octile(a: Cell, b: Cell) -> float:
    raise NotImplementedError


def astar(start: Hashable,
          goals: Collection[Hashable],
          neighbours: Callable[[Hashable], Iterable[tuple[Hashable, float]]],
          heuristic: Callable[[Hashable], float],
          on_expand: Callable[[Hashable], None] | None = None) -> SearchResult:
    raise NotImplementedError


def dijkstra(start: Hashable,
             goals: Collection[Hashable],
             neighbours: Callable[[Hashable], Iterable[tuple[Hashable, float]]],
             on_expand: Callable[[Hashable], None] | None = None) -> SearchResult:
    raise NotImplementedError


def dijkstra_all(start: Hashable,
                 neighbours: Callable[[Hashable], Iterable[tuple[Hashable, float]]],
                 targets: Collection[Hashable] | None = None
                 ) -> tuple[dict[Hashable, float], dict[Hashable, Hashable | None]]:
    raise NotImplementedError


def reconstruct_path(parent: dict[Hashable, Hashable | None], target: Hashable) -> list[Hashable]:
    raise NotImplementedError


class FlatPlanner:

    def __init__(self, algorithm: str = "astar", heuristic: str = "octile") -> None:
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
