from __future__ import annotations
from collections.abc import Callable, Iterator
from .area_graph import Gap, find_crossings, find_gaps
from .grid import Cell, Grid, SearchResult
from .search import dijkstra_all
import numpy as np


def place_doorways(gap: Gap, max_gap: int = 6) -> list[tuple[Cell, Cell]]:
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")

    pairs = sorted(gap.pairs, key=lambda pair: pair[0])
    width = len(pairs)

    if width == 0:
        return []

    if width < max_gap:
        return [pairs[width // 2]]

    return [pairs[0], pairs[-1]]


class DoorwayGraph:

    def __init__(self, grid: Grid, labels: np.ndarray, max_gap: int = 6) -> None:
        from time import perf_counter

        start_time = perf_counter()

        self.grid = grid
        self.labels = labels
        self.max_gap = max_gap
        self.nodes = {}
        self.adj = {}
        self.paths = {}
        self.by_area = {}
        self.temporary = set()


        node_lookup = {}

        def add_node(area_id: int, cell: Cell) -> int:
            key = (area_id, cell)

            if key not in node_lookup:
                node_id = len(self.nodes)
                node_lookup[key] = node_id
                self.nodes[node_id] = key
                self.adj[node_id] = {}
                self.by_area.setdefault(area_id, []).append(node_id)

            return node_lookup[key]

        def add_edge(u: int, v: int, cost: float,
                     path: list[Cell]) -> None:
            self.adj[u][v] = cost
            self.adj[v][u] = cost

            if u < v:
                self.paths[(u, v)] = path
            else:
                self.paths[(v, u)] = list(reversed(path))

        crossings = find_crossings(grid.blocked, labels)
        gaps = find_gaps(crossings)

        for gap in gaps:
            for cell_a, cell_b in place_doorways(gap, max_gap):
                u = add_node(gap.a, cell_a)
                v = add_node(gap.b, cell_b)

                add_edge(u, v, 1.0, [cell_a, cell_b])

        for area_id, doorway_ids in self.by_area.items():
            allowed_neighbours = grid.restricted_neighbours(
                labels, {area_id}
            )

            for u in doorway_ids:
                start_cell = self.nodes[u][1]

                distances, parent = dijkstra_all(
                    start_cell, allowed_neighbours
                )

                for v in doorway_ids:
                    if u >= v:
                        continue

                    goal_cell = self.nodes[v][1]

                    if goal_cell not in distances:
                        continue

                    path = []
                    current = goal_cell

                    while current is not None:
                        path.append(current)
                        if current == start_cell:
                            break
                        current = parent[current]

                    if path[-1] != start_cell:
                        continue

                    path.reverse()
                    add_edge(u, v, distances[goal_cell], path)

        self.build_time_s = perf_counter() - start_time


    def neighbours(self, node: int) -> Iterator[tuple[int, float]]:
        """Return neighbouring doorway nodes and their edge costs."""
        return iter(self.adj[node].items())

    def heuristic(
        self, goal_cell: Cell, name: str = "octile"
    ) -> Callable[[int], float]:
        """Estimate the distance from a doorway to the goal cell."""
        if name not in ("octile", "manhattan"):
            raise ValueError("name must be 'octile' or 'manhattan'")

        def h(node: int) -> float:
            cell = self.nodes[node][1]
            dx = abs(cell[0] - goal_cell[0])
            dy = abs(cell[1] - goal_cell[1])

            if name == "manhattan":
                return float(dx + dy)

            return float(max(dx, dy) + (2**0.5 - 1) * min(dx, dy))

        return h

    def segment(self, u: int, v: int) -> list[Cell]:
        if v not in self.adj[u]:
            raise KeyError((u, v))

        if u < v:
            return list(self.paths[(u, v)])

        return list(reversed(self.paths[(v, u)]))

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
