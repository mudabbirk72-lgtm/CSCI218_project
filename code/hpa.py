from __future__ import annotations
from collections.abc import Callable, Iterator
from .area_graph import Gap, find_crossings, find_gaps
from .grid import Cell, Grid, SearchResult
from .search import astar, dijkstra_all, manhattan, octile
from .partition import get_partitioner
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
        area_id = int(self.labels[cell])
        if area_id < 0:
            raise ValueError("cell must be an open cell")

        for node_id, (node_area, node_cell) in self.nodes.items():
            if node_area == area_id and node_cell == cell:
                return node_id

        node_id = max(self.nodes, default=-1) + 1
        self.nodes[node_id] = (area_id, cell)
        self.adj[node_id] = {}
        self.temporary.add(node_id)

        doorway_ids = self.by_area.get(area_id, [])
        if not doorway_ids:
            return node_id

        allowed_neighbours = self.grid.restricted_neighbours(
            self.labels, {area_id}
        )
        targets = {self.nodes[doorway_id][1] for doorway_id in doorway_ids}

        distances, parent = dijkstra_all(
            cell, allowed_neighbours, targets=targets
        )

        for doorway_id in doorway_ids:
            doorway_cell = self.nodes[doorway_id][1]
            if doorway_cell not in distances:
                continue

            path = []
            current = doorway_cell
            while current is not None:
                path.append(current)
                if current == cell:
                    break
                current = parent.get(current)

            if path[-1] != cell:
                continue

            path.reverse()
            cost = distances[doorway_cell]

            self.adj[node_id][doorway_id] = cost
            self.adj[doorway_id][node_id] = cost
            self.paths[(min(node_id, doorway_id),
                        max(node_id, doorway_id))] = (
                path if node_id < doorway_id else list(reversed(path))
            )

        return node_id

    def remove_temporary(self) -> None:
        for node_id in list(self.temporary):
            for neighbour in list(self.adj.get(node_id, {})):
                self.adj[neighbour].pop(node_id, None)

            self.adj.pop(node_id, None)
            self.nodes.pop(node_id, None)

        for edge in list(self.paths):
            if edge[0] in self.temporary or edge[1] in self.temporary:
                del self.paths[edge]

        self.temporary.clear()


    def rebuild_areas(self, areas: set[int]) -> int:
        areas = {int(area) for area in areas}
        if not areas:
            return 0

        old_nodes = {
            node_id
            for node_id, (area_id, _) in self.nodes.items()
            if area_id in areas and node_id not in self.temporary
        }

        for node_id in old_nodes:
            for neighbour in list(self.adj.get(node_id, {})):
                self.adj[neighbour].pop(node_id, None)
            self.adj.pop(node_id, None)
            self.nodes.pop(node_id, None)

        for edge in list(self.paths):
            if edge[0] in old_nodes or edge[1] in old_nodes:
                del self.paths[edge]

        for area_id in areas:
            self.by_area.pop(area_id, None)

        node_lookup = {
            (area_id, cell): node_id
            for node_id, (area_id, cell) in self.nodes.items()
        }

        def add_node(area_id: int, cell: Cell) -> int:
            key = (area_id, cell)
            if key in node_lookup:
                return node_lookup[key]

            node_id = max(self.nodes, default=-1) + 1
            node_lookup[key] = node_id
            self.nodes[node_id] = key
            self.adj[node_id] = {}

            if area_id in areas:
                self.by_area.setdefault(area_id, []).append(node_id)

            return node_id

        def add_edge(
            u: int, v: int, cost: float, path: list[Cell]
        ) -> None:
            self.adj[u][v] = cost
            self.adj[v][u] = cost
            self.paths[(min(u, v), max(u, v))] = (
                path if u < v else list(reversed(path))
            )

        crossings = find_crossings(self.grid.blocked, self.labels)
        gaps = find_gaps(crossings)

        for gap in gaps:
            if gap.a not in areas and gap.b not in areas:
                continue

            for cell_a, cell_b in place_doorways(gap, self.max_gap):
                key_a = (gap.a, cell_a)
                key_b = (gap.b, cell_b)

                if gap.a not in areas and key_a not in node_lookup:
                    continue
                if gap.b not in areas and key_b not in node_lookup:
                    continue

                u = add_node(gap.a, cell_a)
                v = add_node(gap.b, cell_b)
                add_edge(u, v, 1.0, [cell_a, cell_b])


        paths_recomputed = 0

        for area_id in areas:
            doorway_ids = self.by_area.get(area_id, [])
            allowed_neighbours = self.grid.restricted_neighbours(
                self.labels, {area_id}
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
                        current = parent.get(current)

                    if path[-1] != start_cell:
                        continue

                    path.reverse()
                    add_edge(u, v, distances[goal_cell], path)
                    paths_recomputed += 1

        return paths_recomputed



class HPAPlanner:

    def __init__(self, partition: str = "watershed", partition_params: dict | None = None,
                 max_gap: int = 6, heuristic: str = "octile") -> None:
        if partition not in ("fixed", "watershed", "louvain"):
            raise ValueError("Unknown partition method")
        if max_gap < 1:
            raise ValueError("max_gap must be at least 1")
        if heuristic not in ("octile", "manhattan"):
            raise ValueError("Unknown heuristic")

        self.partition = partition
        self.partition_params = (
            {} if partition_params is None else dict(partition_params)
        )
        self.max_gap = max_gap
        self.heuristic = heuristic

        self.name = f"HPA-{partition}"
        self.grid = None
        self.labels = None
        self.graph = None
        self.preprocess_s = 0.0
        self.last_update = {
            "areas_rebuilt": 0,
            "paths_recomputed": 0,
            "seconds": 0.0,
        }

    def preprocess(self, grid: Grid) -> float:
        from time import perf_counter

        start_time = perf_counter()

        partitioner = get_partitioner(self.partition)
        labels = partitioner(grid.blocked, **self.partition_params)
        graph = DoorwayGraph(grid, labels, max_gap=self.max_gap)

        elapsed = perf_counter() - start_time

        self.grid = grid
        self.labels = labels
        self.graph = graph
        self.preprocess_s = elapsed

        return elapsed

    def query(self, start: Cell, goal: Cell,
              on_expand: Callable[[Cell], None] | None = None) -> SearchResult:

        from time import perf_counter

        if self.grid is None or self.labels is None or self.graph is None:
            raise RuntimeError("Call preprocess() before query()")

        if not self.grid.is_free(start):
            raise ValueError("Invalid start cell")
        if not self.grid.is_free(goal):
            raise ValueError("Invalid goal cell")

        query_start = perf_counter()
        grid = self.grid
        labels = self.labels
        graph = self.graph

        candidate = None
        if labels[start] == labels[goal]:
            area_id = int(labels[start])
            neighbours = grid.restricted_neighbours(labels, {area_id})

            if self.heuristic == "manhattan":
                heuristic_fn = lambda cell: manhattan(cell, goal)
            else:
                heuristic_fn = lambda cell: octile(cell, goal)

            candidate = astar(
                start,
                {goal},
                neighbours,
                heuristic_fn,
                on_expand,
            )

        route = []
        tier1_expanded = 0
        tier1_generated = 0
        hierarchical_path = None
        hierarchical_cost = float("inf")

        try:
            start_node = graph.insert(start)
            goal_node = graph.insert(goal)

            tier1_result = astar(
                start_node,
                {goal_node},
                graph.neighbours,
                graph.heuristic(goal, self.heuristic),
                (
                    (lambda node: on_expand(graph.nodes[node][1]))
                    if on_expand is not None else None
                ),
            )

            tier1_expanded = tier1_result.expanded
            tier1_generated = tier1_result.generated

            if tier1_result.success and tier1_result.path is not None:
                route = tier1_result.path
                hierarchical_path = []

                for u, v in zip(route, route[1:]):
                    segment = graph.segment(u, v)

                    if hierarchical_path:
                        hierarchical_path.extend(segment[1:])
                    else:
                        hierarchical_path.extend(segment)

                if len(route) == 1:
                    hierarchical_path = [graph.nodes[route[0]][1]]

                hierarchical_cost = tier1_result.cost

        finally:
            graph.remove_temporary()

        candidate_success = candidate is not None and candidate.success
        hierarchy_success = hierarchical_path is not None

        if candidate_success and (
            not hierarchy_success or candidate.cost <= hierarchical_cost
        ):
            chosen_path = candidate.path
            chosen_cost = candidate.cost
            success = True
        elif hierarchy_success:
            chosen_path = hierarchical_path
            chosen_cost = hierarchical_cost
            success = True
        else:
            chosen_path = None
            chosen_cost = float("inf")
            success = False

        expanded = (
            candidate.expanded if candidate is not None else 0
        ) + tier1_expanded
        generated = (
            candidate.generated if candidate is not None else 0
        ) + tier1_generated

        return SearchResult(
            path=chosen_path,
            cost=chosen_cost,
            expanded=expanded,
            generated=generated,
            runtime_s=perf_counter() - query_start,
            success=success,
            extra={
                "route": route,
                "tier1_expanded": tier1_expanded,
            },
        )


    def update(self, cell: Cell, blocked: bool) -> float:
        raise NotImplementedError


    def stats(self) -> dict:
        areas = (
            int(np.unique(self.labels[self.labels >= 0]).size)
            if self.labels is not None
            else 0
        )

        doorways = (
            sum(len(node_ids) for node_ids in self.graph.by_area.values())
            if self.graph is not None
            else 0
        )

        return {
            "areas": areas,
            "doorways": doorways,
            "preprocess_s": float(self.preprocess_s),
        }
