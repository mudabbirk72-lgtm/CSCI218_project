from __future__ import annotations
from collections.abc import Callable
from grid import Cell, Grid, SearchResult
import time
from partition import get_partitioner
from area_graph import AreaGraph
from search import astar


class TwoTierPlanner:

    
    def __init__(self, partition: str = "watershed", refine: str = "nearest",
                 partition_params: dict | None = None, cost_mode: str = "centroid",
                 heuristic: str = "octile") -> None:

        # Check that the selected options are supported.
        if partition not in ("fixed", "watershed", "louvain"):
            raise ValueError("Unknown partition")
        if refine not in ("nearest", "corridor"):
            raise ValueError("Unknown refine mode")
        if cost_mode not in ("hop", "centroid"):
            raise ValueError("Unknown cost mode")
        if heuristic not in ("octile", "manhattan"):
            raise ValueError("Unknown heuristic")

        # Store the selected settings for the planner.
        self.partition = partition
        self.refine = refine

        # Copy the partition settings so the original dictionary is not modified.
        self.partition_params = (
            partition_params.copy() if partition_params is not None else {}
        )

        self.cost_mode = cost_mode
        self.heuristic = heuristic

        # Create a name based on the refinement and partition methods.
        self.name = (
            ("Basic-" if refine == "nearest" else "Corridor-") + partition
        )

        # These values will be set when preprocessing is completed.
        self.grid = None
        self.labels = None
        self.area_graph = None

        # Store the time taken to preprocess the grid.
        self.preprocess_s = 0.0

    def preprocess(self, grid: Grid) -> float:
        # Start measuring the preprocessing time.
        start_time = time.perf_counter()

        # Select the partitioning method and divide the grid into areas.
        partitioner = get_partitioner(self.partition)
        labels = partitioner(grid.blocked, **self.partition_params)

        # Build a graph showing the connections between the areas.
        area_graph = AreaGraph(grid.blocked, labels, self.cost_mode)

        # Save the grid, area labels, and area graph for later searches.
        self.grid = grid
        self.labels = labels
        self.area_graph = area_graph

        # Calculate and save the total preprocessing time.
        self.preprocess_s = time.perf_counter() - start_time

        # Return the time taken to preprocess the grid.
        return self.preprocess_s



        


    
    def find_route(self, start: Cell, goal: Cell) -> SearchResult:
        # Find a route between the start and goal areas using A*

        if self.grid is None or self.labels is None or self.area_graph is None:
            raise RuntimeError("preprocess must be called before find_route")

        if not self.grid.is_free(start) or not self.grid.is_free(goal):
            raise ValueError("start and goal must be open cells inside the grid")

        # Find the areas containing the start and goal cells.
        start_area = int(self.labels[start[0], start[1]])
        goal_area = int(self.labels[goal[0], goal[1]])

        # Use A* to find a route through the area graph.
        result = astar(
            start=start_area,
            goals={goal_area},
            neighbours=self.area_graph.neighbours,
            heuristic=self.area_graph.heuristic(goal_area),
        )

        return result


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
