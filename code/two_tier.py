from __future__ import annotations
from collections.abc import Callable
from grid import Cell, Grid, SearchResult
import time
import numpy as np
from partition import get_partitioner
from area_graph import AreaGraph
from search import astar, octile, manhattan


class _InArea:
    """Goal test for A*: ``cell in _InArea(labels, a)`` is True for every cell of area a."""

    def __init__(self, labels, area: int) -> None:
        self.labels, self.area = labels, area

    def __contains__(self, cell) -> bool:
        return int(self.labels[cell]) == self.area


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
        # Start measuring the time used by this refinement.
        start_time = time.perf_counter()

        # Return failure if the area route is empty.
        if not route:
            return SearchResult(
                path=None,
                cost=float("inf"),
                expanded=0,
                generated=0,
                runtime_s=time.perf_counter() - start_time,
                success=False,
                extra={"legs": []},
            )

        # Initialise the complete path and search statistics.
        current = start
        full_path = [start]
        total_cost = 0.0
        total_expanded = 0
        total_generated = 0
        legs = []

        # Select the heuristic configured for this planner.
        distance = octile if self.heuristic == "octile" else manhattan

        # Visit each area after the starting area in the route.
        for next_area in route[1:]:
            # Return failure if the next area does not exist.
            if next_area not in self.area_graph.bboxes:
                return SearchResult(
                    path=None,
                    cost=float("inf"),
                    expanded=total_expanded,
                    generated=total_generated,
                    runtime_s=time.perf_counter() - start_time,
                    success=False,
                    extra={"legs": legs},
                )

            # Any cell whose label is next_area is a target. Checking the label is
            # much faster than listing every cell of the area (which scans the map).
            goal_cells = _InArea(self.labels, next_area)

            # Bounding box of the target area, worked out once in preprocess.
            min_row, min_col, max_row, max_col = self.area_graph.bboxes[next_area]

            # Estimate the distance from a cell to the target area's box.
            def heuristic(cell: Cell) -> float:
                closest = (
                    min(max(cell[0], min_row), max_row),
                    min(max(cell[1], min_col), max_col),
                )
                return distance(cell, closest)

            # Allow movement through the current area and the next area.
            current_area = int(self.labels[current])
            neighbours = self.grid.restricted_neighbours(
                self.labels, {current_area, next_area}
            )

            # Use A* to find the nearest reachable cell in the next area.
            result = astar(
                current,
                goal_cells,
                neighbours,
                heuristic,
                on_expand,
            )

            # Add this search's statistics to the totals.
            total_expanded += result.expanded
            total_generated += result.generated

            # Stop if the next area cannot be reached.
            if not result.success:
                return SearchResult(
                    path=None,
                    cost=float("inf"),
                    expanded=total_expanded,
                    generated=total_generated,
                    runtime_s=time.perf_counter() - start_time,
                    success=False,
                    extra={"legs": legs},
                )

            # Append the path without repeating its starting cell.
            full_path.extend(result.path[1:])
            total_cost += result.cost

            # Save information about this part of the journey.
            legs.append({
                "from": current,
                "to": result.path[-1],
                "cost": result.cost,
                "expanded": result.expanded,
            })

            # Continue from the cell reached in the next area.
            current = result.path[-1]

        # Search for the actual goal within the final area.
        last_area = route[-1]
        neighbours = self.grid.restricted_neighbours(
            self.labels, {last_area}
        )

        final_result = astar(
            current,
            {goal},
            neighbours,
            lambda cell: distance(cell, goal),
            on_expand,
        )

        # Include the final search's statistics.
        total_expanded += final_result.expanded
        total_generated += final_result.generated

        # Return failure if the actual goal cannot be reached.
        if not final_result.success:
            return SearchResult(
                path=None,
                cost=float("inf"),
                expanded=total_expanded,
                generated=total_generated,
                runtime_s=time.perf_counter() - start_time,
                success=False,
                extra={"legs": legs},
            )

        # Add the final section of the path and its cost.
        full_path.extend(final_result.path[1:])
        total_cost += final_result.cost

        # Record the final leg.
        legs.append({
            "from": current,
            "to": goal,
            "cost": final_result.cost,
            "expanded": final_result.expanded,
        })

        # Return the completed path and its statistics.
        return SearchResult(
            path=full_path,
            cost=total_cost,
            expanded=total_expanded,
            generated=total_generated,
            runtime_s=time.perf_counter() - start_time,
            success=True,
            extra={"legs": legs},
        )

    def refine_corridor(self, route: list[int], start: Cell, goal: Cell,
                        on_expand: Callable[[Cell], None] | None = None) -> SearchResult:
        # Start measuring the time used by corridor refinement.
        start_time = time.perf_counter()

        # Return failure if the area route is empty.
        if not route:
            return SearchResult(
                path=None,
                cost=float("inf"),
                expanded=0,
                generated=0,
                runtime_s=time.perf_counter() - start_time,
                success=False,
                extra={"route": route},
            )

        # Allow A* to move only through areas in the tier-1 route.
        allowed_areas = set(route)
        neighbours = self.grid.restricted_neighbours(
            self.labels, allowed_areas
        )

        # Select the heuristic configured for this planner.
        distance = octile if self.heuristic == "octile" else manhattan

        # Estimate the remaining distance to the actual goal.
        heuristic = lambda cell: distance(cell, goal)

        # Search directly from the start to the goal inside the corridor.
        result = astar(
            start,
            {goal},
            neighbours,
            heuristic,
            on_expand,
        )

        # Return the result with the corridor route recorded.
        return SearchResult(
            path=result.path,
            cost=result.cost,
            expanded=result.expanded,
            generated=result.generated,
            runtime_s=time.perf_counter() - start_time,
            success=result.success,
            extra={
                **(result.extra or {}),
                "route": route,
            },
        )

    def query(self, start: Cell, goal: Cell,
              on_expand: Callable[[Cell], None] | None = None) -> SearchResult:
        # Start measuring the total query time.
        query_start = time.perf_counter()

        # Make sure preprocessing has been completed first.
        if self.grid is None or self.labels is None or self.area_graph is None:
            raise RuntimeError("preprocess must be called before query")

        # Check that the start and goal are valid open cells.
        if not self.grid.is_free(start):
            raise ValueError("start must be an open cell inside the grid")

        if not self.grid.is_free(goal):
            raise ValueError("goal must be an open cell inside the grid")

        # Find an area-level route using the first search level.
        tier1 = self.find_route(start, goal)

        # Return failure if the area-level search finds no route.
        if not tier1.success:
            return SearchResult(
                path=None,
                cost=float("inf"),
                expanded=tier1.expanded,
                generated=tier1.generated,
                runtime_s=time.perf_counter() - query_start,
                success=False,
                extra={
                    "route": tier1.path,
                    "tier1_expanded": tier1.expanded,
                },
            )

        # Refine the area route into a cell-by-cell path.
        if self.refine == "nearest":
            tier2 = self.refine_nearest(
                tier1.path, start, goal, on_expand
            )
        else:
            # Use the corridor method when that option is selected.
            tier2 = self.refine_corridor(
                tier1.path, start, goal, on_expand
            )

        # Keep the refinement details and add the area route details.
        extra = dict(tier2.extra)
        extra["route"] = tier1.path
        extra["tier1_expanded"] = tier1.expanded

        # Combine the statistics from both search levels.
        return SearchResult(
            path=tier2.path,
            cost=tier2.cost,
            expanded=tier1.expanded + tier2.expanded,
            generated=tier1.generated + tier2.generated,
            runtime_s=time.perf_counter() - query_start,
            success=tier2.success,
            extra=extra,
        )

    def update(self, cell: Cell, blocked: bool) -> float:
        # Start measuring the complete update time.
        start_time = time.perf_counter()

        # Make sure preprocessing has already assigned the grid.
        if self.grid is None:
            raise RuntimeError("Call preprocess() before update()")

        # Change the requested cell's blocked/open state.
        if blocked:
            self.grid.block(cell)
        else:
            self.grid.unblock(cell)

        # Rebuild the partition and area graph from the changed grid.
        self.preprocess(self.grid)

        # Return the total time taken for the update and rebuild.
        return time.perf_counter() - start_time

    def stats(self) -> dict:
        # Return empty statistics if preprocessing has not been completed.
        if self.labels is None:
            return {
                "areas": 0,
                "doorways": 0,
                "preprocess_s": self.preprocess_s,
            }

        # Count the distinct valid area labels.
        valid_labels = self.labels[self.labels >= 0]
        area_count = int(np.unique(valid_labels).size)

        # Return the planner's preprocessing statistics.
        return {
            "areas": area_count,
            "doorways": 0,
            "preprocess_s": self.preprocess_s,
        }
