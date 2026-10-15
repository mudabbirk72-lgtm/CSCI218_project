from __future__ import annotations
from pathlib import Path
from grid import Grid, Scenario, SearchResult
from search import dijkstra_all
import numpy as np


METHODS: dict[str, dict] = {
    "Dijkstra":           {"planner": "flat", "algorithm": "dijkstra"},
    "A*":                 {"planner": "flat", "algorithm": "astar"},
    "Basic-fixed":        {"planner": "two_tier", "partition": "fixed", "refine": "nearest"},
    "Basic-watershed":    {"planner": "two_tier", "partition": "watershed", "refine": "nearest"},
    "Basic-louvain":      {"planner": "two_tier", "partition": "louvain", "refine": "nearest"},
    "Corridor-fixed":     {"planner": "two_tier", "partition": "fixed", "refine": "corridor"},
    "Corridor-watershed": {"planner": "two_tier", "partition": "watershed", "refine": "corridor"},
    "HPA-fixed":          {"planner": "hpa", "partition": "fixed"},
    "HPA-watershed":      {"planner": "hpa", "partition": "watershed"},
}


CSV_COLUMNS: list[str] = [
    "experiment", "method", "map_name", "map_kind", "map_size", "map_seed",
    "query_id", "success", "cost", "optimal_cost", "suboptimality",
    "expanded", "generated", "tier1_expanded", "runtime_s",
    "preprocess_s", "areas", "doorways",
]


def make_planner(method: str):
    raise NotImplementedError


def make_map(kind: str, size: int, seed: int, **params) -> Grid:
    raise NotImplementedError


def make_scenarios(grid: Grid, n: int, seed: int) -> list[Scenario]:
    if n < 1:
        raise ValueError("n must be at least 1")
    open_cells = np.argwhere(~grid.blocked)
    if len(open_cells) < 2:
        raise ValueError("the map needs at least 2 open cells")

    rng = np.random.default_rng(seed)
    scenarios: list[Scenario] = []
    attempts = 0
    while len(scenarios) < n:
        attempts += 1
        if attempts > 100 * n:
            raise ValueError("could not find enough reachable start/goal pairs")

        r, c = open_cells[rng.integers(len(open_cells))]
        start = (int(r), int(c))
        dist, _ = dijkstra_all(start, grid.neighbours)
        reachable = sorted((d, cell) for cell, d in dist.items() if cell != start)
        if not reachable:
            continue                       # isolated cell: try another start

        # One search gives distances to every cell, so each start supplies up to three
        # queries: one short, one medium and one long (thirds of the reachable cells).
        k = len(reachable)
        bands = [reachable[: max(1, k // 3)],
                 reachable[k // 3: max(k // 3 + 1, 2 * k // 3)],
                 reachable[2 * k // 3:]]
        for band in bands:
            if len(scenarios) == n:
                break
            if not band:
                continue
            d, goal = band[rng.integers(len(band))]
            scenarios.append(Scenario(start=start, goal=goal, optimal_cost=float(d),
                                      id=len(scenarios)))
    return scenarios


def suboptimality(cost: float, optimal_cost: float) -> float:
    raise NotImplementedError


def run_queries(planner, grid: Grid, scenarios: list[Scenario], repeats: int = 1) -> list[SearchResult]:
    raise NotImplementedError


def result_row(result: SearchResult, scenario: Scenario, planner, *, experiment: str,
               map_name: str, map_kind: str, map_size: int, map_seed: int) -> dict:
    raise NotImplementedError


def run_experiment(config_path: str | Path, dry_run: bool = False) -> Path:
    raise NotImplementedError


def run_dynamic_experiment(grid: Grid, n_changes: int, seed: int,
                           partition: str = "watershed") -> list[dict]:
    raise NotImplementedError
