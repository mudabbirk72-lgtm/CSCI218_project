from __future__ import annotations
from pathlib import Path
from grid import Grid, Scenario, SearchResult
from search import dijkstra_all
import numpy as np
import math


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
    settings = dict(METHODS[method])          # KeyError for an unknown label
    kind = settings.pop("planner")
    if kind == "flat":
        from search import FlatPlanner
        planner = FlatPlanner(**settings)
    elif kind == "two_tier":
        from two_tier import TwoTierPlanner
        planner = TwoTierPlanner(**settings)
    elif kind == "hpa":
        from hpa import HPAPlanner
        planner = HPAPlanner(**settings)
    else:
        raise ValueError(f"unknown planner type {kind!r} for method {method!r}")
    planner.name = method
    return planner


def make_map(kind: str, size: int, seed: int, **params) -> Grid:
    import generators
    if kind == "random":
        params.setdefault("density", 0.2)
        blocked = generators.random_map(size, params.pop("density"), seed, **params)
    elif kind == "rooms":
        blocked = generators.room_map(size, seed, **params)
    elif kind == "maze":
        blocked = generators.maze_map(size, seed, **params)
    else:
        raise ValueError(f"unknown map kind {kind!r}: use 'random', 'rooms' or 'maze'")
    return Grid(blocked)


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
    if math.isinf(cost):
        return math.inf
    if optimal_cost == 0:
        return 1.0 if cost == 0 else math.inf
    return cost / optimal_cost


def run_queries(planner, grid: Grid, scenarios: list[Scenario], repeats: int = 1) -> list[SearchResult]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    results = []
    for sc in scenarios:
        runs = [planner.query(sc.start, sc.goal) for _ in range(repeats)]
        runs.sort(key=lambda r: r.runtime_s)
        result = runs[len(runs) // 2]
        if result.success:
            assert grid.is_valid_path(result.path), (
                f"{planner.name}: query {sc.id} returned an illegal path")
            assert result.path[0] == sc.start and result.path[-1] == sc.goal, (
                f"{planner.name}: query {sc.id} path does not run from start to goal")
            assert math.isclose(result.cost, grid.path_cost(result.path), rel_tol=1e-6, abs_tol=1e-9), (
                f"{planner.name}: query {sc.id} reported cost {result.cost} "
                f"but the path costs {grid.path_cost(result.path)}")
        results.append(result)
    return results


def result_row(result: SearchResult, scenario: Scenario, planner, *, experiment: str,
               map_name: str, map_kind: str, map_size: int, map_seed: int) -> dict:
    stats = planner.stats()
    row = {
        "experiment": experiment,
        "method": planner.name,
        "map_name": map_name,
        "map_kind": map_kind,
        "map_size": map_size,
        "map_seed": map_seed,
        "query_id": scenario.id,
        "success": result.success,
        "cost": result.cost,
        "optimal_cost": scenario.optimal_cost,
        "suboptimality": suboptimality(result.cost, scenario.optimal_cost),
        "expanded": result.expanded,
        "generated": result.generated,
        "tier1_expanded": result.extra.get("tier1_expanded", 0),
        "runtime_s": result.runtime_s,
        "preprocess_s": stats.get("preprocess_s", 0.0),
        "areas": stats.get("areas", 0),
        "doorways": stats.get("doorways", 0),
    }
    return {key: row[key] for key in CSV_COLUMNS}


def run_experiment(config_path: str | Path, dry_run: bool = False) -> Path:
    raise NotImplementedError


def run_dynamic_experiment(grid: Grid, n_changes: int, seed: int,
                           partition: str = "watershed") -> list[dict]:
    raise NotImplementedError
