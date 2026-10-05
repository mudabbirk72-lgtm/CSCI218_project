

from __future__ import annotations

from pathlib import Path

from hierpath.grid import Grid, Scenario, SearchResult

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
    raise NotImplementedError


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
