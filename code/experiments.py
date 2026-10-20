from __future__ import annotations
from pathlib import Path
from grid import Grid, Scenario, SearchResult
from search import dijkstra_all
from pathlib import Path
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
    "HPA-louvain":        {"planner": "hpa", "partition": "louvain"},
    "HPA-watershed-nocap":   {"planner": "hpa", "partition": "watershed",
                              "partition_params": {"max_area_size": None}},
    "HPA-watershed-cap1024": {"planner": "hpa", "partition": "watershed",
                              "partition_params": {"max_area_size": 1024}},
    "HPA-watershed-cap256":  {"planner": "hpa", "partition": "watershed",
                              "partition_params": {"max_area_size": 256}},
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


def _load_maps(config: dict, config_dir: Path):
    """Yield one dict per map in the config: grid, name, kind, size, seed, scenarios."""
    entries = config.get("maps")
    if not entries:
        raise ValueError("the config has no 'maps' list")
    for entry in entries:
        entry = dict(entry)
        kind = entry.pop("kind")
        if kind == "file":
            import map_io
            pad = entry.get("pad_border", True)
            for p in entry["paths"]:
                path = Path(p)
                if not path.is_absolute() and not path.exists():
                    path = config_dir / path            # also allow paths relative to the config
                if not path.exists():
                    raise FileNotFoundError(f"map file not found: {p}")
                grid = map_io.read_map(path, pad_border=pad)
                scen_path = path.with_name(path.name + ".scen")
                scenarios = (map_io.read_scenarios(scen_path, offset=1 if pad else 0)
                             if scen_path.exists() else None)
                yield {"grid": grid, "name": path.stem, "kind": entry.get("label", "benchmark"),
                       "size": max(grid.height, grid.width) - (2 if pad else 0),
                       "seed": -1, "scenarios": scenarios}
        else:
            params = entry.get("params", {})
            for size in entry["sizes"]:
                for seed in entry["seeds"]:
                    grid = make_map(kind, size, seed, **params)
                    yield {"grid": grid, "name": f"{kind}_{size}_s{seed}", "kind": kind,
                           "size": size, "seed": seed, "scenarios": None}


def _pick_evenly(scenarios: list[Scenario], n: int) -> list[Scenario]:
    """n scenarios spread through the list (a .scen file runs from short to long)."""
    if len(scenarios) <= n:
        return list(scenarios)
    step = len(scenarios) / n
    return [scenarios[int(i * step)] for i in range(n)]


def run_experiment(config_path: str | Path, dry_run: bool = False) -> Path:
    """Run a whole experiment from its YAML config and save the CSV."""
    import csv
    import time

    import yaml

    config_path = Path(config_path)
    with config_path.open() as f:                      # FileNotFoundError if missing
        config = yaml.safe_load(f)

    name = config["name"]
    out = Path(config["output"])
    n_queries = 5 if dry_run else int(config.get("queries_per_map", 50))
    repeats = int(config.get("repeats", 1))
    methods = list(config["methods"])
    unknown = [m for m in methods if m not in METHODS]
    if unknown:                                        # fail before any long work starts
        raise KeyError(f"unknown method(s) {unknown}; choose from {list(METHODS)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for m in _load_maps(config, config_path.parent):
            grid = m["grid"]
            if m["scenarios"]:
                scenarios = _pick_evenly(m["scenarios"], n_queries)
            else:
                seed = m["seed"] if m["seed"] >= 0 else 0
                scenarios = make_scenarios(grid, n_queries, seed)
            print(f"{m['name']} ({grid.height}x{grid.width}, {len(scenarios)} queries)")
            for method in methods:
                planner = make_planner(method)
                planner.preprocess(grid)
                results = run_queries(planner, grid, scenarios, repeats)
                for result, sc in zip(results, scenarios):
                    writer.writerow(result_row(result, sc, planner, experiment=name,
                                               map_name=m["name"], map_kind=m["kind"],
                                               map_size=m["size"], map_seed=m["seed"]))
                f.flush()                              # a crash later loses nothing done
                ok = sum(r.success for r in results)
                worst = max((suboptimality(r.cost, s.optimal_cost)
                             for r, s in zip(results, scenarios) if r.success), default=math.nan)
                print(f"  {method:<20} {ok}/{len(results)} solved, worst suboptimality {worst:.3f}")
            if dry_run:
                break
    print(f"Done in {time.perf_counter() - started:.1f}s. Results: {out}")
    return out


def run_dynamic_experiment(grid: Grid, n_changes: int, seed: int,
                           partition: str = "watershed") -> list[dict]:
    """Measure local repair against a full rebuild (the dynamic-map experiment)."""
    import time
    from hpa import DoorwayGraph, HPAPlanner

    grid = grid.copy()                               # never change the caller's map
    rng = np.random.default_rng(seed)
    planner = HPAPlanner(partition=partition)
    planner.preprocess(grid)

    trials = []
    for _ in range(n_changes):
        # 1. pick a random doorway cell (a permanent node of the doorway graph)
        doorways = sorted({cell for node, (_, cell) in planner.graph.nodes.items()
                           if node not in planner.graph.temporary})
        cell = doorways[rng.integers(len(doorways))]

        # 2. block it: local repair vs full rebuild
        repair_s = planner.update(cell, True)
        info = dict(planner.last_update)

        fresh = HPAPlanner(partition=partition)
        t = time.perf_counter()
        fresh.preprocess(grid)
        rebuild_s = time.perf_counter() - t

        # 3. the repaired graph must answer like a graph built from scratch with the
        #    same labels (the spec's definition of a correct repair)
        reference = HPAPlanner(partition=partition)
        reference.grid, reference.labels = grid, planner.labels.copy()
        reference.graph = DoorwayGraph(grid, reference.labels, planner.max_gap)

        free = np.argwhere(~grid.blocked)
        answers_match = True
        for _ in range(10):
            a, b = free[rng.choice(len(free), 2, replace=False)]
            start, goal = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
            got, want = planner.query(start, goal), reference.query(start, goal)
            if got.success != want.success or (
                    got.success and not math.isclose(got.cost, want.cost, rel_tol=1e-9)):
                answers_match = False

        trials.append({"cell": cell, "repair_s": float(repair_s), "rebuild_s": rebuild_s,
                       "areas_rebuilt": int(info.get("areas_rebuilt", 0)),
                       "paths_recomputed": int(info.get("paths_recomputed", 0)),
                       "answers_match": answers_match})

        # 4. open the cell again so the next trial starts from (almost) the same map
        planner.update(cell, False)
    return trials


DYNAMIC_COLUMNS: list[str] = [
    "experiment", "map_name", "map_kind", "map_size", "map_seed", "partition", "trial",
    "cell_row", "cell_col", "repair_s", "rebuild_s", "speedup", "areas_rebuilt",
    "paths_recomputed", "answers_match",
]
"""Columns of the dynamic experiment's CSV (one row per block/unblock trial)."""


def run_dynamic_from_config(config_path: str | Path, dry_run: bool = False) -> Path:
    """Run the dynamic-map experiment from its YAML config and save the CSV."""
    import csv
    import time

    import yaml

    config_path = Path(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    name = config["name"]
    out = Path(config["output"])
    changes = 2 if dry_run else int(config.get("changes", 20))
    seed = int(config.get("seed", 0))
    partitions = config.get("partitions", {})          # map kind -> partition method

    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DYNAMIC_COLUMNS)
        writer.writeheader()
        for m in _load_maps(config, config_path.parent):
            partition = partitions.get(m["kind"], "watershed")
            trials = run_dynamic_experiment(m["grid"], changes, seed, partition)
            for i, t in enumerate(trials):
                writer.writerow({
                    "experiment": name, "map_name": m["name"], "map_kind": m["kind"],
                    "map_size": m["size"], "map_seed": m["seed"], "partition": partition,
                    "trial": i, "cell_row": t["cell"][0], "cell_col": t["cell"][1],
                    "repair_s": t["repair_s"], "rebuild_s": t["rebuild_s"],
                    "speedup": t["rebuild_s"] / max(t["repair_s"], 1e-9),
                    "areas_rebuilt": t["areas_rebuilt"],
                    "paths_recomputed": t["paths_recomputed"],
                    "answers_match": t["answers_match"],
                })
            f.flush()
            ok = sum(t["answers_match"] for t in trials)
            rep = sorted(t["repair_s"] for t in trials)[len(trials) // 2]
            reb = sorted(t["rebuild_s"] for t in trials)[len(trials) // 2]
            print(f"{m['name']:<16} {partition:<10} median repair {1000 * rep:7.1f} ms   "
                  f"median rebuild {1000 * reb:8.1f} ms   ({reb / max(rep, 1e-9):5.1f}x faster)   "
                  f"answers match {ok}/{len(trials)}", flush=True)
            if dry_run:
                break
    print(f"Done in {time.perf_counter() - started:.1f}s. Results: {out}")
    return out


if __name__ == "__main__":
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description="Run an experiment from a YAML config.")
    ap.add_argument("config", help="e.g. configs/pilot.yaml or configs/dynamic.yaml")
    ap.add_argument("--dry-run", action="store_true", help="first map only, a few queries/changes")
    args = ap.parse_args()
    with open(args.config) as f:
        experiment_type = (yaml.safe_load(f) or {}).get("type", "queries")
    if experiment_type == "dynamic":
        run_dynamic_from_config(args.config, dry_run=args.dry_run)
    else:
        run_experiment(args.config, dry_run=args.dry_run)
