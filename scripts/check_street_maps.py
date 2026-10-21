"""Check our A* against the optimal lengths published with the Moving AI street maps.

Usage (from the project folder):
    python scripts/check_street_maps.py
    python scripts/check_street_maps.py --per-map 20 --folder maps/benchmark/street

For every <name>.map in the folder that has a matching <name>.map.scen, it loads the map
(with a wall border added), picks queries spread evenly through the .scen file (short to
long), runs A* with the octile heuristic, and compares our cost with the published one.
Results go to results/street_check.csv; a summary per map is printed.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import map_io  # noqa: E402
import search  # noqa: E402

COLUMNS = ["map", "size", "query_id", "start_row", "start_col", "goal_row", "goal_col",
           "published", "ours", "difference", "match", "expanded", "runtime_s"]


def check_map(map_path: Path, scen_path: Path, per_map: int, tol: float) -> list[dict]:
    grid = map_io.read_map(map_path, pad_border=True)
    scenarios = map_io.read_scenarios(scen_path, offset=1)
    if not scenarios:
        return []
    step = max(1, len(scenarios) // per_map)
    picked = scenarios[::step][:per_map]                 # spread from short to long
    rows = []
    for sc in picked:
        r = search.astar(sc.start, {sc.goal}, grid.neighbours,
                         lambda c, g=sc.goal: search.octile(c, g))
        diff = r.cost - sc.optimal_cost
        rows.append({
            "map": map_path.stem, "size": max(grid.height, grid.width) - 2,
            "query_id": sc.id, "start_row": sc.start[0] - 1, "start_col": sc.start[1] - 1,
            "goal_row": sc.goal[0] - 1, "goal_col": sc.goal[1] - 1,
            "published": sc.optimal_cost, "ours": r.cost, "difference": diff,
            "match": r.success and math.isclose(r.cost, sc.optimal_cost, abs_tol=tol),
            "expanded": r.expanded, "runtime_s": r.runtime_s,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(ROOT / "maps" / "benchmark" / "street"))
    ap.add_argument("--per-map", type=int, default=20, help="queries checked per map")
    ap.add_argument("--tol", type=float, default=1e-3, help="allowed cost difference")
    ap.add_argument("--out", default=str(ROOT / "results" / "street_check.csv"))
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    pairs = [(m, m.with_name(m.name + ".scen")) for m in sorted(folder.glob("*.map"))]
    pairs = [(m, s) for m, s in pairs if s.exists()]
    if not pairs:
        print(f"No <name>.map + <name>.map.scen pairs found in {folder}")
        return 1

    all_rows = []
    for map_path, scen_path in pairs:
        rows = check_map(map_path, scen_path, args.per_map, args.tol)
        ok = sum(r["match"] for r in rows)
        print(f"{map_path.stem:<20} {ok:>3}/{len(rows)} match")
        all_rows += rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    bad = [r for r in all_rows if not r["match"]]
    print(f"\n{len(all_rows) - len(bad)}/{len(all_rows)} queries match. Results: {out}")
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())