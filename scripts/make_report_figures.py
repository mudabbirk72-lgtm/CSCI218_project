"""Make the report figures that make_all() does not: map pictures, the speed-vs-quality
trade-off, and the results tables.

Usage (from the project folder, after the experiments have been run):
    python scripts/make_report_figures.py

Writes to results/figures/report/:
    F1_map_types.png          the four kinds of map we test on
    F2_pipeline.png           the method step by step on one room map
    F3_partitions.png         fixed blocks, watershed and Louvain on the same map
    F4_area_cap.png           watershed areas without and with the area-size cap
    F5_search_effort.png      cells explored by A* vs HPA on the same query
    F8_tradeoff.png           speed-up over A* against path quality, at 1024
    T2_main_results.csv/.md   main results table (every map kind, size 1024)
    T3_partitions.csv/.md     partition comparison table (room maps)
    T5_dynamic.csv/.md        dynamic repair table
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import experiments  # noqa: E402
import map_io  # noqa: E402
from partition import distance_map, get_partitioner  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "figures" / "report"
STREET = ROOT / "maps" / "benchmark" / "street"

# one colour per method, used in every figure of this script
COLOURS = {
    "A*": "#222222", "Basic-fixed": "#1f77b4", "Basic-watershed": "#6baed6",
    "Corridor-fixed": "#2ca02c", "Corridor-watershed": "#98df8a",
    "HPA-fixed": "#d62728", "HPA-watershed": "#ff9896",
}
WALL, OPEN = np.array([40, 42, 50]) / 255, np.array([236, 236, 232]) / 255
A_STAR_CELLS, HPA_CELLS, PATH = [0.45, 0.65, 1.0], [1.0, 0.6, 0.2], "#d6006e"


# ----------------------------------------------------------------- drawing helpers


def area_colour(area: int) -> np.ndarray:
    """A fixed, well-spread colour per area id (same idea as the app)."""
    r, g, b = colorsys.hsv_to_rgb((area * 0.6180339887) % 1, (.5, .68, .58)[area % 3],
                                  (.95, .82, .9, .74)[area // 3 % 4])
    return np.array([r, g, b])


def picture(blocked: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """RGB image, one pixel per cell: walls dark, open cells light or coloured by area."""
    img = np.where(blocked[..., None], WALL, OPEN).astype(float)
    if labels is not None:
        table = np.array([area_colour(i) for i in range(int(labels.max()) + 1)])
        ok = labels >= 0
        img[ok] = table[labels[ok]]
    return img


def show(ax, img, title):
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def draw_path(ax, path):
    if path:
        ax.plot([c[1] for c in path], [c[0] for c in path], color=PATH, lw=1.8)


def mark_ends(ax, start, goal):
    ax.scatter([start[1]], [start[0]], s=45, c="#2ecc71", edgecolors="black", zorder=5)
    ax.scatter([goal[1]], [goal[0]], s=45, c="#e74c3c", edgecolors="black", zorder=5)


def far_pair(grid) -> tuple[tuple[int, int], tuple[int, int]]:
    """An open cell near the top-left corner and one near the bottom-right corner."""
    free = np.argwhere(~grid.blocked)
    a = free[np.argmin(free.sum(axis=1))]
    b = free[np.argmax(free.sum(axis=1))]
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))
    return path


def street(name: str):
    return map_io.read_map(STREET / f"{name}.map", pad_border=True)


# ----------------------------------------------------------------- map figures


def f1_map_types():
    maps = [("Rooms (128, seed 2)", experiments.make_map("rooms", 128, 2)),
            ("Random obstacles (128, seed 2)", experiments.make_map("random", 128, 2)),
            ("Maze (128, seed 2)", experiments.make_map("maze", 128, 2)),
            ("Street map: Berlin (512)", street("Berlin_0_512"))]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (title, grid) in zip(axes, maps):
        show(ax, picture(grid.blocked), title)
    return save(fig, "F1_map_types.png")


def f2_pipeline():
    grid = experiments.make_map("rooms", 128, 2)
    planner = experiments.make_planner("HPA-watershed")
    planner.preprocess(grid)
    start, goal = far_pair(grid)
    explored: list = []
    result = planner.query(start, goal, on_expand=explored.append)
    labels = planner.labels

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    show(axes[0], picture(grid.blocked), "(a) Map")

    d = distance_map(grid.blocked).astype(float)
    axes[1].imshow(np.where(grid.blocked, np.nan, d), cmap="viridis", interpolation="nearest")
    axes[1].set_title("(b) Distance to the nearest wall", fontsize=10)
    axes[1].axis("off")

    show(axes[2], picture(grid.blocked, labels), f"(c) Watershed areas ({planner.stats()['areas']})")

    g = planner.graph
    show(axes[3], picture(grid.blocked, labels), f"(d) Doorway graph ({len(g.nodes)} doorways)")
    for u, nbrs in g.adj.items():
        for v in nbrs:
            if u < v:
                (r1, c1), (r2, c2) = g.nodes[u][1], g.nodes[v][1]
                axes[3].plot([c1, c2], [r1, r2], color="black", lw=0.4, alpha=0.6)
    cells = np.array([cell for _, cell in g.nodes.values()])
    axes[3].scatter(cells[:, 1], cells[:, 0], s=6, c="black", zorder=3)

    img = picture(grid.blocked, labels) * 0.35 + 0.65                  # faded areas
    if explored:
        e = np.array(explored)
        img[e[:, 0], e[:, 1]] = HPA_CELLS
    show(axes[4], img, f"(e) Final path and explored cells ({result.expanded} expanded)")
    draw_path(axes[4], result.path)
    for ax in axes[2:]:
        mark_ends(ax, start, goal)
    return save(fig, "F2_pipeline.png")


def f3_partitions():
    grid = experiments.make_map("rooms", 256, 2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (name, title) in zip(axes, [("fixed", "Fixed 16x16 blocks"),
                                        ("watershed", "Watershed (cap 512)"),
                                        ("louvain", "Louvain communities")]):
        labels = get_partitioner(name)(grid.blocked)
        show(ax, picture(grid.blocked, labels), f"{title}: {int(labels.max()) + 1} areas")
    return save(fig, "F3_partitions.png")


def f4_area_cap():
    grid = experiments.make_map("rooms", 256, 2)
    ws = get_partitioner("watershed")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))
    for ax, (cap, title) in zip(axes, [(None, "Watershed, no cap"),
                                       (512, "Watershed, cap 512 cells")]):
        labels = ws(grid.blocked, max_area_size=cap)
        sizes = np.bincount(labels[labels >= 0])
        show(ax, picture(grid.blocked, labels),
             f"{title}: {len(sizes)} areas, largest {sizes.max()} cells")
    return save(fig, "F4_area_cap.png")


def f5_search_effort():
    cases = [("Rooms 256", experiments.make_map("rooms", 256, 2), "HPA-watershed"),
             ("Berlin 512", street("Berlin_0_512"), "HPA-fixed")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for row, (name, grid, method) in enumerate(cases):
        start, goal = far_pair(grid)
        for col, label in enumerate(["A*", method]):
            planner = experiments.make_planner(label)
            planner.preprocess(grid)
            explored: list = []
            result = planner.query(start, goal, on_expand=explored.append)
            img = picture(grid.blocked)
            if explored:
                e = np.array(explored)
                img[e[:, 0], e[:, 1]] = A_STAR_CELLS if label == "A*" else HPA_CELLS
            show(axes[row, col], img,
                 f"{name}: {label}\n{result.expanded:,} cells expanded, "
                 f"{result.runtime_s * 1e3:.1f} ms, path cost {result.cost:.1f}")
            draw_path(axes[row, col], result.path)
            mark_ends(axes[row, col], start, goal)
    return save(fig, "F5_search_effort.png")


# ----------------------------------------------------------------- results figure and tables


def per_method(df: pd.DataFrame) -> pd.DataFrame:
    """Per method on one set of maps: speed-up, cells vs A*, path quality, preparation."""
    ok = df[df["success"] == True]
    g = ok.groupby("method")
    t = pd.DataFrame({
        "query_ms": g["runtime_s"].mean() * 1000,
        "expanded": g["expanded"].mean(),
        "mean_pct_longer": (g["suboptimality"].mean() - 1) * 100,
        "worst_pct_longer": (g["suboptimality"].max() - 1) * 100,
        "preprocess_s": ok.drop_duplicates(["map_name", "method"]).groupby("method")["preprocess_s"].mean(),
    })
    a = t.loc["A*"]
    t["speedup_vs_A*"] = a["query_ms"] / t["query_ms"]
    t["expanded_vs_A*"] = t["expanded"] / a["expanded"]
    saving_s = (a["query_ms"] - t["query_ms"]) / 1000
    t["breakeven_queries"] = (t["preprocess_s"] / saving_s.where(saving_s > 0)).fillna(np.inf)
    t.loc["A*", "breakeven_queries"] = 0
    return t


def f8_tradeoff():
    scaling = pd.read_csv(RESULTS / "scaling.csv")
    street_df = pd.read_csv(RESULTS / "street.csv")
    sets = [("Rooms 1024", scaling[(scaling.map_kind == "rooms") & (scaling.map_size == 1024)]),
            ("Street maps 1024", street_df[street_df.map_size == 1024])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (title, df) in zip(axes, sets):
        t = per_method(df).drop(index="A*")
        for method, row in t.iterrows():
            ax.scatter(row["speedup_vs_A*"], row["mean_pct_longer"], s=90,
                       color=COLOURS.get(method, "grey"), edgecolors="black", zorder=3)
            ax.annotate(method, (row["speedup_vs_A*"], row["mean_pct_longer"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Speed-up over A* (log scale, higher is better)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Mean path length: % longer than optimal (lower is better)")
    fig.suptitle("Speed against path quality: the best methods sit bottom-right")
    return save(fig, "F8_tradeoff.png")


def markdown(df: pd.DataFrame) -> str:
    """A simple Markdown table (no extra packages needed)."""
    cols = list(df.columns)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(markdown(df))
    print((OUT / f"{name}.md").relative_to(ROOT))


def t2_main_results():
    scaling = pd.read_csv(RESULTS / "scaling.csv")
    street_df = pd.read_csv(RESULTS / "street.csv")
    sets = [(f"{kind} 1024", scaling[(scaling.map_kind == kind) & (scaling.map_size == 1024)])
            for kind in ("rooms", "random", "maze")]
    sets.append(("street 1024", street_df[street_df.map_size == 1024]))
    blocks = [per_method(df).assign(maps=name) for name, df in sets]
    t = pd.concat(blocks).reset_index().rename(columns={"index": "method"})
    cols = ["maps", "method", "speedup_vs_A*", "expanded_vs_A*", "mean_pct_longer",
            "worst_pct_longer", "preprocess_s", "breakeven_queries"]
    write_table(t[cols].round(3), "T2_main_results")


def t3_partitions():
    p = pd.read_csv(RESULTS / "partitions.csv")
    blocks = []
    for size in (256, 512):
        df = p[(p.map_kind == "rooms") & (p.map_size == size)]
        t = per_method(df)
        t["areas"] = df.groupby("method")["areas"].mean()
        blocks.append(t.drop(index="A*").assign(maps=f"rooms {size}"))
    t = pd.concat(blocks).reset_index().rename(columns={"index": "method"})
    cols = ["maps", "method", "areas", "preprocess_s", "speedup_vs_A*",
            "mean_pct_longer", "worst_pct_longer"]
    write_table(t[cols].round(3), "T3_partitions")


def t5_dynamic():
    d = pd.read_csv(RESULTS / "dynamic.csv")
    g = d.groupby("map_name", sort=False)
    t = pd.DataFrame({
        "partition": g["partition"].first(),
        "trials": g.size(),
        "median_repair_ms": g["repair_s"].median() * 1000,
        "median_rebuild_ms": g["rebuild_s"].median() * 1000,
        "speedup": g["rebuild_s"].median() / g["repair_s"].median(),
        "median_areas_rebuilt": g["areas_rebuilt"].median(),
        "answers_match_pct": g["answers_match"].mean() * 100,
    }).reset_index()
    write_table(t.round(2), "T5_dynamic")


def main() -> int:
    for make in (f1_map_types, f2_pipeline, f3_partitions, f4_area_cap, f5_search_effort,
                 f8_tradeoff, t2_main_results, t3_partitions, t5_dynamic):
        make()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())