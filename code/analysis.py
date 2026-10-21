from __future__ import annotations
from pathlib import Path
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_COLUMNS: list[str] = [
    "experiment", "method", "map_name", "map_kind", "map_size", "map_seed",
    "query_id", "success", "cost", "optimal_cost", "suboptimality",
    "expanded", "generated", "tier1_expanded", "runtime_s",
    "preprocess_s", "areas", "doorways",
]


def load_results(paths: list[str | Path]) -> pd.DataFrame:
    dataframes = []
    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        df = pd.read_csv(path_obj)
        
        missing = set(CSV_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"File {path} lacks expected columns: {missing}")
            
        dataframes.append(df)
        
    return pd.concat(dataframes, ignore_index=True)

def confidence_interval(values, level: float = 0.95, n_boot: int = 10_000,
                        seed: int = 0) -> tuple[float, float]:
    data = np.asarray(values)
    data = data[~np.isnan(data)]
    
    if len(data) < 2:
        return (np.nan, np.nan)
        
    res = stats.bootstrap(
        (data,), np.mean, confidence_level=level, 
        n_resamples=n_boot, random_state=seed, method='percentile'
    )
    return (res.confidence_interval.low, res.confidence_interval.high)


def summarize(df: pd.DataFrame, metric: str, by: list[str] | None = None) -> pd.DataFrame:
    if by is None:
        by = ["method", "map_kind", "map_size"]
        
   
    success_df = df[df["success"] == True]
    failed_counts = df[df["success"] == False].groupby(by).size().rename("failed")
    
    
    grouped = success_df.groupby(by)[metric]
    summary = grouped.agg(
        n='count',
        mean='mean',
        sd='std',
        median='median'
    )
    
    
    ci_lows = []
    ci_highs = []
    for _, group_data in grouped:
        low, high = confidence_interval(group_data)
        ci_lows.append(low)
        ci_highs.append(high)
        
    summary['ci_low'] = ci_lows
    summary['ci_high'] = ci_highs
    
    
    summary = summary.join(failed_counts, how='left')
    summary['failed'] = summary['failed'].fillna(0).astype(int)
    
    return summary.reset_index()


def compare(df: pd.DataFrame, method_a: str, method_b: str, metric: str) -> dict:
    df_a = df[df["method"] == method_a][["map_name", "query_id", metric]]
    df_b = df[df["method"] == method_b][["map_name", "query_id", metric]]
    
    merged = pd.merge(df_a, df_b, on=["map_name", "query_id"], suffixes=('_a', '_b')).dropna()
    
    if len(merged) == 0:
        raise ValueError(f"Methods {method_a} and {method_b} share no queries.")
        
    vals_a = merged[f"{metric}_a"].values
    vals_b = merged[f"{metric}_b"].values
    diff = vals_a - vals_b
    
    
    non_zero_diffs = diff[diff != 0]
    if len(non_zero_diffs) == 0:
        p_val = 1.0
    else:
        res = stats.wilcoxon(vals_a, vals_b)
        p_val = res.pvalue
        
    return {
        "n": len(merged),
        "median_a": float(np.median(vals_a)),
        "median_b": float(np.median(vals_b)),
        "median_diff": float(np.median(diff)),
        "p_value": float(p_val),
        "a_lower_fraction": float(np.mean(vals_a < vals_b))
    }


def breakeven_queries(df: pd.DataFrame, method: str, baseline: str = "A*") -> pd.DataFrame:
    """Calculate the number of queries needed to recover preprocessing cost."""

    required = {
        "method", "map_kind", "map_size",
        "success", "runtime_s", "preprocess_s",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    columns = [
        "map_kind", "map_size", "preprocess_s",
        "baseline_query_s", "method_query_s", "breakeven",
    ]

    if method == baseline:
        raise ValueError("Method and baseline must be different.")

    data = df.copy()

    # Keep successful queries only.
    if data["success"].dtype == object:
        success = data["success"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    else:
        success = data["success"].fillna(False).astype(bool)

    data = data.loc[success]
    data = data[data["method"].isin([method, baseline])]

    if data.empty:
        return pd.DataFrame(columns=columns)

    # Calculate average query and preprocessing times for each method and map.
    averages = (
        data.groupby(
            ["method", "map_kind", "map_size"],
            as_index=False,
        )
        .agg(
            query_s=("runtime_s", "mean"),
            preprocess_s=("preprocess_s", "mean"),
        )
    )

    method_data = averages[averages["method"] == method][
        ["map_kind", "map_size", "query_s", "preprocess_s"]
    ].rename(columns={"query_s": "method_query_s"})

    baseline_data = averages[averages["method"] == baseline][
        ["map_kind", "map_size", "query_s"]
    ].rename(columns={"query_s": "baseline_query_s"})

    result = method_data.merge(
        baseline_data,
        on=["map_kind", "map_size"],
        how="inner",
    )

    # Calculate the time saved per query compared with the baseline.
    saving = result["baseline_query_s"] - result["method_query_s"]

    # If the method is not faster, its preprocessing cost cannot be recovered.
    result["breakeven"] = float("inf")

    faster = saving > 0
    free_preprocessing = faster & (result["preprocess_s"] <= 0)
    positive_preprocessing = faster & (result["preprocess_s"] > 0)

    result.loc[free_preprocessing, "breakeven"] = 0.0
    result.loc[positive_preprocessing, "breakeven"] = (
        result.loc[positive_preprocessing, "preprocess_s"]
        / saving.loc[positive_preprocessing]
    )

    return result[columns].sort_values(
        ["map_kind", "map_size"]
    ).reset_index(drop=True)
    
    


def plot_scaling(df: pd.DataFrame, out: str | Path, metric: str = "expanded",
                 map_kind: str | None = None) -> Path:
    if map_kind:
        df = df[df["map_kind"] == map_kind]
        
    summary = summarize(df, metric, by=["method", "map_size"])
    
    plt.figure(figsize=(8, 6))
    for method in summary["method"].unique():
        method_data = summary[summary["method"] == method].sort_values("map_size")
        
        yerr_lower = method_data["mean"] - method_data["ci_low"]
        yerr_upper = method_data["ci_high"] - method_data["mean"]
        
        plt.errorbar(
            method_data["map_size"], method_data["mean"], 
            yerr=[yerr_lower, yerr_upper], 
            label=method, marker='o', capsize=4
        )
        
    plt.yscale('log')
    plt.xlabel("Map Size")
    plt.ylabel(metric.capitalize())
    plt.title(f"Scaling of {metric}{' (' + map_kind + ')' if map_kind else ''}")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    
    return out


def plot_path_quality(df: pd.DataFrame, out: str | Path) -> Path:
    hierarchical_df = df[(df["suboptimality"] > 1.0) | (df["method"].str.contains("-"))]
    
    plt.figure(figsize=(10, 6))
    
    methods = hierarchical_df["method"].unique()
    data = [hierarchical_df[hierarchical_df["method"] == m]["suboptimality"].dropna() for m in methods]
    
    plt.boxplot(data, labels=methods)
    plt.axhline(1.0, color='red', linestyle='--', label='Optimal (1.0)')
    plt.ylabel("Suboptimality (cost / optimal_cost)")
    plt.title("Path Quality of Hierarchical Methods")
    plt.xticks(rotation=45, ha="right")
    
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    
    return out


def plot_partitions(df: pd.DataFrame, out: str | Path, metric: str = "expanded") -> Path:
    df_part = df[df["method"].str.contains("-")] 
    summary = summarize(df_part, metric, by=["method", "map_kind"])
    
    kinds = summary["map_kind"].unique()
    methods = summary["method"].unique()
    
    x = np.arange(len(kinds))
    width = 0.8 / len(methods)
    
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        method_data = summary[summary["method"] == method]
    
        means = [method_data[method_data["map_kind"] == k]["mean"].values[0] if k in method_data["map_kind"].values else 0 for k in kinds]
        
        plt.bar(x + i*width - width*(len(methods)-1)/2, means, width, label=method)
        
    plt.xlabel("Map Kind")
    plt.ylabel(metric.capitalize())
    plt.title(f"Comparison of Partition Methods ({metric})")
    plt.xticks(x, kinds)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    
    return out


def plot_breakeven(df: pd.DataFrame, out: str | Path, method: str, baseline: str = "A*") -> Path:
    """Plot estimated cumulative time for a method and its baseline."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = breakeven_queries(df, method, baseline)
    fig, ax = plt.subplots(figsize=(9, 5))

    if results.empty:
        ax.text(
            0.5, 0.5,
            "No matching successful results to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        # Plot up to 100 queries, or enough to show the largest
        # finite break-even point (with a sensible upper limit).
        finite_points = results.loc[
            np.isfinite(results["breakeven"]),
            "breakeven",
        ]

        max_queries = 100
        if not finite_points.empty:
            max_queries = max(
                max_queries,
                min(10000, int(np.ceil(finite_points.max())) + 10),
            )

        queries = np.arange(max_queries + 1)

        for _, row in results.iterrows():
            label = f"{row['map_kind']} ({row['map_size']})"

            baseline_time = row["baseline_query_s"] * queries
            method_time = (
                row["preprocess_s"]
                + row["method_query_s"] * queries
            )

            ax.plot(
                queries,
                baseline_time,
                linestyle="--",
                label=f"{label}: {baseline}",
            )
            ax.plot(
                queries,
                method_time,
                label=f"{label}: {method}",
            )

        ax.set_xlabel("Number of queries")
        ax.set_ylabel("Estimated cumulative time (seconds)")
        ax.set_title(f"Break-even analysis: {method} vs {baseline}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out


def plot_repair(trials: list[dict], out: str | Path) -> Path:
    """Figure: local repair time against full rebuild time, one pair of bars per map.

    Bars show the median over the map's trials; dots show every trial. The y-axis is
    logarithmic because rebuilds take 10-30x longer than repairs. Trials without a
    ``map_name`` are grouped together as "all maps".
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[str, dict[str, list[float]]] = {}
    for trial in trials:
        try:
            repair, rebuild = float(trial["repair_s"]), float(trial["rebuild_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(repair) and np.isfinite(rebuild)):
            continue
        g = groups.setdefault(str(trial.get("map_name", "all maps")), {"repair": [], "rebuild": []})
        g["repair"].append(repair)
        g["rebuild"].append(rebuild)

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(groups) + 2), 5))
    if not groups:
        ax.text(0.5, 0.5, "No repair/rebuild timings", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        names = list(groups)
        x = np.arange(len(names))
        width = 0.38
        rng = np.random.default_rng(0)                      # fixed jitter for the dots
        for offset, key, label, colour in ((-width / 2, "repair", "Local repair", "tab:blue"),
                                           (width / 2, "rebuild", "Full rebuild", "tab:orange")):
            medians = [np.median(groups[n][key]) for n in names]
            ax.bar(x + offset, medians, width, label=label, color=colour, alpha=0.75)
            for i, n in enumerate(names):
                vals = groups[n][key]
                ax.scatter(x[i] + offset + rng.uniform(-width / 4, width / 4, len(vals)), vals,
                           s=10, color="black", alpha=0.5, zorder=3)
        for i, n in enumerate(names):                       # speed-up above each pair
            speedup = np.median(groups[n]["rebuild"]) / np.median(groups[n]["repair"])
            top = max(max(groups[n]["rebuild"]), max(groups[n]["repair"]))
            ax.text(x[i], top * 1.35, f"{speedup:.0f}x faster", ha="center", fontsize=9)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("Time per change (seconds, log scale)")
        ax.set_title("Blocking one doorway: local repair vs full rebuild")
        ax.set_ylim(top=ax.get_ylim()[1] * 2)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all(results_dir: str | Path = "results", out_dir: str | Path = "results/figures") -> list[Path]:
    """Regenerate every figure and summary table of the report in one call.

    Each experiment's CSV is loaded on its own: the scaling and partitions experiments
    share some maps, so loading them together would count those queries twice, and
    dynamic.csv has different columns. pilot.csv is a test run and is not used.
    Missing files are skipped with a message.
    """
    res = Path(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def load(name: str):
        path = res / f"{name}.csv"
        if not path.exists():
            print(f"make_all: {path} not found, skipping its figures")
            return None
        return load_results([path])

    # summary tables (mean, sd, median, 95% CI) for every query experiment
    for name in ("scaling", "partitions", "street"):
        df = load(name)
        if df is None:
            continue
        for metric in ("runtime_s", "expanded", "suboptimality"):
            path = out / f"summary_{name}_{metric}.csv"
            summarize(df, metric).to_csv(path, index=False)
            written.append(path)

    # scaling: one figure per map kind and measure, plus path quality
    scaling = load("scaling")
    if scaling is not None:
        for kind in sorted(scaling["map_kind"].unique()):
            for metric in ("runtime_s", "expanded"):
                written.append(plot_scaling(scaling, out / f"scaling_{kind}_{metric}.png",
                                            metric=metric, map_kind=kind))
        written.append(plot_path_quality(scaling, out / "path_quality_scaling.png"))
        rooms = scaling[scaling["map_kind"] == "rooms"]
        written.append(plot_breakeven(rooms, out / "breakeven_rooms_HPA-watershed.png",
                                      "HPA-watershed"))

    # partitions: the three partition methods, then the watershed area-cap sweep
    partitions = load("partitions")
    if partitions is not None:
        caps = ["HPA-watershed-nocap", "HPA-watershed-cap1024", "HPA-watershed",
                "HPA-watershed-cap256"]
        main = partitions[~partitions["method"].isin(caps[:2] + caps[3:])]
        sweep = partitions[partitions["method"].isin(caps)]
        for metric in ("runtime_s", "suboptimality"):
            written.append(plot_partitions(main, out / f"partitions_{metric}.png", metric=metric))
            written.append(plot_partitions(sweep, out / f"area_cap_{metric}.png", metric=metric))

    # street maps: the real-city case study
    street = load("street")
    if street is not None:
        for metric in ("runtime_s", "expanded"):
            written.append(plot_scaling(street, out / f"street_{metric}.png", metric=metric))
        written.append(plot_path_quality(street, out / "path_quality_street.png"))
        written.append(plot_breakeven(street, out / "breakeven_street_HPA-fixed.png", "HPA-fixed"))

    # dynamic maps: local repair vs full rebuild (different columns, read directly)
    dyn_path = res / "dynamic.csv"
    if dyn_path.exists():
        written.append(plot_repair(pd.read_csv(dyn_path).to_dict("records"), out / "repair.png"))
    else:
        print(f"make_all: {dyn_path} not found, skipping the repair figure")

    return written
