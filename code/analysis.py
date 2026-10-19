from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(paths: list[str | Path]) -> pd.DataFrame:
    raise NotImplementedError


def confidence_interval(values, level: float = 0.95, n_boot: int = 10_000,
                        seed: int = 0) -> tuple[float, float]:
    raise NotImplementedError


def summarize(df: pd.DataFrame, metric: str, by: list[str] | None = None) -> pd.DataFrame:
    raise NotImplementedError


def compare(df: pd.DataFrame, method_a: str, method_b: str, metric: str) -> dict:
    raise NotImplementedError


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
    raise NotImplementedError


def plot_path_quality(df: pd.DataFrame, out: str | Path) -> Path:
    raise NotImplementedError


def plot_partitions(df: pd.DataFrame, out: str | Path, metric: str = "expanded") -> Path:
    raise NotImplementedError


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
    """Plot repair and full-rebuild timings from dynamic experiment trials."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    # Accept common names for repair and full-rebuild timing fields.

    repair_keys = (
        "repair_s", "repair_time_s", "local_repair_s",
        "repair_time", "repair_runtime_s",
    )
    rebuild_keys = (
        "rebuild_s", "rebuild_time_s", "full_rebuild_s",
        "rebuild_time", "rebuild_runtime_s",
    )

    def get_timing(trial: dict, keys: tuple[str, ...]):
        for key in keys:
            value = trial.get(key)

            if value is not None:
                try:
                    number = float(value)
                    if np.isfinite(number) and number >= 0:
                        return number
                except (TypeError, ValueError):
                    continue

        return None

    trial_numbers = []
    repair_times = []
    rebuild_times = []

    for index, trial in enumerate(trials, start=1):
        repair_time = get_timing(trial, repair_keys)
        rebuild_time = get_timing(trial, rebuild_keys)
        
        # Include a trial if at least one timing value is available.

        if repair_time is None and rebuild_time is None:
            continue

        trial_numbers.append(
            trial.get("trial", trial.get("change_id", index))
        )
        repair_times.append(
            np.nan if repair_time is None else repair_time
        )
        rebuild_times.append(
            np.nan if rebuild_time is None else rebuild_time
        )

    if not trial_numbers:
        ax.text(
            0.5, 0.5,
            "No recognised repair/rebuild timing data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        ax.plot(
            trial_numbers,
            repair_times,
            marker="o",
            label="Repair",
        )
        ax.plot(
            trial_numbers,
            rebuild_times,
            marker="s",
            label="Full rebuild",
        )

        ax.set_xlabel("Trial / change number")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Dynamic update: repair vs full rebuild")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out


def make_all(results_dir: str | Path = "results", out_dir: str | Path = "results/figures") -> list[Path]:
    raise NotImplementedError
