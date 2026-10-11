from __future__ import annotations
from pathlib import Path
import pandas as pd


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
    raise NotImplementedError


def plot_scaling(df: pd.DataFrame, out: str | Path, metric: str = "expanded",
                 map_kind: str | None = None) -> Path:
    raise NotImplementedError


def plot_path_quality(df: pd.DataFrame, out: str | Path) -> Path:
    raise NotImplementedError


def plot_partitions(df: pd.DataFrame, out: str | Path, metric: str = "expanded") -> Path:
    raise NotImplementedError


def plot_breakeven(df: pd.DataFrame, out: str | Path, method: str, baseline: str = "A*") -> Path:
    raise NotImplementedError


def plot_repair(trials: list[dict], out: str | Path) -> Path:
    raise NotImplementedError


def make_all(results_dir: str | Path = "results", out_dir: str | Path = "results/figures") -> list[Path]:
    raise NotImplementedError
