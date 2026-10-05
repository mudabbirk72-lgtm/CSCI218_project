
from __future__ import annotations

from pathlib import Path

from hierpath.grid import Grid, Scenario


def read_map(path: str | Path, diagonal: bool = True) -> Grid:
    raise NotImplementedError


def write_map(grid: Grid, path: str | Path) -> None:
    raise NotImplementedError


def read_scenarios(path: str | Path) -> list[Scenario]:
    raise NotImplementedError


def write_scenarios(path: str | Path, scenarios: list[Scenario], map_name: str,
                    width: int, height: int) -> None:
    raise NotImplementedError


def image_to_grid(image_path: str | Path, cell_px: int = 4, threshold: int = 128,
                  diagonal: bool = True) -> Grid:

    raise NotImplementedError
