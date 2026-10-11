from __future__ import annotations
from pathlib import Path
import numpy as np
from grid import Cell, Grid, SearchResult


STAGES: list[str] = ["map", "distance", "areas", "graph", "route", "path", "compare"]

def area_color(area: int) -> tuple[int, int, int]:
    raise NotImplementedError


class App:

    def __init__(self, method: str = "HPA-watershed", window_size: tuple[int, int] = (1280, 800)) -> None:
        raise NotImplementedError

    def load_map(self, path: str | Path) -> None:
        raise NotImplementedError

    def generate_map(self, kind: str, size: int, seed: int) -> None:
        raise NotImplementedError

    def set_start(self, cell: Cell) -> bool:
        raise NotImplementedError

    def set_goal(self, cell: Cell) -> bool:
        raise NotImplementedError

    def run_query(self) -> None:
        raise NotImplementedError

    def next_stage(self) -> None:
        raise NotImplementedError

    def prev_stage(self) -> None:
        raise NotImplementedError

    def toggle_block(self, cell: Cell) -> None:
        raise NotImplementedError

    def cell_at(self, x: int, y: int) -> Cell | None:
        raise NotImplementedError

    # ------------------------------------------------------------------ editor mode

    def toggle_editor(self) -> None:
        raise NotImplementedError

    def paint(self, cell: Cell, wall: bool) -> None:
        raise NotImplementedError

    def save_map(self, path: str | Path) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ drawing and loop

    def draw(self) -> None:
        raise NotImplementedError

    def handle_event(self, event) -> bool:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError


def save_figure(grid: Grid, labels: np.ndarray | None, path_cells: list[Cell] | None,
                out: str | Path, result: SearchResult | None = None) -> Path:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError
