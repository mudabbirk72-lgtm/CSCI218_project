from __future__ import annotations
from pathlib import Path
from hierpath.grid import Grid, Scenario
import numpy as np

OPEN_CHARS = frozenset(".GS")
WALL_CHARS = frozenset("@OTW")

def read_map(path: str | Path, diagonal: bool = True) -> Grid:
    path = Path(path)
    lines = path.read_text().splitlines()

    header: dict[str, str] = {}
    map_line = None
    for i, line in enumerate(lines):
        words = line.split()
        if not words:
            continue
        if words[0].lower() == "map":
            map_line = i
            break
        if len(words) != 2:
            raise ValueError(f"{path}: line {i + 1}: expected 'key value' in the header, got {line!r}")
        header[words[0].lower()] = words[1]
    if map_line is None:
        raise ValueError(f"{path}: no 'map' line found")
    for key in ("height", "width"):
        if key not in header:
            raise ValueError(f"{path}: header has no '{key}' line")
    try:
        height, width = int(header["height"]), int(header["width"])
    except ValueError:
        raise ValueError(f"{path}: height and width must be whole numbers") from None
    if height < 1 or width < 1:
        raise ValueError(f"{path}: height and width must be at least 1")

    rows = lines[map_line + 1:]
    while rows and not rows[-1].strip():
        rows.pop()
    if len(rows) != height:
        raise ValueError(f"{path}: header says height {height}, but the file has {len(rows)} map rows")

    blocked = np.zeros((height, width), dtype=bool)
    for r, row in enumerate(rows):
        line_no = map_line + 2 + r
        row = row.rstrip("\r")
        if len(row) != width:
            raise ValueError(f"{path}: line {line_no}: row has {len(row)} characters, expected {width}")
        for c, ch in enumerate(row):
            if ch in WALL_CHARS:
                blocked[r, c] = True
            elif ch not in OPEN_CHARS:
                raise ValueError(f"{path}: line {line_no}, column {c + 1}: unknown character {ch!r}")
    return Grid(blocked, diagonal)


def write_map(grid: Grid, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["".join("@" if wall else "." for wall in row) for row in grid.blocked]
    text = "\n".join(["type octile", f"height {grid.height}", f"width {grid.width}", "map", *rows])
    path.write_text(text + "\n")


def read_scenarios(path: str | Path) -> list[Scenario]:
    raise NotImplementedError


def write_scenarios(path: str | Path, scenarios: list[Scenario], map_name: str,
                    width: int, height: int) -> None:
    raise NotImplementedError


def image_to_grid(image_path: str | Path, cell_px: int = 4, threshold: int = 128,
                  diagonal: bool = True) -> Grid:

    raise NotImplementedError
