from __future__ import annotations
from pathlib import Path
from grid import Grid, Scenario
import numpy as np


OPEN_CHARS = frozenset(".GS")
WALL_CHARS = frozenset("@OTW")

def read_map(path: str | Path, diagonal: bool = True, pad_border: bool = False) -> Grid:
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

    if pad_border:
        blocked = np.pad(blocked, 1, constant_values=True)
    return Grid(blocked, diagonal)


def write_map(grid: Grid, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["".join("@" if wall else "." for wall in row) for row in grid.blocked]
    text = "\n".join(["type octile", f"height {grid.height}", f"width {grid.width}", "map", *rows])
    path.write_text(text + "\n")


def read_scenarios(path: str | Path, offset: int = 0) -> list[Scenario]:
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines or not lines[0].strip().lower().startswith("version"):
        raise ValueError(f"{path}: line 1: expected 'version 1', got {lines[0] if lines else ''!r}")

    scenarios: list[Scenario] = []
    for i, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 9:
            raise ValueError(f"{path}: line {i}: expected 9 fields, got {len(fields)}")
        try:
            sx, sy, gx, gy = (int(v) for v in fields[4:8])
            optimal = float(fields[8])
        except ValueError:
            raise ValueError(f"{path}: line {i}: coordinates must be whole numbers "
                             "and the optimal length a number") from None
        scenarios.append(Scenario(start=(sy + offset, sx + offset),
                                  goal=(gy + offset, gx + offset),
                                  optimal_cost=optimal,
                                  id=len(scenarios)))
    return scenarios


def write_scenarios(path: str | Path, scenarios: list[Scenario], map_name: str,
                    width: int, height: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["version 1"]
    for sc in scenarios:
        (sr, scol), (gr, gcol) = sc.start, sc.goal
        lines.append("\t".join([str(int(sc.optimal_cost // 4)), map_name, str(width), str(height),
                                str(scol), str(sr), str(gcol), str(gr),
                                f"{sc.optimal_cost:.8f}"]))
    path.write_text("\n".join(lines) + "\n")

