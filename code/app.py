from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import numpy as np
import pygame

import experiments
from grid import Cell, Grid, SearchResult
from map_io import read_map, write_map
from partition import distance_map

STAGES: list[str] = ["map", "distance", "areas", "graph", "route", "path", "compare"]
DEMO_METHODS: list[str] = ["HPA-watershed", "HPA-fixed", "Corridor-watershed", "Corridor-fixed",
                           "Basic-watershed", "Basic-fixed"]
"""Methods the M key cycles through, in order."""
PANEL = 340
BG, PANEL_BG, TXT, DIM = (22, 24, 30), (30, 33, 41), (222, 226, 234), (140, 146, 160)
START, GOAL, HIER, FLAT = (46, 204, 113), (231, 76, 60), (255, 140, 0), (52, 152, 219)
HIER_PATH, FLAT_PATH, WALL, OPEN = (214, 40, 160), (0, 200, 220), (34, 36, 44), (236, 236, 232)


def area_color(area: int) -> tuple[int, int, int]:
    """A fixed colour for an area id, so an area keeps its colour between frames.

    Args:
        area (int): Area id, >= 0.

    Returns:
        tuple[int, int, int]: An ``(R, G, B)`` colour, each 0-255. Different ids give
        visibly different colours.
    """
    if area < 0:
        return (90, 90, 96)
    # golden-ratio hue steps spread consecutive ids around the colour wheel
    r, g, b = colorsys.hsv_to_rgb((area * 0.6180339887) % 1, (.5, .68, .58)[area % 3],
                                  (.95, .82, .9, .74)[area // 3 % 4])
    return int(r * 255), int(g * 255), int(b * 255)


def _base(walls: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """One RGB pixel per cell: walls dark, open cells light or coloured by area."""
    img = np.where(walls[..., None], WALL, OPEN).astype(np.uint8)
    if labels is not None:
        ok = (labels >= 0) & ~walls
        ids, inv = np.unique(labels[ok], return_inverse=True)
        img[ok] = np.array([area_color(int(i)) for i in ids], np.uint8).reshape(-1, 3)[inv.ravel()]
    return img


def _scale(img: np.ndarray, px: int) -> pygame.Surface:
    """Turn a ``(rows, cols, 3)`` image into a surface with ``px`` pixels per cell."""
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
    return pygame.transform.scale(surf, (img.shape[1] * px, img.shape[0] * px))


class App:

    def __init__(self, method: str = "HPA-watershed", window_size: tuple[int, int] = (1280, 800)) -> None:
        """Open the window.

        Args:
            method (str): Which hierarchical method to show.
            window_size (tuple[int, int]): ``(width, height)`` of the window in pixels.

        Returns:
            None.
        """
        pygame.init()
        pygame.display.set_caption(f"Hierarchical path planning - {method}")
        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 15)
        self.method, self.diagonal = method, True        # diagonal: set before load/generate
        self.grid = self.planner = self.baseline = self.labels = None
        self.start = self.goal = self.result = self.baseline_result = None
        self.explored, self.baseline_explored = [], []
        self.stage, self.editing, self.brush, self.cell_px = 0, False, 1, 1
        self.walls, self.stats, self.prep = np.zeros((1, 1), bool), {}, 0.0
        self.dirty, self.msg, self.ox, self.oy = False, "Left click: start, right click: goal", 0, 0
        self._surf, self._nodes, self._edges, self._cen, self._route = None, [], [], {}, []

    # helpers
    
    def _fit(self) -> None:
        """Pick ``cell_px`` and the map origin so the map fits left of the panel."""
        w, h = self.screen.get_size()
        rows, cols = self.walls.shape
        self.cell_px = max(1, min((w - PANEL) // cols, h // rows))
        self.ox, self.oy = (w - PANEL - cols * self.cell_px) // 2, (h - rows * self.cell_px) // 2
        self._surf = None

    def _use(self, grid: Grid, reset: bool = True) -> None:
        """Preprocess both planners on ``grid`` (the old map stays if this raises)."""
        planner, baseline = experiments.make_planner(self.method), experiments.make_planner("A*")
        self.prep = float(planner.preprocess(grid) or 0)
        baseline.preprocess(grid)
        lab = getattr(planner, "labels", None)
        self.grid, self.planner, self.baseline = grid, planner, baseline
        self.walls, self.dirty, self.stats = grid.blocked.copy(), False, planner.stats()
        self.labels = None if lab is None else np.asarray(lab).astype(int)
        if reset:
            self.start = self.goal = self.result = self.baseline_result = None
            self.explored, self.baseline_explored, self.stage = [], [], 0
        self._fit()

    def _rebuild(self) -> None:
        """Apply pending wall edits: new grid, new preprocessing, query re-run."""
        if not self.dirty:
            return
        walls, start, goal = self.walls.copy(), self.start, self.goal
        self._use(Grid(walls, self.diagonal), reset=False)
        self.start, self.goal = (None if c and walls[c] else c for c in (start, goal))
        if None in (self.start, self.goal):
            self.result = self.baseline_result = None
            self.explored, self.baseline_explored = [], []
        self.run_query()
        self.msg = "Map edited; planners re-preprocessed"

    # map and queries
    
    def load_map(self, path: str | Path) -> None:
        """Load a ``.map`` file and prepare both planners on it.

        Args:
            path (str | Path): The map file.

        Returns:
            None. ``grid``, ``planner`` and ``baseline`` are set; ``start``, ``goal``
            and results are cleared; ``stage`` returns to 0.
        """
        grid = read_map(path, self.diagonal)
        b = grid.blocked
        if not (b[0].all() and b[-1].all() and b[:, 0].all() and b[:, -1].all()):
            grid = read_map(path, self.diagonal, pad_border=True)   # e.g. street maps
        self._use(grid)
        self.msg = f"Loaded {Path(path).name}"

    def generate_map(self, kind: str, size: int, seed: int) -> None:
        """Create a new generated map and prepare both planners on it.

        Args:
            kind (str): ``"random"``, ``"rooms"`` or ``"maze"``.
            size (int): Side length in cells.
            seed (int): Random seed.

        Returns:
            None. Same effects as ``load_map``.
        """
        grid = experiments.make_map(kind, size, seed)
        grid.diagonal = self.diagonal
        self._use(grid)
        self.generated = (kind, size, seed)          # remembered for the G key
        self.msg = f"Generated {kind} {size}x{size} (seed {seed})"

    def set_start(self, cell: Cell) -> bool:
        """Choose the start cell.

        Args:
            cell (Cell): The clicked cell.

        Returns:
            bool: ``True`` if the cell is open and was set (the query is re-run if a
            goal exists); ``False`` if it is a wall (nothing changes).
        """
        if self.grid is None or not self.grid.is_free(cell):
            return False
        self.start = cell
        self.run_query()
        return True

    def set_goal(self, cell: Cell) -> bool:
        """Choose the goal cell.

        Args:
            cell (Cell): The clicked cell.

        Returns:
            bool: ``True`` if set (the query is re-run if a start exists); ``False`` if
            the cell is a wall.
        """
        if self.grid is None or not self.grid.is_free(cell):
            return False
        self.goal = cell
        self.run_query()
        return True

    def run_query(self) -> None:
        """Run the current start/goal on both planners, recording explored cells.

        Passes an ``on_expand`` function to each ``query`` call that appends the cell
        to ``explored`` / ``baseline_explored``.

        Returns:
            None. Sets ``result``, ``baseline_result``, ``explored``,
            ``baseline_explored``. Does nothing if start or goal is missing.
        """
        if self.start is None or self.goal is None:
            return
        self.result = self.baseline_result = None
        self.explored, self.baseline_explored, self._surf = [], [], None
        try:
            self.result = self.planner.query(self.start, self.goal, on_expand=self.explored.append)
            self.baseline_result = self.baseline.query(self.start, self.goal,
                                                       on_expand=self.baseline_explored.append)
            self.msg = "Arrows / wheel: change stage" if self.result.success else "No path found"
        except Exception as err:                           # a planner bug must not close the window
            self.result = self.baseline_result = None
            self.msg = f"Query failed: {err}"

    def next_method(self) -> None:
        """Switch to the next method in DEMO_METHODS on the same map, keeping start and goal."""
        i = DEMO_METHODS.index(self.method) if self.method in DEMO_METHODS else -1
        self.method = DEMO_METHODS[(i + 1) % len(DEMO_METHODS)]
        self._use(self.grid, reset=False)
        self._surf = None
        self.run_query()
        self.msg = f"Method: {self.method}"

    def next_stage(self) -> None:
        """Show the next stage (wraps around)."""
        self.stage, self._surf = (self.stage + 1) % len(STAGES), None

    def prev_stage(self) -> None:
        """Show the previous stage (wraps around)."""
        self.stage, self._surf = (self.stage - 1) % len(STAGES), None

    # editing
    
    def toggle_block(self, cell: Cell) -> None:
        """Block an open cell or open a wall cell, live (the door-blocking demo).

        Calls ``planner.update`` and ``baseline.update`` with the change, then re-runs
        the query so the new path appears.

        Args:
            cell (Cell): The cell under the mouse; border cells are ignored.

        Returns:
            None.
        """
        rows, cols = self.walls.shape
        if not (0 < cell[0] < rows - 1 and 0 < cell[1] < cols - 1):
            return                                         # border cells are ignored
        self.walls[cell] ^= True
        self.dirty, self._surf = True, None
        if self.editing:
            return                                         # editor edits are applied when it closes
        blocked = bool(self.walls[cell])
        try:                                               # live: incremental update, no re-preprocessing
            secs = float(self.planner.update(cell, blocked) or 0)
            self.baseline.update(cell, blocked)
        except (NotImplementedError, AttributeError):      # no update(): rebuild everything
            return self._rebuild()
        lab = getattr(self.planner, "labels", None)
        self.labels = None if lab is None else np.asarray(lab).astype(int)
        self.stats, self.dirty = self.planner.stats(), False
        self.start, self.goal = (None if c and self.walls[c] else c for c in (self.start, self.goal))
        if None in (self.start, self.goal):
            self.result = self.baseline_result = None
            self.explored, self.baseline_explored = [], []
        self.run_query()
        self.msg = f"{cell} {'blocked' if blocked else 'opened'}; updated in {secs * 1e3:.2f} ms"

    def cell_at(self, x: int, y: int) -> Cell | None:
        """Translate a mouse position into a cell.

        Args:
            x (int): Pixel column in the window.
            y (int): Pixel row in the window.

        Returns:
            Cell | None: The ``(row, col)`` under that pixel, or ``None`` if the pixel
            is outside the map picture.
        """
        r, c = (y - self.oy) // self.cell_px, (x - self.ox) // self.cell_px
        return (int(r), int(c)) if 0 <= r < self.walls.shape[0] and 0 <= c < self.walls.shape[1] else None

    def toggle_editor(self) -> None:
        """Switch editor mode on or off.

        Returns:
            None. Flips ``editing``. When switching OFF, both planners are prepared
            again on the edited map and start/goal are kept if still open.
        """
        self.editing, self._surf = not self.editing, None
        if self.editing:
            self.msg = "EDITOR: left drag = wall, right drag = erase, [ ] = brush"
        else:
            self._rebuild()

    def paint(self, cell: Cell, wall: bool) -> None:
        """Paint a square of walls or open space (editor mode only).

        Args:
            cell (Cell): Centre of the brush.
            wall (bool): ``True`` = paint walls, ``False`` = erase to open space.

        Returns:
            None. A ``brush`` x ``brush`` square centred on ``cell`` is changed; the
            outer border of the map always stays wall.
        """
        if not self.editing:
            return
        r, c, h, b = *cell, self.brush // 2, self.brush
        w = self.walls
        w[max(0, r - h):max(0, r - h + b), max(0, c - h):max(0, c - h + b)] = wall
        w[[0, -1]] = w[:, [0, -1]] = True                  # the outer border always stays wall
        self.dirty, self._surf = True, None

    def save_map(self, path: str | Path) -> None:
        """Save the current map as a ``.map`` file.

        Args:
            path (str | Path): Where to write, e.g. ``"maps/generated/my_map.map"``.

        Returns:
            None.
        """
        self._rebuild()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        write_map(self.grid, path)
        self.msg = f"Saved {path}"

    # drawing and loop
    
    def draw(self) -> None:
        """Draw the current stage and the statistics panel.

        What is drawn depends on ``STAGES[stage]`` (see ``STAGES``). The panel shows,
        for each planner: path cost, nodes expanded, query time, and for the
        hierarchical method the number of areas/doorways and how much longer its path
        is than the optimal one.

        Returns:
            None.
        """
        scr, px, w, lab = self.screen, self.cell_px, self.walls, self.labels
        st = "map" if self.editing else STAGES[self.stage]
        W, H = scr.get_size()
        scr.fill(BG)
        path = lambda r: (r.path if r and r.success else None) or []
        pt = lambda rc: (self.ox + (rc[1] + .5) * px, self.oy + (rc[0] + .5) * px)
        line = lambda cells, col, wd: len(cells) > 1 and pygame.draw.lines(scr, col, False, [pt(c) for c in cells], wd)

        if self._surf is None:                             # rebuilt only when something changed
            img = _base(w, lab if st in ("areas", "graph", "route") else None)

            def tint(cells, col, a):
                if cells:
                    rc = np.array(cells)
                    img[rc[:, 0], rc[:, 1]] = img[rc[:, 0], rc[:, 1]] * (1 - a) + np.array(col) * a

            if st == "distance" and (~w).any():            # distance to the nearest wall
                d = distance_map(w)
                t = (d / max(d.max(), 1))[..., None]
                img[~w] = (t * (253, 231, 37) + (1 - t) * (68, 1, 84))[~w]
            if st in ("path", "compare"):
                tint(self.baseline_explored if st == "compare" else [], FLAT, .45)
                tint(self.explored, HIER, .5)
            pl = self.planner
            self._nodes, self._edges, self._cen, self._route = [], [], {}, []
            if lab is not None and st == "graph":
                if hasattr(pl, "area_graph"):              # two-tier: areas joined by crossings
                    cen = pl.area_graph.centroids
                    self._nodes, self._edges = list(cen.values()), [(cen[a], cen[b]) for a, b in pl.area_graph.edges]
                else:                                      # HPA: doorway nodes joined by edges
                    pos = {i: n[1] for i, n in pl.graph.nodes.items()}
                    self._nodes = list(pos.values())
                    self._edges = [(pos[u], pos[v]) for u, nb in pl.graph.adj.items() for v in nb if u < v]
            if lab is not None and st == "route" and path(self.result):
                ids = [int(lab[c]) for c in path(self.result)]
                self._route = [a for i, a in enumerate(ids) if i == 0 or a != ids[i - 1]]
                if hasattr(pl, "area_graph"):              # two-tier planners report their real area route
                    self._route = list(self.result.extra.get("route") or self._route)
                self._cen = {a: np.argwhere(lab == a).mean(0) for a in self._route}
                dim = ~np.isin(lab, self._route) & ~w
                img[dim] = img[dim] * .3 + 190 * .7
            self._surf = _scale(img, px)
        scr.blit(self._surf, (self.ox, self.oy))

        for p, q in self._edges:
            line([p, q], (60, 60, 70), 2)
        for p in self._nodes:
            pygame.draw.circle(scr, (20, 20, 24), pt(p), max(2, px // 2 + 1))
        line([self._cen[a] for a in self._route], (20, 20, 24), max(3, px))
        if st in ("route", "path", "compare"):
            line(path(self.baseline_result) if st == "compare" else [], FLAT_PATH, max(3, px // 2 + 1))
            line(path(self.result), HIER_PATH, max(2, px // 2))
        for c, col in ((self.start, START), (self.goal, GOAL)):
            if c:
                pygame.draw.circle(scr, (0, 0, 0), pt(c), max(3, px) + 2)
                pygame.draw.circle(scr, col, pt(c), max(3, px))
        if self.editing and (c := self.cell_at(*pygame.mouse.get_pos())):
            pygame.draw.rect(scr, (255, 220, 0), (self.ox + (c[1] - self.brush // 2) * px,
                                                  self.oy + (c[0] - self.brush // 2) * px,
                                                  self.brush * px, self.brush * px), 2)

        # statistics panel
        h, f = self.result, self.baseline_result
        rows = lambda r: ([f"  cost     {r.cost:.2f}", f"  expanded {r.expanded}",
                           f"  time     {r.runtime_s * 1e3:.2f} ms"] if r.success else ["  no path"]) if r else ["  -"]
        s = self.stats
        out = [(self.method, HIER), ("EDITOR - press E to apply" if self.editing else f"Stage {self.stage + 1}/{len(STAGES)}: {st}", TXT), ("", TXT),
               ("Hierarchical", HIER), *((t, TXT) for t in rows(h)),
               (f"  areas {s.get('areas', '?')}  doorways {s.get('doorways', '?')}", TXT),
               (f"  preprocess {self.prep:.3f} s", TXT), ("", TXT),
               ("Flat A*", FLAT), *((t, TXT) for t in rows(f))]
        if h and f and h.success and f.cost:
            out += [("", TXT), (f"Suboptimality {h.cost / f.cost:.3f}x", TXT)]
        pygame.draw.rect(scr, PANEL_BG, (W - PANEL, 0, PANEL, H))
        y = 12
        for text, col in out:
            scr.blit(self.font.render(text, True, col), (W - PANEL + 14, y))
            y += 19
        y = max(y + 10, H - 170)
        for text in (self.msg[:36], "", "L/R click: start / goal", "M click / B: toggle wall",
                     "arrows/wheel: stage", "E editor  [ ] brush", "M method  G new map", "S save map  F figure", "ESC quit"):
            scr.blit(self.font.render(text, True, DIM if text != self.msg[:36] else TXT), (W - PANEL + 14, y))
            y += 17
        pygame.display.flip()

    def handle_event(self, event) -> bool:
        """React to one mouse or keyboard event.

        Args:
            event: A ``pygame.event.Event``.

        Returns:
            bool: ``False`` if the window should close (quit event or ESC), else ``True``.
        """
        t = event.type
        free = lambda s: next(p for i in range(999) if not (p := Path(f"{s[0]}{i or ''}{s[1]}")).exists())
        if t == pygame.QUIT:
            return False
        if t == pygame.VIDEORESIZE:
            self.screen = pygame.display.get_surface()
            self._fit()
        elif t == pygame.KEYDOWN:
            k = event.key
            if k == pygame.K_ESCAPE:
                return False
            elif k in (pygame.K_RIGHT, pygame.K_DOWN):
                self.next_stage()
            elif k in (pygame.K_LEFT, pygame.K_UP):
                self.prev_stage()
            elif k == pygame.K_e:
                self.toggle_editor()
            elif k == pygame.K_b and self.grid and not self.editing:
                cell = self.cell_at(*pygame.mouse.get_pos())
                if cell:
                    self.toggle_block(cell)
            elif k == pygame.K_m and self.grid and not self.editing:
                self.next_method()
            elif k == pygame.K_g and not self.editing and getattr(self, "generated", None):
                kind, size, seed = self.generated
                self.generate_map(kind, size, seed + 1)
            elif k == pygame.K_LEFTBRACKET:
                self.brush = max(1, self.brush - 1)
            elif k == pygame.K_RIGHTBRACKET:
                self.brush = min(32, self.brush + 1)
            elif k == pygame.K_s and self.grid:
                self.save_map(free(("maps/generated/map_edited", ".map")))
            elif k == pygame.K_f and self.grid:
                out = save_figure(self.grid, self.labels, self.result.path if self.result and self.result.success else None,
                                  free((f"figure_{STAGES[self.stage]}", ".png")), self.result)
                self.msg = f"Saved {out}"
        elif t == pygame.MOUSEWHEEL and event.y:
            (self.next_stage if event.y < 0 else self.prev_stage)()
        elif t in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION) and self.grid:
            b = event.button if t == pygame.MOUSEBUTTONDOWN else (1 if event.buttons[0] else 3 if event.buttons[2] else 0)
            if b in (1, 2, 3) and self.editing and b != 2:
                steps = max(1, max(abs(v) for v in event.rel) // self.cell_px) if t == pygame.MOUSEMOTION else 1
                for i in range(steps + 1):                 # fill the gap between mouse events
                    cell = self.cell_at(event.pos[0] - event.rel[0] * i // steps, event.pos[1] - event.rel[1] * i // steps) \
                        if t == pygame.MOUSEMOTION else self.cell_at(*event.pos)
                    if cell:
                        self.paint(cell, b == 1)
            elif b in (1, 2, 3) and t == pygame.MOUSEBUTTONDOWN and (cell := self.cell_at(*event.pos)):
                if b == 2:
                    self.toggle_block(cell)
                elif not (self.set_start if b == 1 else self.set_goal)(cell):
                    self.msg = "Choose an open cell"
        return True

    def run(self) -> None:
        """Main loop: handle events, draw, repeat about 60 times a second until closed.

        Returns:
            None.
        """
        while all([self.handle_event(e) for e in pygame.event.get()]):
            self.draw()
            self.clock.tick(60)


def save_figure(grid: Grid, labels: np.ndarray | None, path_cells: list[Cell] | None,
                out: str | Path, result: SearchResult | None = None) -> Path:
    """Save a still picture of a map for the report (matplotlib; no window needed).

    Args:
        grid (Grid): The map.
        labels (np.ndarray | None): If given, areas are coloured.
        path_cells (list[Cell] | None): If given, drawn as a line.
        out (str | Path): PNG file to write.
        result (SearchResult | None): If given, its cost and expansions go in the title.

    Returns:
        Path: ``out``.
    """
    import matplotlib
    matplotlib.use("Agg")                                  # file output only: no window
    import matplotlib.pyplot as plt

    out = Path(out) if Path(out).suffix else Path(out).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    walls = np.array(grid.blocked, bool)
    img = _base(walls, None if labels is None else np.asarray(labels).astype(int))
    cells = path_cells or (result.path if result and result.success else None) or []
    fig, ax = plt.subplots(figsize=(8, 8 * walls.shape[0] / walls.shape[1] + .4))
    ax.imshow(img, interpolation="nearest")
    if len(cells) > 1:
        ax.plot([c[1] for c in cells], [c[0] for c in cells], color=np.array(HIER_PATH) / 255, lw=2)
        for c, col in ((cells[0], START), (cells[-1], GOAL)):
            ax.scatter(c[1], c[0], s=60, color=np.array(col) / 255, edgecolors="black", zorder=3)
    if result is not None:
        ax.set_title(f"cost {result.cost:.2f}, expanded {result.expanded}" if result.success else "no path")
    ax.axis("off")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line start of the demo.

    Args:
        argv (list[str] | None): Arguments; ``None`` = use the real command line.
            ``--map PATH`` loads a file; or ``--generate KIND --size N --seed S``;
            ``--method LABEL`` picks the hierarchical method.

    Returns:
        int: Exit code, 0 on a normal close.
    """
    ap = argparse.ArgumentParser(description="Hierarchical path planning demo")
    ap.add_argument("--map", help=".map file to load")
    ap.add_argument("--generate", choices=("random", "rooms", "maze"), help="generate a map instead")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="HPA-watershed", help="key of experiments.METHODS")
    ap.add_argument("--no-diagonal", action="store_true", help="4-connected moves")
    ap.add_argument("--window", default="1280x800", help="e.g. 1280x800")
    a = ap.parse_args(argv)
    if a.map and a.generate:
        ap.error("use either --map or --generate, not both")
    if a.size < 8:
        ap.error("--size must be at least 8")
    try:
        size = tuple(int(v) for v in a.window.lower().split("x"))
        app = App(a.method, size)
        app.diagonal = not a.no_diagonal
        app.load_map(a.map) if a.map else app.generate_map(a.generate or "rooms", a.size, a.seed)
        app.run()
    except (OSError, KeyError, ValueError, ImportError) as err:
        print(f"error: {err.args[0] if isinstance(err, KeyError) and err.args else err}", file=sys.stderr)
        return 1
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
