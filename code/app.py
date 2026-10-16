from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import numpy as np
import pygame

import experiments
from grid import Cell, Grid, SearchResult
from mapio import read_map, write_map      # <- change to the module that holds read_map / write_map
from planners import FlatPlanner           # <- change to the module that holds FlatPlanner

STAGES: list[str] = ["map", "distance", "areas", "graph", "route", "path", "compare"]
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
        self._surf, self._cen, self._adj, self._route = None, {}, [], []


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
        planner, baseline = experiments.METHODS[self.method](), FlatPlanner()
        self.prep = float(planner.preprocess(grid) or 0)
        baseline.preprocess(grid)
        lab = getattr(planner, "labels", None)
        self.grid, self.planner, self.baseline = grid, planner, baseline
        self.walls, self.dirty, self.stats = grid.blocked.copy(), False, planner.stats()
        self.labels = None if lab is None else np.asarray(lab).astype(int)
        if reset:
            self.start = self.goal = None
            self.stage = 0
            self.run_query()
        self._fit()

    
    def _rebuild(self) -> None:
        """Apply pending wall edits: new grid, new preprocessing, query re-run."""
        if not self.dirty:
            return
        walls, start, goal = self.walls.copy(), self.start, self.goal
        self._use(Grid(walls, self.diagonal), reset=False)
        self.start, self.goal = (None if c and walls[c] else c for c in (start, goal))
        self.run_query()
        self.msg = "Map edited; planners re-preprocessed"


    # map and queries
    def load_map(self, path: str | Path) -> None:

        self._use(read_map(path, self.diagonal))
        self.msg = f"Loaded {Path(path).name}"

    def generate_map(self, kind: str, size: int, seed: int) -> None:

        rng, n = np.random.default_rng(seed), size
        w = np.zeros((n, n), bool)
        if kind == "random":
            w = rng.random((n, n)) < .28
        elif kind == "rooms":
            s = max(8, n // 6)
            w[::s] = w[:, ::s] = True
            for r in range(0, n, s):                       # one door per wall segment
                for c in range(0, n, s):
                    w[r, min(n - 1, c + rng.integers(1, s))] = w[min(n - 1, r + rng.integers(1, s)), c] = False
            w[[0, -1]] = w[:, [0, -1]] = True
        elif kind == "maze":                               # depth-first carving on odd cells
            w[:], m, stack = True, n - (n % 2 == 0), [(1, 1)]
            w[1, 1] = False
            while stack:
                r, c = stack[-1]
                nb = [(r + a, c + b) for a, b in ((2, 0), (-2, 0), (0, 2), (0, -2))
                      if 0 < r + a < m - 1 and 0 < c + b < m - 1 and w[r + a, c + b]]
                if not nb:
                    stack.pop()
                    continue
                y, x = nb[rng.integers(len(nb))]
                w[(r + y) // 2, (c + x) // 2] = w[y, x] = False
                stack.append((y, x))
        else:
            raise ValueError(f"unknown map kind {kind!r}; use random, rooms or maze")
        self._use(Grid(w, self.diagonal))
        self.msg = f"Generated {kind} {n}x{n} (seed {seed})"

    def set_start(self, cell: Cell) -> bool:

        if self.grid is None or not self.grid.is_free(cell):
            return False
        self.start = cell
        self.run_query()
        return True

    def set_goal(self, cell: Cell) -> bool:

        if self.grid is None or not self.grid.is_free(cell):
            return False
        self.goal = cell
        self.run_query()
        return True

    def run_query(self) -> None:

        self.result = self.baseline_result = None
        self.explored, self.baseline_explored, self._surf = [], [], None
        if self.start is None or self.goal is None:
            return
        try:
            self.result = self.planner.query(self.start, self.goal, on_expand=self.explored.append)
            self.baseline_result = self.baseline.query(self.start, self.goal,
                                                       on_expand=self.baseline_explored.append)
            self.msg = "Arrows / wheel: change stage" if self.result.success else "No path found"
        except Exception as err:                           # a planner bug must not close the window
            self.result = self.baseline_result = None
            self.msg = f"Query failed: {err}"

    def next_stage(self) -> None:
        """Show the next stage (wraps around)."""
        self.stage, self._surf = (self.stage + 1) % len(STAGES), None

    def prev_stage(self) -> None:
        """Show the previous stage (wraps around)."""
        self.stage, self._surf = (self.stage - 1) % len(STAGES), None

    # editing
    def toggle_block(self, cell: Cell) -> None:

        self.walls[cell] ^= True
        self.dirty, self._surf = True, None
        if not self.editing:
            self._rebuild()

    def cell_at(self, x: int, y: int) -> Cell | None:

        r, c = (y - self.oy) // self.cell_px, (x - self.ox) // self.cell_px
        return (int(r), int(c)) if 0 <= r < self.walls.shape[0] and 0 <= c < self.walls.shape[1] else None

    def toggle_editor(self) -> None:
        """Switch editor mode; leaving it re-preprocesses the planners on the edited map."""
        self.editing, self._surf = not self.editing, None
        if self.editing:
            self.msg = "EDITOR: left drag = wall, right drag = erase, [ ] = brush"
        else:
            self._rebuild()

    def paint(self, cell: Cell, wall: bool) -> None:

        r, c, h, b = *cell, self.brush // 2, self.brush
        self.walls[max(0, r - h):max(0, r - h + b), max(0, c - h):max(0, c - h + b)] = wall
        self.dirty, self._surf = True, None

    def save_map(self, path: str | Path) -> None:
        """Write the current map (including pending edits) to a ``.map`` file.

        Args:
            path (str | Path): Destination file.

        Returns:
            None.
        """
        self._rebuild()
        write_map(self.grid, path)
        self.msg = f"Saved {path}"

    # drawing and loop
    def draw(self) -> None:

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
                d = np.where(w, 0, w.size)
                while True:
                    p = np.pad(d, 1, constant_values=w.size)
                    n = np.minimum(d, np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]) + 1)
                    if (n == d).all():
                        break
                    d = n
                t = (d / max(d[~w].max(), 1))[..., None]
                img[~w] = (t * (253, 231, 37) + (1 - t) * (68, 1, 84))[~w]
            if st in ("path", "compare"):
                tint(self.baseline_explored if st == "compare" else [], FLAT, .45)
                tint(self.explored, HIER, .5)
            self._cen, self._adj, self._route = {}, [], []
            if lab is not None and st in ("graph", "route"):
                self._cen = {int(a): np.argwhere(lab == a).mean(0) for a in np.unique(lab[lab >= 0])}
                p = np.concatenate([np.c_[lab[:, :-1].ravel(), lab[:, 1:].ravel()],
                                    np.c_[lab[:-1].ravel(), lab[1:].ravel()]])
                p = p[(p[:, 0] != p[:, 1]) & (p.min(1) >= 0)]
                self._adj = np.unique(np.sort(p, 1), axis=0).tolist()
            if lab is not None and st == "route" and path(self.result):
                ids = [int(lab[c]) for c in path(self.result)]
                self._route = [a for i, a in enumerate(ids) if i == 0 or a != ids[i - 1]]
                dim = ~np.isin(lab, self._route) & ~w
                img[dim] = img[dim] * .3 + 190 * .7
            self._surf = _scale(img, px)
        scr.blit(self._surf, (self.ox, self.oy))

        for a, b in self._adj:
            line([self._cen[a], self._cen[b]], (60, 60, 70), 2)
        for c in self._cen.values():
            pygame.draw.circle(scr, (20, 20, 24), pt(c), max(4, px + 1))
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
        out = [(self.method, HIER), (f"Stage {self.stage + 1}/{len(STAGES)}: {st}", TXT), ("", TXT),
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
        y = max(y + 10, H - 150)
        for text in (self.msg[:36], "", "L/R click: start / goal", "M click: toggle wall",
                     "arrows/wheel: stage", "E editor  [ ] brush", "S save map  F figure", "ESC quit"):
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
            elif k == pygame.K_LEFTBRACKET:
                self.brush = max(1, self.brush - 1)
            elif k == pygame.K_RIGHTBRACKET:
                self.brush = min(32, self.brush + 1)
            elif k == pygame.K_s and self.grid:
                self.save_map(free(("map_edited", ".map")))
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

        while all([self.handle_event(e) for e in pygame.event.get()]):
            self.draw()
            self.clock.tick(60)


def save_figure(grid: Grid, labels: np.ndarray | None, path_cells: list[Cell] | None,
                out: str | Path, result: SearchResult | None = None) -> Path:

    out = Path(out) if Path(out).suffix else Path(out).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    walls = np.array(grid.blocked, bool)
    px = max(1, min(1600 // walls.shape[1], 1200 // walls.shape[0]))
    surf = _scale(_base(walls, None if labels is None else np.asarray(labels).astype(int)), px)
    cells = path_cells or (result.path if result and result.success else None) or []
    pt = lambda c: ((c[1] + .5) * px, (c[0] + .5) * px)
    if len(cells) > 1:
        pygame.draw.lines(surf, HIER_PATH, False, [pt(c) for c in cells], max(2, px // 3))
        for c, col in ((cells[0], START), (cells[-1], GOAL)):
            pygame.draw.circle(surf, col, pt(c), max(3, px))
    pygame.image.save(surf, str(out))
    return out


def main(argv: list[str] | None = None) -> int:

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
