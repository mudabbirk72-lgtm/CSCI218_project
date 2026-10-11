from __future__ import annotations
from collections import deque
import numpy as np


try:  # optional fast path
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover - scipy is optional
    _ndimage = None

# connectivity

def keep_largest_region(blocked: np.ndarray) -> np.ndarray:

    blocked = np.asarray(blocked, dtype=bool)
    if blocked.ndim != 2:
        raise ValueError("blocked must be a 2-D array")

    open_mask = ~blocked
    if not open_mask.any():
        return blocked.copy()

    if _ndimage is not None:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
        labels, n_labels = _ndimage.label(open_mask, structure=structure)
        if n_labels <= 1:
            return blocked.copy()
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # background = walls
        best = int(counts.argmax())
        return labels != best

    # pure-python fallback: flood fill each open component once

    h, w = blocked.shape
    flat_open = open_mask.ravel()
    label = np.full(h * w, -1, dtype=np.int64)
    sizes: list[int] = []
    cur = 0

    for start in range(h * w):
        if not flat_open[start] or label[start] != -1:
            continue
        label[start] = cur
        queue = deque((start,))
        count = 0
        while queue:
            p = queue.popleft()
            count += 1
            r, c = divmod(p, w)
            if r > 0:
                q = p - w
                if flat_open[q] and label[q] == -1:
                    label[q] = cur
                    queue.append(q)
            if r < h - 1:
                q = p + w
                if flat_open[q] and label[q] == -1:
                    label[q] = cur
                    queue.append(q)
            if c > 0:
                q = p - 1
                if flat_open[q] and label[q] == -1:
                    label[q] = cur
                    queue.append(q)
            if c < w - 1:
                q = p + 1
                if flat_open[q] and label[q] == -1:
                    label[q] = cur
                    queue.append(q)
        sizes.append(count)
        cur += 1

    if cur <= 1:
        return blocked.copy()

    best = int(np.argmax(sizes))
    return (label != best).reshape(h, w)

# random scatter

def random_map(size: int, density: float, seed: int) -> np.ndarray:

    size = int(size)
    density = float(density)
    if size < 8:
        raise ValueError(f"size must be >= 8, got {size}")
    if not (0.0 <= density <= 0.4):
        raise ValueError(f"density must be in [0.0, 0.4], got {density}")

    rng = np.random.default_rng(seed)

    grid = np.zeros((size, size), dtype=bool)
    # walled border
    grid[0, :] = True
    grid[-1, :] = True
    grid[:, 0] = True
    grid[:, -1] = True

    inner = size - 2
    n_inner = inner * inner
    n_walls = int(round(density * n_inner))
    if n_walls > 0:
        picks = rng.choice(n_inner, size=n_walls, replace=False)
        rows, cols = np.divmod(picks, inner)
        grid[rows + 1, cols + 1] = True

    return keep_largest_region(grid)

# office-like rooms

def _carve_room(grid, rng, r0, c0, r1, c1, min_room, max_room):

    h = r1 - r0 + 1
    w = c1 - c0 + 1

    # Too small to hold a room with its own walls: donate the block to the corridors.
    if h - 2 < min_room or w - 2 < min_room:
        grid[r0:r1 + 1, c0:c1 + 1] = False
        return

    room_h = min(max_room, h - 2)
    room_w = min(max_room, w - 2)
    rr0 = r0 + 1 + int(rng.integers(0, h - 2 - room_h + 1))
    cc0 = c0 + 1 + int(rng.integers(0, w - 2 - room_w + 1))
    rr1 = rr0 + room_h - 1
    cc1 = cc0 + room_w - 1
    grid[rr0:rr1 + 1, cc0:cc1 + 1] = False

    # One door, 1 cell wide: step out from a random point on a random wall until
    # the first already-open cell (a corridor) is reached.
    height, width = grid.shape
    for side in rng.permutation(4):
        if side == 0:      # north
            dr, dc = -1, 0
            r, c = rr0 - 1, int(rng.integers(cc0, cc1 + 1))
        elif side == 1:    # south
            dr, dc = 1, 0
            r, c = rr1 + 1, int(rng.integers(cc0, cc1 + 1))
        elif side == 2:    # west
            dr, dc = 0, -1
            r, c = int(rng.integers(rr0, rr1 + 1)), cc0 - 1
        else:              # east
            dr, dc = 0, 1
            r, c = int(rng.integers(rr0, rr1 + 1)), cc1 + 1

        tunnel = []
        rr, cc = r, c
        while 0 < rr < height - 1 and 0 < cc < width - 1:
            if not grid[rr, cc]:          # reached a corridor
                for tr, tc in tunnel:
                    grid[tr, tc] = False
                return
            tunnel.append((rr, cc))
            rr += dr
            cc += dc
        # ran into the border: try another side


def _split_block(grid, rng, r0, c0, r1, c1, min_room, max_room, corridor_width):

    h = r1 - r0 + 1
    w = c1 - c0 + 1
    min_block = min_room + 2  # room plus its own walls

    can_rows = h >= 2 * min_block + corridor_width
    can_cols = w >= 2 * min_block + corridor_width
    if not (can_rows or can_cols):
        _carve_room(grid, rng, r0, c0, r1, c1, min_room, max_room)
        return

    oversized = (h > max_room + 2) or (w > max_room + 2)
    if not oversized and rng.random() < 0.3:
        _carve_room(grid, rng, r0, c0, r1, c1, min_room, max_room)
        return

    if can_rows and can_cols:
        split_rows = h >= w if h != w else bool(rng.integers(0, 2))
    else:
        split_rows = can_rows

    if split_rows:
        lo = r0 + min_block
        hi = r1 - min_block - corridor_width + 1
        band = int(rng.integers(lo, hi + 1))
        grid[band:band + corridor_width, c0:c1 + 1] = False
        _split_block(grid, rng, r0, c0, band - 1, c1,
                     min_room, max_room, corridor_width)
        _split_block(grid, rng, band + corridor_width, c0, r1, c1,
                     min_room, max_room, corridor_width)
    else:
        lo = c0 + min_block
        hi = c1 - min_block - corridor_width + 1
        band = int(rng.integers(lo, hi + 1))
        grid[r0:r1 + 1, band:band + corridor_width] = False
        _split_block(grid, rng, r0, c0, r1, band - 1,
                     min_room, max_room, corridor_width)
        _split_block(grid, rng, r0, band + corridor_width, r1, c1,
                     min_room, max_room, corridor_width)


def room_map(size: int, seed: int, min_room: int = 6, max_room: int = 20,
             corridor_width: int = 3) -> np.ndarray:

    size = int(size)
    min_room = int(min_room)
    max_room = int(max_room)
    corridor_width = int(corridor_width)

    if size < 32:
        raise ValueError(f"size must be >= 32, got {size}")
    if min_room > max_room:
        raise ValueError(f"min_room ({min_room}) must be <= max_room ({max_room})")
    min_room = max(min_room, 3)
    max_room = max(max_room, min_room)
    corridor_width = max(corridor_width, 1)

    rng = np.random.default_rng(seed)
    grid = np.ones((size, size), dtype=bool)

    # A ring of corridor just inside the border, so every block is reachable from
    # at least one side and all corridor bands end up joined.
    cw = min(corridor_width, (size - 2) // 2)
    grid[1:1 + cw, 1:size - 1] = False
    grid[size - 1 - cw:size - 1, 1:size - 1] = False
    grid[1:size - 1, 1:1 + cw] = False
    grid[1:size - 1, size - 1 - cw:size - 1] = False

    r0 = c0 = 1 + cw
    r1 = c1 = size - 2 - cw
    if r0 <= r1 and c0 <= c1:
        _split_block(grid, rng, r0, c0, r1, c1, min_room, max_room, corridor_width)

    # Border stays walled whatever the carving did.
    grid[0, :] = True
    grid[-1, :] = True
    grid[:, 0] = True
    grid[:, -1] = True

    return keep_largest_region(grid)

# maze

def maze_map(size: int, seed: int, corridor_width: int = 1) -> np.ndarray:

    size = int(size)
    corridor_width = int(corridor_width)

    if size < 16:
        raise ValueError(f"size must be >= 16, got {size}")
    if corridor_width < 1:
        raise ValueError(f"corridor_width must be >= 1, got {corridor_width}")

    cw = corridor_width
    # n coarse cells need cw * (2n + 1) real cells: wall, cell, wall, cell, ..., wall.
    n = (size // cw - 1) // 2
    if n < 2:
        raise ValueError(
            f"size {size} is too small for a 2x2 maze at corridor_width {cw}; "
            f"need at least {5 * cw}"
        )

    rng = np.random.default_rng(seed)
    grid = np.ones((size, size), dtype=bool)

    def block(i, j):
        """Real-cell slice of coarse cell (i, j)."""
        r = cw * (2 * i + 1)
        c = cw * (2 * j + 1)
        return slice(r, r + cw), slice(c, c + cw)

    # randomised depth-first search (iterative, explicit stack)
    visited = np.zeros((n, n), dtype=bool)
    start = (int(rng.integers(0, n)), int(rng.integers(0, n)))
    visited[start] = True
    rs, cs = block(*start)
    grid[rs, cs] = False
    stack = [start]

    while stack:
        i, j = stack[-1]
        neighbours = []
        if i > 0 and not visited[i - 1, j]:
            neighbours.append((i - 1, j))
        if i < n - 1 and not visited[i + 1, j]:
            neighbours.append((i + 1, j))
        if j > 0 and not visited[i, j - 1]:
            neighbours.append((i, j - 1))
        if j < n - 1 and not visited[i, j + 1]:
            neighbours.append((i, j + 1))

        if not neighbours:
            stack.pop()
            continue

        ni, nj = neighbours[int(rng.integers(0, len(neighbours)))]

        # open the wall band between (i, j) and (ni, nj)
        if ni == i:  # horizontal step
            jj = min(j, nj)
            r = cw * (2 * i + 1)
            c = cw * (2 * jj + 2)
            grid[r:r + cw, c:c + cw] = False
        else:        # vertical step
            ii = min(i, ni)
            r = cw * (2 * ii + 2)
            c = cw * (2 * j + 1)
            grid[r:r + cw, c:c + cw] = False

        rs, cs = block(ni, nj)
        grid[rs, cs] = False
        visited[ni, nj] = True
        stack.append((ni, nj))

    return grid
