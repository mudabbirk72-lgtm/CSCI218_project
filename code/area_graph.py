from __future__ import annotations
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from grid import Cell, SQRT2
from scipy.ndimage import find_objects
import numpy as np


@dataclass
class Gap:
    a: int
    b: int
    pairs: list[tuple[Cell, Cell]]


def find_crossings(
    blocked: np.ndarray,
    labels: np.ndarray,
) -> dict[tuple[int, int], list[tuple[Cell, Cell]]]:
    height, width = blocked.shape
    crossings: dict[tuple[int, int], list[tuple[Cell, Cell]]] = {}

    for row in range(height):
        for col in range(width):
            if blocked[row, col]:
                continue

            cell = (row, col)
            area = int(labels[row, col])

            for nr, nc in ((row, col + 1), (row + 1, col)):
                if nr >= height or nc >= width:
                    continue
                if blocked[nr, nc]:
                    continue

                neighbour_area = int(labels[nr, nc])
                if area == neighbour_area:
                    continue

                other_cell = (nr, nc)
                key = (min(area, neighbour_area), max(area, neighbour_area))

                if area < neighbour_area:
                    pair = (cell, other_cell)
                else:
                    pair = (other_cell, cell)

                crossings.setdefault(key, []).append(pair)

    for pairs in crossings.values():
        pairs.sort(key=lambda pair: pair[0])

    return dict(sorted(crossings.items()))


def find_gaps(
    crossings: dict[tuple[int, int], list[tuple[Cell, Cell]]],
) -> list[Gap]:
    gaps: list[Gap] = []

    for (a, b), pairs in sorted(crossings.items()):
        if not pairs:
            continue

        ordered_pairs = sorted(pairs, key=lambda pair: pair[0])
        remaining = set(range(len(ordered_pairs)))

        while remaining:
            first = min(remaining, key=lambda i: ordered_pairs[i][0])
            remaining.remove(first)
            component = [first]
            stack = [first]

            while stack:
                current = stack.pop()
                row, col = ordered_pairs[current][0]

                neighbours = {
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                }

                connected = [
                    i
                    for i in remaining
                    if ordered_pairs[i][0] in neighbours
                ]

                for i in connected:
                    remaining.remove(i)
                    component.append(i)
                    stack.append(i)

            gap_pairs = sorted(
                (ordered_pairs[i] for i in component),
                key=lambda pair: pair[0],
            )
            gaps.append(Gap(a, b, gap_pairs))

    gaps.sort(key=lambda gap: (gap.a, gap.b, gap.pairs[0][0]))
    return gaps


def _octile(a: tuple[float, float], b: tuple[float, float]) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (SQRT2 - 1) * min(dr, dc)


class AreaGraph:

    def __init__(self, blocked: np.ndarray, labels: np.ndarray, cost_mode: str = "centroid") -> None:
        if cost_mode not in ("hop", "centroid"):
            raise ValueError(
                f"Unknown cost_mode {cost_mode!r}; expected 'hop' or 'centroid'"
            )

        blocked = np.asarray(blocked).astype(bool)
        labels = np.asarray(labels)

        self.labels = labels
        self.cost_mode = cost_mode

        open_mask = labels >= 0
        k = int(labels.max()) + 1 if open_mask.any() else 0
        self.num_areas = k

        
        rows, cols = np.nonzero(open_mask)
        ids = labels[rows, cols].astype(np.int64)
        counts = np.bincount(ids, minlength=k)
        safe = np.maximum(counts, 1)
        mean_r = np.bincount(ids, weights=rows, minlength=k) / safe
        mean_c = np.bincount(ids, weights=cols, minlength=k) / safe
        self.sizes = {i: int(counts[i]) for i in range(k)}
        self.centroids = {i: (float(mean_r[i]), float(mean_c[i])) for i in range(k)}

        
        shifted = np.where(open_mask, labels.astype(np.int64) + 1, 0)
        self.bboxes = {}
        for i, sl in enumerate(find_objects(shifted, max_label=k)):
            self.bboxes[i] = (
                int(sl[0].start), int(sl[1].start),
                int(sl[0].stop) - 1, int(sl[1].stop) - 1,
            )

        
        self.crossings = find_crossings(blocked, labels)
        self.edges = {}
        for a, b in self.crossings:
            if cost_mode == "hop":
                self.edges[(a, b)] = 1.0
            else:
                self.edges[(a, b)] = _octile(self.centroids[a], self.centroids[b])

        
        self._adj = {i: [] for i in range(k)}
        for (a, b), cost in sorted(self.edges.items()):
            self._adj[a].append((b, cost))
            self._adj[b].append((a, cost))

    def area_of(self, cell: Cell) -> int:
        return int(self.labels[cell[0], cell[1]])

    def neighbours(self, area: int) -> Iterator[tuple[int, float]]:
         return iter(self._adj[area])

    def heuristic(self, goal_area: int) -> Callable[[int], float]:
        if self.cost_mode == "hop":
            return lambda area: 0.0

        centroids = self.centroids
        goal = centroids[goal_area]

        def h(area: int) -> float:
            return _octile(centroids[area], goal)

        return h
