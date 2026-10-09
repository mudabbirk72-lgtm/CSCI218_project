
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from hierpath.grid import Cell


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



class AreaGraph:

    def __init__(self, blocked: np.ndarray, labels: np.ndarray, cost_mode: str = "centroid") -> None:
        raise NotImplementedError

    def area_of(self, cell: Cell) -> int:
        raise NotImplementedError

    def neighbours(self, area: int) -> Iterator[tuple[int, float]]:
        raise NotImplementedError

    def heuristic(self, goal_area: int) -> Callable[[int], float]:
        raise NotImplementedError
