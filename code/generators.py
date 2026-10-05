
from __future__ import annotations

import numpy as np


def keep_largest_region(blocked: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def random_map(size: int, density: float, seed: int) -> np.ndarray:
    raise NotImplementedError


def room_map(size: int, seed: int, min_room: int = 6, max_room: int = 20,
             corridor_width: int = 3) -> np.ndarray:
    raise NotImplementedError


def maze_map(size: int, seed: int, corridor_width: int = 1) -> np.ndarray:
    raise NotImplementedError
