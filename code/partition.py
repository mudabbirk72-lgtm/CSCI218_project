

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def validate(blocked: np.ndarray, labels: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def fixed_blocks(blocked: np.ndarray, block_size: int = 16) -> np.ndarray:
    raise NotImplementedError


def distance_map(blocked: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def watershed(blocked: np.ndarray, min_seed_distance: int = 2,
              min_area_size: int = 4) -> np.ndarray:
    raise NotImplementedError


def louvain(blocked: np.ndarray, coarsen: int = 4, resolution: float = 1.0,
            seed: int = 0) -> np.ndarray:
    raise NotImplementedError


def get_partitioner(name: str) -> Callable[..., np.ndarray]:
    raise NotImplementedError
