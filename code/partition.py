from __future__ import annotations
from collections.abc import Callable
from scipy.ndimage import find_objects, generate_binary_structure
from scipy.ndimage import label as nd_label
import numpy as np


def validate(blocked: np.ndarray, labels: np.ndarray) -> np.ndarray:
    blocked = np.asarray(blocked)
    labels = np.asarray(labels)

    if blocked.shape != labels.shape:
        raise ValueError(
            f"Shapes differ: blocked {blocked.shape}, labels {labels.shape}"
        )

    open_cells = ~blocked.astype(bool)
    if np.any(open_cells & (labels == -1)):
        raise ValueError("An open cell has label -1")

    
    out = np.full(labels.shape, -1, dtype=np.int32)
    if not open_cells.any():
        return out

    structure = generate_binary_structure(2, 1)

    
    _, inverse = np.unique(labels[open_cells], return_inverse=True)
    dense = np.zeros(labels.shape, dtype=np.int64)
    dense[open_cells] = inverse.reshape(-1) + 1

    next_id = 0  
    for idx, sl in enumerate(find_objects(dense), start=1):
        if sl is None:
            continue
      
        comps, n = nd_label(dense[sl] == idx, structure=structure)
        view = out[sl]
        inside = comps > 0
        view[inside] = comps[inside] - 1 + next_id
        next_id += n

    return out



def fixed_blocks(blocked: np.ndarray, block_size: int = 16) -> np.ndarray:
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")

    h, w = blocked.shape
    r, c = np.indices((h, w))
    blocks_across = (w + block_size - 1) // block_size  
    proposed = (r // block_size) * blocks_across + (c // block_size)
    return validate(blocked, proposed)



def distance_map(blocked: np.ndarray) -> np.ndarray:
    blocked = np.asarray(blocked).astype(bool)
    open_cells = ~blocked

    if not open_cells.any():
        return np.zeros(blocked.shape, dtype=np.int32)

    if not blocked.any():
      
        padded = np.pad(open_cells, 1, constant_values=False)
        dist = distance_transform_cdt(padded, metric="chessboard")
        return dist[1:-1, 1:-1].astype(np.int32)

    return distance_transform_cdt(open_cells, metric="chessboard").astype(np.int32)

def _border_counts(labels: np.ndarray) -> dict[tuple[int, int], int]:

    los, his = [], []
    for x, y in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (x >= 0) & (y >= 0) & (x != y)
        los.append(np.minimum(x[m], y[m]).astype(np.int64))
        his.append(np.maximum(x[m], y[m]).astype(np.int64))
    lo = np.concatenate(los)
    hi = np.concatenate(his)
    if lo.size == 0:
        return {}
    base = int(hi.max()) + 1
    keys, counts = np.unique(lo * base + hi, return_counts=True)
    return {(int(k) // base, int(k) % base): int(n) for k, n in zip(keys, counts)}


def _merge_small_areas(labels: np.ndarray, num: int, min_size: int) -> np.ndarray:
   
    sizes = np.bincount(labels[labels >= 0], minlength=num).astype(np.int64).tolist()
    adj: list[dict[int, int]] = [dict() for _ in range(num)]
    for (a, b), n in _border_counts(labels).items():
        adj[a][b] = n
        adj[b][a] = n

    parent = list(range(num))
    alive = [True] * num
    heap = [(s, i) for i, s in enumerate(sizes) if s < min_size]
    heapq.heapify(heap)

    while heap:
        size, a = heapq.heappop(heap)
        if not alive[a] or size != sizes[a]:
            continue  
        if not adj[a]:
            continue 
        nbrs = adj[a]
        t = max(nbrs, key=lambda n: (nbrs[n], -n))  

        for n, cnt in nbrs.items():
            if n == t:
                continue
            adj[t][n] = adj[t].get(n, 0) + cnt
            adj[n][t] = adj[n].get(t, 0) + cnt
            del adj[n][a]
        del adj[t][a]
        adj[a] = {}

        alive[a] = False
        parent[a] = t
        sizes[t] += size
        if sizes[t] < min_size:
            heapq.heappush(heap, (sizes[t], t))

    lookup = np.arange(num, dtype=np.int64)
    for i in range(num):
        r = i
        while parent[r] != r:
            r = parent[r]
        lookup[i] = r

    out = labels.astype(np.int64)
    open_cells = out >= 0
    out[open_cells] = lookup[out[open_cells]]
    return out



def watershed(blocked: np.ndarray, min_seed_distance: int = 2,
              min_area_size: int = 4) -> np.ndarray:
    blocked = np.asarray(blocked).astype(bool)
    open_cells = ~blocked
    if not open_cells.any():
        return validate(blocked, np.full(blocked.shape, -1, dtype=np.int32))

    four = generate_binary_structure(2, 1)
    eight = generate_binary_structure(2, 2)

    
    dist = distance_map(blocked)

   
    maxima = local_maxima(dist, connectivity=2).astype(bool)
    maxima &= open_cells & (dist >= min_seed_distance)
    seeds, num = nd_label(maxima, structure=eight)

    
    regions, num_regions = nd_label(open_cells, structure=four)
    seeded = np.zeros(num_regions + 1, dtype=bool)
    seeded[regions[seeds > 0]] = True
    missing = (np.flatnonzero(~seeded[1:]) + 1).tolist()
    if missing:
        for r, c in maximum_position(dist, regions, missing):
            num += 1
            seeds[r, c] = num

   
    flooded = sk_watershed(-dist.astype(np.float64), markers=seeds, mask=open_cells)
    labels = flooded.astype(np.int64) - 1 

    
    if min_area_size > 1:
        labels = _merge_small_areas(labels, num, min_area_size)

   
    return validate(blocked, labels)


def louvain(blocked: np.ndarray, coarsen: int = 4, resolution: float = 1.0,
            seed: int = 0) -> np.ndarray:
        import networkx as nx

    if coarsen < 1:
        raise ValueError(f"coarsen must be >= 1, got {coarsen}")
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    blocked = np.asarray(blocked).astype(bool)
    h, w = blocked.shape
    if blocked.all():
        return validate(blocked, np.full(blocked.shape, -1, dtype=np.int32))

   

    r, c = np.indices((h, w))
    rr = r + r // coarsen
    cc = c + c // coarsen
    spread = np.zeros((int(rr.max()) + 1, int(cc.max()) + 1), dtype=bool)
    spread[rr, cc] = ~blocked
    pieces, k = nd_label(spread, structure=generate_binary_structure(2, 1))
    nodes = np.where(blocked, -1, pieces[rr, cc] - 1).astype(np.int64)

    graph = nx.Graph()
    graph.add_nodes_from(range(k))
    graph.add_weighted_edges_from(
        (a, b, n) for (a, b), n in _border_counts(nodes).items()
    )

    
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=seed
    )

    
    community_of = np.zeros(k, dtype=np.int64)
    for i, members in enumerate(communities):
        community_of[list(members)] = i
    labels = np.where(nodes >= 0, community_of[np.maximum(nodes, 0)], -1)
    return validate(blocked, labels)


def get_partitioner(name: str) -> Callable[..., np.ndarray]:
     methods = {
        "fixed": fixed_blocks,
        "watershed": watershed,
        "louvain": louvain,
    }
    try:
        return methods[name]
    except (KeyError, TypeError):
        raise KeyError(
            f"Unknown partition method {name!r}; expected one of {sorted(methods)}"
        ) from None
