"""Tests for Khizar's, Maysoon's and Zoren's work (every task up to Week 5A).

    Khizar   search.py, two_tier.py, analysis.py (breakeven_queries, plot_breakeven, plot_repair)
    Maysoon  area_graph.py (find_crossings, find_gaps), hpa.py
    Zoren    generators.py, app.py

Run from the project folder (the one that contains ``code/`` and ``tests/``):

    python -m pytest tests/test_team_tasks.py -v

A function that still raises NotImplementedError shows as SKIPPED ("not implemented
yet"), not FAILED, so this file can be run while work is still in progress. Each
section header says whose code it checks. The planner tests also use Riya's
partitions and AreaGraph, so a failure there can come from her code: read the
error message before blaming the planner.
"""

from __future__ import annotations

import copy
import functools
import heapq
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from grid import SQRT2, Grid  # noqa: E402

SEED = 7


# ============================================================================ helpers


def true_cost(grid: Grid, start, goal, allowed=None, labels=None) -> float:
    """Independent Dijkstra (does not use search.py). Optionally restricted to areas."""
    dist, heap = {start: 0.0}, [(0.0, start)]
    while heap:
        d, cell = heapq.heappop(heap)
        if cell == goal:
            return d
        if d > dist[cell]:
            continue
        for nxt, cost in grid.neighbours(cell):
            if allowed is not None and int(labels[nxt]) not in allowed:
                continue
            if d + cost < dist.get(nxt, math.inf):
                dist[nxt] = d + cost
                heapq.heappush(heap, (d + cost, nxt))
    return math.inf


def office(rooms: int = 3, room: int = 10) -> Grid:
    """A rooms x rooms block of square rooms, each joined to its neighbours by doors.

    Built here (not with generators.py) so planner tests do not depend on Zoren's code.
    """
    n = rooms * (room + 1) + 1
    b = np.ones((n, n), dtype=bool)
    for i in range(rooms):
        for j in range(rooms):
            r0, c0 = i * (room + 1) + 1, j * (room + 1) + 1
            b[r0:r0 + room, c0:c0 + room] = False
            if j + 1 < rooms:                                # door to the right (2 wide)
                b[r0 + room // 2: r0 + room // 2 + 2, c0 + room] = False
            if i + 1 < rooms:                                # door below (1 wide)
                b[r0 + room, c0 + (i + j) % (room - 2) + 1] = False
    return Grid(b)


def open_cells(grid: Grid) -> list:
    return [(int(r), int(c)) for r, c in np.argwhere(~grid.blocked)]


def query_pairs(grid: Grid, n: int, seed: int = SEED) -> list:
    rng = np.random.default_rng(seed)
    cells = open_cells(grid)
    pairs = []
    while len(pairs) < n:
        a, b = (cells[i] for i in rng.choice(len(cells), 2, replace=False))
        pairs.append((a, b))
    return pairs


def connected(blocked: np.ndarray) -> bool:
    _, n = ndi.label(~blocked)
    return n == 1


def border_is_wall(blocked: np.ndarray) -> bool:
    return bool(blocked[0].all() and blocked[-1].all() and blocked[:, 0].all() and blocked[:, -1].all())


def check_result_path(grid: Grid, res, start, goal, who: str):
    assert res.success, f"{who}: no path found from {start} to {goal}"
    assert res.path[0] == start and res.path[-1] == goal, f"{who}: path must run from start to goal"
    assert grid.is_valid_path(res.path), f"{who}: path goes through a wall or cuts a corner"
    assert math.isclose(res.cost, grid.path_cost(res.path), rel_tol=1e-9, abs_tol=1e-9), (
        f"{who}: reported cost {res.cost} but the path costs {grid.path_cost(res.path)}")


# ============================================================================ KHIZAR: search.py

import search  # noqa: E402


def test_octile_and_manhattan():
    assert search.octile((0, 0), (3, 4)) == pytest.approx(3 * SQRT2 + 1)
    assert search.octile((3, 4), (0, 0)) == pytest.approx(3 * SQRT2 + 1)
    assert search.octile((2, 2), (2, 2)) == 0
    assert search.manhattan((0, 0), (3, 4)) == pytest.approx(7.0)
    assert search.manhattan((5, 1), (2, 3)) == pytest.approx(5.0)


def test_astar_finds_optimal_paths():
    g = office()
    for s, t in query_pairs(g, 15):
        res = search.astar(s, {t}, g.neighbours, lambda c, t=t: search.octile(c, t))
        check_result_path(g, res, s, t, "astar")
        assert res.cost == pytest.approx(true_cost(g, s, t)), "astar: path is not the shortest"
        assert res.expanded > 0 and res.generated > 0 and res.runtime_s >= 0


def test_astar_start_is_goal_and_unreachable():
    g = Grid.from_strings(["#######", "#..#..#", "#######"])
    same = search.astar((1, 1), {(1, 1)}, g.neighbours, lambda c: 0.0)
    assert same.success and same.path == [(1, 1)] and same.cost == 0.0
    none = search.astar((1, 1), {(1, 4)}, g.neighbours, lambda c: 0.0)
    assert not none.success and none.path is None and math.isinf(none.cost)


def test_astar_on_expand_and_any_goal():
    g = office()
    seen = []
    res = search.astar((1, 1), {(5, 5), (2, 2)}, g.neighbours, lambda c: 0.0, seen.append)
    assert res.path[-1] == (2, 2), "astar should stop at the nearer of several goals"
    assert len(seen) == res.expanded, "on_expand must be called once per expanded node"


def test_astar_works_on_any_graph():
    graph = {"A": [("B", 1.0), ("C", 4.0)], "B": [("C", 1.0)], "C": []}
    res = search.astar("A", {"C"}, lambda n: graph[n], lambda n: 0.0)
    assert res.path == ["A", "B", "C"] and res.cost == pytest.approx(2.0)


def test_astar_expands_no_more_than_dijkstra():
    g = office()
    s, t = (1, 1), (g.height - 2, g.width - 2)
    a = search.astar(s, {t}, g.neighbours, lambda c: search.octile(c, t))
    d = search.dijkstra(s, {t}, g.neighbours)
    assert a.cost == pytest.approx(d.cost)
    assert a.expanded <= d.expanded


def test_dijkstra_all_and_reconstruct_path():
    g = office()
    start = (1, 1)
    dist, parent = search.dijkstra_all(start, g.neighbours)
    assert dist[start] == 0.0 and parent[start] is None
    assert len(dist) == len(open_cells(g)), "every open cell is reachable on this map"
    for s, t in query_pairs(g, 5):
        assert dist[t] == pytest.approx(true_cost(g, start, t))
        path = search.reconstruct_path(parent, t)
        assert path[0] == start and path[-1] == t and g.is_valid_path(path)
        assert g.path_cost(path) == pytest.approx(dist[t])
    with pytest.raises(KeyError):
        search.reconstruct_path(parent, (0, 0))              # a wall: never reached


def test_dijkstra_all_targets_stop_early():
    g = office()
    near = {(2, 2), (3, 3)}
    dist, _ = search.dijkstra_all((1, 1), g.neighbours, targets=near)
    for t in near:
        assert dist[t] == pytest.approx(true_cost(g, (1, 1), t))
    assert len(dist) < len(open_cells(g)), "with targets the search should stop early"


# ---------------------------------------------------------------- FlatPlanner (tasks 4, 25)


def test_flat_planner_basics():
    with pytest.raises(ValueError):
        search.FlatPlanner(algorithm="bfs")
    with pytest.raises(ValueError):
        search.FlatPlanner(heuristic="euclid")
    g = office()
    p = search.FlatPlanner()
    with pytest.raises(RuntimeError):
        p.query((1, 1), (2, 2))                              # before preprocess
    assert p.preprocess(g) == pytest.approx(0.0, abs=0.01)
    with pytest.raises(ValueError):
        p.query((0, 0), (2, 2))                              # start is a wall
    assert p.stats() == {"areas": 0, "doorways": 0, "preprocess_s": 0.0}


@pytest.mark.parametrize("algorithm", ["astar", "dijkstra"])
def test_flat_planner_is_optimal(algorithm):
    g = office()
    p = search.FlatPlanner(algorithm=algorithm)
    p.preprocess(g)
    for s, t in query_pairs(g, 10):
        res = p.query(s, t)
        check_result_path(g, res, s, t, f"FlatPlanner({algorithm})")
        assert res.cost == pytest.approx(true_cost(g, s, t))


def test_flat_planner_update():
    g = office().copy()
    p = search.FlatPlanner()
    p.preprocess(g)
    s, t = (1, 1), (1, 10)
    door = p.query(s, t).path[5]
    assert isinstance(p.update(door, True), float)
    assert not g.is_free(door), "update(cell, True) must turn the cell into a wall"
    res = p.query(s, t)
    check_result_path(g, res, s, t, "FlatPlanner after update")
    assert door not in res.path
    p.update(door, False)
    assert g.is_free(door), "update(cell, False) must open the cell again"


# ============================================================================ KHIZAR: two_tier.py

PARTITION_PARAMS = {"fixed": {"block_size": 8}, "watershed": {}}


def make_two_tier(partition="watershed", refine="nearest", grid=None):
    from two_tier import TwoTierPlanner
    p = TwoTierPlanner(partition=partition, refine=refine,
                       partition_params=PARTITION_PARAMS.get(partition))
    p.preprocess(grid if grid is not None else office())
    return p


def test_two_tier_rejects_bad_options():
    from two_tier import TwoTierPlanner
    for kwargs in ({"partition": "nope"}, {"refine": "nope"}, {"cost_mode": "nope"},
                   {"heuristic": "nope"}):
        with pytest.raises(ValueError):
            TwoTierPlanner(**kwargs)


@pytest.mark.parametrize("partition,refine,name", [
    ("watershed", "nearest", "Basic-watershed"), ("fixed", "corridor", "Corridor-fixed")])
def test_two_tier_name(partition, refine, name):
    from two_tier import TwoTierPlanner
    assert TwoTierPlanner(partition=partition, refine=refine).name == name


def test_two_tier_needs_preprocess_and_open_cells():
    from two_tier import TwoTierPlanner
    p = TwoTierPlanner()
    with pytest.raises(RuntimeError):
        p.query((1, 1), (2, 2))
    p = make_two_tier()
    with pytest.raises(ValueError):
        p.query((0, 0), (2, 2))


def test_two_tier_preprocess_sets_attributes():
    g = office()
    p = make_two_tier(grid=g)
    assert p.grid is g and p.area_graph is not None and p.preprocess_s >= 0
    assert p.labels.shape == g.blocked.shape
    assert (p.labels[g.blocked] == -1).all() and (p.labels[~g.blocked] >= 0).all()


@pytest.mark.parametrize("partition", ["fixed", "watershed"])
def test_find_route(partition):
    p = make_two_tier(partition)
    for s, t in query_pairs(p.grid, 10):
        res = p.find_route(s, t)
        assert res.success
        route = res.path
        assert route[0] == p.labels[s] and route[-1] == p.labels[t]
        for a, b in zip(route, route[1:]):
            assert (min(a, b), max(a, b)) in p.area_graph.crossings, (
                f"find_route: areas {a} and {b} are next to each other on the route but do not touch")
    s = (1, 1)
    assert p.find_route(s, s).path == [int(p.labels[s])]


@pytest.mark.parametrize("partition", ["fixed", "watershed"])
def test_refine_nearest_stays_on_route(partition):
    p = make_two_tier(partition)
    for s, t in query_pairs(p.grid, 10):
        route = p.find_route(s, t).path
        seen = []
        res = p.refine_nearest(route, s, t, seen.append)
        check_result_path(p.grid, res, s, t, "refine_nearest")
        assert {int(p.labels[c]) for c in res.path} <= set(route), "refine_nearest left the route's areas"
        assert len(res.extra["legs"]) == len(route), "one leg per area change, plus the final leg"
        assert len(seen) == res.expanded


@pytest.mark.parametrize("partition", ["fixed", "watershed"])
def test_refine_corridor(partition):
    p = make_two_tier(partition, refine="corridor")
    for s, t in query_pairs(p.grid, 10):
        route = p.find_route(s, t).path
        corridor = p.refine_corridor(route, s, t)
        nearest = p.refine_nearest(route, s, t)
        check_result_path(p.grid, corridor, s, t, "refine_corridor")
        assert {int(p.labels[c]) for c in corridor.path} <= set(route), "refine_corridor left the route's areas"
        assert corridor.cost == pytest.approx(true_cost(p.grid, s, t, set(route), p.labels)), (
            "refine_corridor must be optimal among paths inside the route's areas")
        assert corridor.cost <= nearest.cost + 1e-9, "corridor must never be longer than nearest"


@pytest.mark.parametrize("refine", ["nearest", "corridor"])
@pytest.mark.parametrize("partition", ["fixed", "watershed"])
def test_two_tier_query(partition, refine):
    p = make_two_tier(partition, refine)
    for s, t in query_pairs(p.grid, 15):
        res = p.query(s, t)
        check_result_path(p.grid, res, s, t, p.name)
        assert res.cost >= true_cost(p.grid, s, t) - 1e-9, "a path cannot beat the optimum"
        assert "route" in res.extra and "tier1_expanded" in res.extra


def test_two_tier_stats():
    p = make_two_tier()
    st = p.stats()
    assert st["areas"] == int(p.labels.max()) + 1 and st["doorways"] == 0
    assert st["preprocess_s"] == pytest.approx(p.preprocess_s)


def test_two_tier_update_rebuilds():
    g = office().copy()
    p = make_two_tier(grid=g)
    s, t = (1, 1), (g.height - 2, g.width - 2)
    cell = p.query(s, t).path[3]
    assert p.update(cell, True) >= 0
    assert not g.is_free(cell) and p.labels[cell] == -1, "update must change the map and re-partition"
    res = p.query(s, t)
    check_result_path(g, res, s, t, "TwoTierPlanner after update")
    assert cell not in res.path


# ============================================================================ MAYSOON: area_graph.py

import area_graph  # noqa: E402


def two_area_map():
    """5x6 open area split down the middle: left = 0, right = 1. One wall at (2, 2)."""
    blocked = np.zeros((5, 6), dtype=bool)
    blocked[2, 2] = True
    labels = np.where(np.arange(6) < 3, 0, 1)[None, :].repeat(5, axis=0).astype(np.int32)
    labels[blocked] = -1
    return blocked, labels


def test_find_crossings():
    blocked, labels = two_area_map()
    cr = area_graph.find_crossings(blocked, labels)
    assert list(cr) == [(0, 1)]
    assert cr[(0, 1)] == [((r, 2), (r, 3)) for r in (0, 1, 3, 4)], (
        "pairs must be (cell_in_a, cell_in_b), sorted, and skip walls")


def test_find_crossings_orders_area_ids():
    blocked = np.zeros((2, 2), dtype=bool)
    labels = np.array([[5, 2], [5, 2]], dtype=np.int32)
    cr = area_graph.find_crossings(blocked, labels)
    assert cr == {(2, 5): [((0, 1), (0, 0)), ((1, 1), (1, 0))]}


def test_find_gaps_splits_separate_doors():
    blocked, labels = two_area_map()
    gaps = area_graph.find_gaps(area_graph.find_crossings(blocked, labels))
    assert [(g.a, g.b) for g in gaps] == [(0, 1), (0, 1)]
    assert gaps[0].pairs == [((0, 2), (0, 3)), ((1, 2), (1, 3))]
    assert gaps[1].pairs == [((3, 2), (3, 3)), ((4, 2), (4, 3))]


# ============================================================================ MAYSOON: hpa.py

import hpa  # noqa: E402


def gap_of_width(n):
    return area_graph.Gap(0, 1, [((r, 0), (r, 1)) for r in range(n)])


def test_place_doorways():
    assert hpa.place_doorways(gap_of_width(1)) == [((0, 0), (0, 1))]
    middle = hpa.place_doorways(gap_of_width(3))
    assert middle == [((1, 0), (1, 1))], "a narrow opening gets one doorway, in the middle"
    four = hpa.place_doorways(gap_of_width(4))
    assert len(four) == 1 and four[0] in (((1, 0), (1, 1)), ((2, 0), (2, 1)))
    wide = hpa.place_doorways(gap_of_width(6), max_gap=6)
    assert wide == [((0, 0), (0, 1)), ((5, 0), (5, 1))], "a wide opening gets its first and last pair"


def doorway_graph(partition="watershed"):
    from partition import get_partitioner
    g = office()
    labels = get_partitioner(partition)(g.blocked, **PARTITION_PARAMS.get(partition, {}))
    return g, labels, hpa.DoorwayGraph(g, labels)


def edge_set(dg):
    """Edges by cell, so two graphs can be compared even if node ids differ."""
    out = set()
    for u, nbrs in dg.adj.items():
        for v, cost in nbrs.items():
            out.add((frozenset((dg.nodes[u][1], dg.nodes[v][1])), round(cost, 6)))
    return out


@pytest.mark.parametrize("partition", ["fixed", "watershed"])
def test_doorway_graph_structure(partition):
    g, labels, dg = doorway_graph(partition)
    assert dg.nodes, "no doorways were placed"
    for nid, (area, cell) in dg.nodes.items():
        assert g.is_free(cell) and labels[cell] == area
    for area, ids in dg.by_area.items():
        assert all(dg.nodes[i][0] == area for i in ids)
    for u, nbrs in dg.adj.items():
        assert dict(dg.neighbours(u)) == nbrs
        for v, cost in nbrs.items():
            assert dg.adj[v][u] == cost, "edges must be stored in both directions"
            path = dg.paths[(min(u, v), max(u, v))]
            assert g.is_valid_path(path)
            assert g.path_cost(path) == pytest.approx(cost)
            (au, cu), (av, cv) = dg.nodes[u], dg.nodes[v]
            if au != av:
                assert cost == 1.0, "a crossing edge costs one step"
            else:
                assert {int(labels[c]) for c in path} == {au}, "an inner path must stay in its area"
                assert cost == pytest.approx(true_cost(g, cu, cv, {au}, labels))


def test_doorway_graph_segment_and_heuristic():
    g, labels, dg = doorway_graph()
    u = next(iter(dg.adj))
    v = next(iter(dg.adj[u]))
    fwd, back = dg.segment(u, v), dg.segment(v, u)
    assert fwd[0] == dg.nodes[u][1] and fwd[-1] == dg.nodes[v][1] and back == fwd[::-1]
    unjoined = next(w for w in dg.nodes if w != u and w not in dg.adj[u])
    with pytest.raises(KeyError):
        dg.segment(u, unjoined)
    goal = (g.height - 2, g.width - 2)
    h = dg.heuristic(goal)
    assert h(u) == pytest.approx(search.octile(dg.nodes[u][1], goal))


def test_insert_and_remove_temporary():
    g, labels, dg = doorway_graph()
    before = (copy.deepcopy(dg.nodes), copy.deepcopy(dg.adj), copy.deepcopy(dg.paths))
    cell = (5, 5)
    nid = dg.insert(cell)
    assert nid in dg.temporary and dg.nodes[nid][1] == cell
    area = int(labels[cell])
    for d in dg.by_area.get(area, []):
        assert dg.adj[nid][d] == pytest.approx(true_cost(g, cell, dg.nodes[d][1], {area}, labels))
    dg.remove_temporary()
    assert (dg.nodes, dg.adj, dg.paths) == before, "remove_temporary must restore the graph exactly"
    door_id = next(iter(dg.nodes))
    assert dg.insert(dg.nodes[door_id][1]) == door_id, "inserting a doorway cell returns its id"
    dg.remove_temporary()


def test_rebuild_areas_on_unchanged_map():
    g, labels, dg = doorway_graph()
    before = edge_set(dg)
    count = dg.rebuild_areas({0, 1})
    assert isinstance(count, int) and count >= 0
    assert edge_set(dg) == before, "rebuilding unchanged areas must give the same edges"


def make_hpa(partition="watershed", grid=None):
    from hpa import HPAPlanner
    p = HPAPlanner(partition=partition, partition_params=PARTITION_PARAMS.get(partition))
    p.preprocess(grid if grid is not None else office())
    return p


def test_hpa_options_and_name():
    from hpa import HPAPlanner
    with pytest.raises(ValueError):
        HPAPlanner(partition="nope")
    with pytest.raises(ValueError):
        HPAPlanner(heuristic="nope")
    assert HPAPlanner(partition="fixed").name == "HPA-fixed"
    with pytest.raises(RuntimeError):
        HPAPlanner().query((1, 1), (2, 2))


@pytest.mark.parametrize("partition", ["fixed", "watershed"])
def test_hpa_query(partition):
    p = make_hpa(partition)
    n_nodes = len(p.graph.nodes)
    for s, t in query_pairs(p.grid, 15) + [((1, 1), (2, 3))]:       # last one: same area
        res = p.query(s, t)
        check_result_path(p.grid, res, s, t, p.name)
        opt = true_cost(p.grid, s, t)
        assert res.cost >= opt - 1e-9
        assert res.cost <= 1.5 * opt + 2, f"{p.name}: path is far longer than the optimum"
        assert not p.graph.temporary and len(p.graph.nodes) == n_nodes, (
            "query must remove its temporary start/goal nodes")
    with pytest.raises(ValueError):
        p.query((0, 0), (1, 1))


def test_hpa_stats():
    p = make_hpa()
    st = p.stats()
    assert st["areas"] == int(p.labels.max()) + 1
    assert st["doorways"] == len(p.graph.nodes)
    assert st["preprocess_s"] == pytest.approx(p.preprocess_s)


def test_hpa_update_matches_fresh_graph():
    g = office().copy()
    p = make_hpa(grid=g)
    door = next(c for _, c in p.graph.nodes.values())            # block a doorway cell
    pairs = query_pairs(g, 10)
    for blocked in (True, False):
        assert p.update(door, blocked) >= 0
        assert g.blocked[door] == blocked
        assert {"areas_rebuilt", "paths_recomputed", "seconds"} <= set(p.last_update)
        fresh = hpa.DoorwayGraph(g, p.labels.copy(), p.max_gap)
        for s, t in pairs:
            if not g.is_free(s) or not g.is_free(t):
                continue
            res = p.query(s, t)
            ref = make_hpa(grid=g)
            ref.labels, ref.graph = p.labels.copy(), fresh
            expected = ref.query(s, t)
            assert res.success == expected.success, (
                "after update, answers must match a graph built from scratch on the changed map")
            if res.success:
                check_result_path(g, res, s, t, "HPA after update")
                assert res.cost == pytest.approx(expected.cost), (
                    "after update, answers must match a graph built from scratch on the changed map")


# ============================================================================ KHIZAR: analysis.py (task 28)


def results_frame():
    import pandas as pd
    rows = []
    for size, (base_q, hpa_q, prep) in {64: (0.010, 0.002, 0.08), 128: (0.002, 0.004, 0.1)}.items():
        for qid in range(4):
            rows.append({"method": "A*", "map_kind": "rooms", "map_size": size, "runtime_s": base_q,
                         "preprocess_s": 0.0, "query_id": qid, "success": True})
            rows.append({"method": "HPA-watershed", "map_kind": "rooms", "map_size": size,
                         "runtime_s": hpa_q, "preprocess_s": prep, "query_id": qid, "success": True})
    return pd.DataFrame(rows)


def test_breakeven_queries():
    import analysis
    t = analysis.breakeven_queries(results_frame(), "HPA-watershed").set_index(["map_kind", "map_size"])
    for col in ("preprocess_s", "baseline_query_s", "method_query_s", "breakeven"):
        assert col in t.columns
    assert t.loc[("rooms", 64), "breakeven"] == pytest.approx(0.08 / (0.010 - 0.002))
    assert math.isinf(t.loc[("rooms", 128), "breakeven"]), "slower queries never pay back: inf"


def test_plot_breakeven_and_repair(tmp_path):
    import analysis
    out = analysis.plot_breakeven(results_frame(), tmp_path / "be.png", "HPA-watershed")
    assert Path(out).exists() and Path(out).stat().st_size > 0
    trials = [{"cell": (1, i), "repair_s": 0.001 * i, "rebuild_s": 0.05, "areas_rebuilt": 2,
               "paths_recomputed": 6, "answers_match": True} for i in range(1, 6)]
    out = analysis.plot_repair(trials, tmp_path / "repair.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0


# ============================================================================ ZOREN: generators.py

import generators  # noqa: E402


def test_keep_largest_region():
    b = np.ones((7, 9), dtype=bool)
    b[1:3, 1:3] = False                      # small pocket (4 cells)
    b[1:6, 5:8] = False                      # big pocket (15 cells)
    before = b.copy()
    out = generators.keep_largest_region(b)
    assert (b == before).all(), "the input array must not be changed"
    assert out[1:3, 1:3].all() and not out[1:6, 5:8].any()


@pytest.mark.parametrize("make", [
    lambda s: generators.random_map(64, 0.2, s),
    lambda s: generators.room_map(64, s),
    lambda s: generators.maze_map(64, s),
], ids=["random", "rooms", "maze"])
def test_generators_shape_border_connected_repeatable(make):
    b = make(3)
    assert b.dtype == bool and b.shape == (64, 64)
    assert border_is_wall(b) and connected(b) and (~b).sum() > 100
    assert (make(3) == b).all(), "same seed must give the same map"
    assert not (make(4) == b).all(), "different seeds should give different maps"


def test_random_map_density_and_errors():
    b = generators.random_map(128, 0.2, 0)
    inside = b[1:-1, 1:-1]
    assert 0.12 < inside.mean() < 0.35
    with pytest.raises(ValueError):
        generators.random_map(4, 0.2, 0)
    with pytest.raises(ValueError):
        generators.random_map(64, 0.5, 0)


def test_room_map_errors():
    with pytest.raises(ValueError):
        generators.room_map(16, 0)
    with pytest.raises(ValueError):
        generators.room_map(64, 0, min_room=10, max_room=5)


def test_maze_has_exactly_one_route():
    b = generators.maze_map(41, 1)
    cells = int((~b).sum())
    edges = int((~b[:, :-1] & ~b[:, 1:]).sum() + (~b[:-1, :] & ~b[1:, :]).sum())
    assert edges == cells - 1, "a perfect maze is a tree: open cells - 1 = number of links"
    with pytest.raises(ValueError):
        generators.maze_map(4, 0)


# ============================================================================ ZOREN: app.py

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")      # no real window needed
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture
def app():
    pytest.importorskip("pygame")
    import app as app_module
    try:
        a = app_module.App(method="HPA-fixed", window_size=(1000, 700))
        a.generate_map("rooms", 64, 0)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}" if str(e) else "not implemented yet")
    return a


def first_open(grid, near=(10, 10)):
    return min(open_cells(grid), key=lambda c: abs(c[0] - near[0]) + abs(c[1] - near[1]))


def test_area_color():
    pytest.importorskip("pygame")
    import app as app_module
    colours = [app_module.area_color(i) for i in range(10)]
    for c in colours:
        assert len(c) == 3 and all(isinstance(v, (int, np.integer)) and 0 <= v <= 255 for v in c)
    assert len(set(colours)) == 10 and app_module.area_color(3) == colours[3]


def test_app_generate_and_query(app):
    assert app.grid.height == 64 and app.stage == 0 and app.start is None
    wall = tuple(int(v) for v in np.argwhere(app.grid.blocked)[0])
    assert app.set_start(wall) is False and app.start is None
    s, t = first_open(app.grid), first_open(app.grid, (50, 50))
    assert app.set_start(s) and app.set_goal(t)
    assert app.result.success and app.baseline_result.success
    assert app.explored and app.baseline_explored
    assert app.baseline_result.cost <= app.result.cost + 1e-9


def test_app_stages_wrap(app):
    import app as app_module
    n = len(app_module.STAGES)
    app.prev_stage()
    assert app.stage == n - 1
    app.next_stage()
    assert app.stage == 0
    app.set_start(first_open(app.grid))
    app.set_goal(first_open(app.grid, (50, 50)))
    for _ in range(n):
        app.draw()                                      # every stage must draw without crashing
        app.next_stage()


def test_app_cell_at(app):
    assert app.cell_at(-1000, -1000) is None
    hits = [app.cell_at(x, y) for x in range(0, 1000, 37) for y in range(0, 700, 37)]
    cells = [c for c in hits if c is not None]
    assert cells, "no pixel maps to a cell"
    assert all(app.grid.in_bounds(c) for c in cells)


def test_app_editor_paint_and_save(app, tmp_path):
    # Edits may be kept in a working copy while the editor is open; the spec only
    # requires them to be on the map once the editor is switched OFF.
    import map_io
    assert app.editing is False
    app.toggle_editor()
    assert app.editing is True
    cell = first_open(app.grid, (30, 30))
    app.paint(cell, True)
    app.paint((0, 5), False)                            # trying to open the border
    app.toggle_editor()
    assert app.editing is False and app.planner is not None
    assert app.grid.blocked[cell], "paint(cell, True) must make the cell a wall once the editor closes"
    assert border_is_wall(app.grid.blocked), "the outer border must stay wall"
    app.toggle_editor()
    app.paint(cell, False)
    app.toggle_editor()
    assert not app.grid.blocked[cell], "paint(cell, False) must open the cell again"
    app.toggle_editor()
    app.paint(cell, True)
    app.toggle_editor()
    out = tmp_path / "edited.map"
    app.save_map(out)
    assert (map_io.read_map(out).blocked == app.grid.blocked).all()
    app.load_map(out)
    assert app.grid.height == 64 and app.start is None


def test_app_toggle_block(app):
    s, t = first_open(app.grid), first_open(app.grid, (50, 50))
    app.set_start(s)
    app.set_goal(t)
    # a path cell with open space all round, so blocking it cannot cut the map in two
    interior = [c for c in app.result.path[1:-1]
                if not app.grid.blocked[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2].any()]
    if not interior:
        pytest.skip("no suitable cell on this path")
    cell = interior[len(interior) // 2]
    app.toggle_block(cell)
    assert app.grid.blocked[cell], "toggle_block must block an open cell"
    assert app.result.success and cell not in app.result.path, "the query must be re-run around the new wall"
    app.toggle_block(cell)
    assert not app.grid.blocked[cell], "toggle_block on a wall must open it"
    app.toggle_block((0, 5))
    assert app.grid.blocked[0, 5], "border cells are ignored"


def test_save_figure(tmp_path):
    pytest.importorskip("pygame")
    import app as app_module
    g = office()
    out = app_module.save_figure(g, None, [(1, 1), (1, 2), (2, 3)], tmp_path / "fig.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0


# ============================================================================ skip unfinished work


def _skip_if_not_implemented(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as e:
            pytest.skip(f"not implemented yet: {e}" if str(e) else "not implemented yet")
    return wrapper


for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj):
        globals()[_name] = _skip_if_not_implemented(_obj)