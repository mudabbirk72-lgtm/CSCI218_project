# CSCI218 – Multi-Layered (Hierarchical) Path Planning

Two-tier path planning on grid maps: plan a route through *areas* first, then fill in the
cells. We compare three hierarchical methods (the brief's Basic two-tier search, a Corridor
variant and an HPA*-style doorway graph) and three ways of splitting a map into areas
(fixed blocks, watershed, Louvain) against flat A* and Dijkstra, on generated maps and on
real city maps from the Moving AI street benchmark.

## Setup

Python 3.9 or newer.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project layout

```
code/        the planner and experiment code (plain modules, imported as `import grid`)
  grid.py         the Grid map, movement rules, SearchResult, Scenario
  map_io.py       .map / .scen files (Moving AI format)
  generators.py   random, room and maze map generators
  search.py       A*, Dijkstra, heuristics, FlatPlanner
  partition.py    fixed blocks, watershed (with area-size cap), Louvain
  area_graph.py   areas and the crossings between them
  two_tier.py     TwoTierPlanner (Basic and Corridor refinement)
  hpa.py          HPAPlanner: doorway graph, queries, local repair when the map changes
  experiments.py  runs an experiment from a YAML config and writes a CSV
  analysis.py     statistics and figures from the result CSVs
  app.py          interactive demo (pygame)
configs/     experiment configs (pilot, scaling, partitions, street, dynamic)
maps/        benchmark street maps (Berlin, Boston, Paris at 256/512/1024) and saved maps
results/     result CSVs, run logs and figures
scripts/     check_street_maps.py: checks our A* against the benchmark's optimal lengths
tests/       pytest tests
```

## Running the tests

```bash
python -m pytest tests -q
```

## Checking correctness against the street benchmark

```bash
python scripts/check_street_maps.py
```

Runs A* on queries from the Moving AI `.scen` files and compares each path length with the
published optimum (all 180 matched). Output: `results/street_check.csv`.

## Running the experiments

Every experiment is described by a YAML file in `configs/`:

```bash
python code/experiments.py configs/scaling.yaml --dry-run   # quick check: first map, few queries
python code/experiments.py configs/scaling.yaml             # full run
```

| Config | What it measures | Output | Time (approx.) |
|---|---|---|---|
| `pilot.yaml` | Small test run of every method | `results/pilot.csv` | < 1 min |
| `street.yaml` | Real city maps, 256–1024 | `results/street.csv` | ~6 min |
| `partitions.yaml` | Fixed vs watershed vs Louvain, and the watershed area-cap sweep | `results/partitions.csv` | ~35 min |
| `scaling.yaml` | Every method on room, random and maze maps from 64 to 1024 | `results/scaling.csv` | ~30 min |
| `dynamic.yaml` | Blocking doorways: HPA local repair vs full rebuild | `results/dynamic.csv` | ~7 min |

Maps and queries come from fixed seeds, so every run is repeatable. Only the timing columns
change between machines.

## Making the figures and summary tables

```bash
python -c "import sys; sys.path.insert(0, 'code'); import analysis; [print(p) for p in analysis.make_all()]"
```

Writes every figure and summary table to `results/figures/`.

## The demo app

```bash
python code/app.py --generate rooms --size 128 --seed 2 --method HPA-watershed
python code/app.py --map maps/benchmark/street/Berlin_0_512.map --method HPA-fixed
```

Options: `--generate random|rooms|maze`, `--size N`, `--seed S`, `--method NAME` (any key of
`METHODS` in `experiments.py`), `--no-diagonal`, `--window 1280x800`.

| Input | Action |
|---|---|
| Left / right click | Set start / goal (the query runs when both are set) |
| `B` or middle click | Block or open the cell under the mouse; the path is repaired live |
| Arrow keys / mouse wheel | Step through the stages: map, distance, areas, graph, route, path, compare with A* |
| `M` | Switch to the next method on the same map |
| `G` | Generate a new map of the same kind (next seed) |
| `E` | Map editor on/off (left-drag walls, right-drag erase, `[` `]` brush size) |
| `S` / `F` | Save the map / save a figure of the current view |
| `Esc` | Quit |
