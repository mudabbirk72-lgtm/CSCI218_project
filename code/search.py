from __future__ import annotations
from collections.abc import Callable,Collection,Hashable,Iterable
from itertools import count
from grid import Cell,Grid,SearchResult
import heapq
import math
import time


def manhattan(a:Cell,b:Cell)->float:
    # calculate Manhattan distance
    return float(abs(a[0]-b[0])+abs(a[1]-b[1]))


def octile(a:Cell,b:Cell)->float:
    # calculate row and column differences
    dr=abs(a[0]-b[0])
    dc=abs(a[1]-b[1])
    return float(max(dr,dc)+(math.sqrt(2)-1)*min(dr,dc))


def astar(start:Hashable,
          goals:Collection[Hashable],
          neighbours:Callable[[Hashable],Iterable[tuple[Hashable,float]]],
          heuristic:Callable[[Hashable],float],
          on_expand:Callable[[Hashable],None]|None=None)->SearchResult:

    start_time=time.perf_counter()

    # check if start is already a goal
    if start in goals:
        return SearchResult(
            path=[start],
            cost=0.0,
            expanded=0,
            generated=1,
            runtime_s=time.perf_counter()-start_time,
            success=True
        )

    # store the best cost to each node
    g_score={start:0.0}

    # store the parent of each node
    parent={start:None}

    # nodes that have already been expanded
    closed=set()

    # priority queue for A*
    tie_breaker=count()
    open_list=[(heuristic(start),-0.0,next(tie_breaker),start,0.0)]

    expanded=0
    generated=1

    while open_list:
        _,_,_,current,current_g=heapq.heappop(open_list)

        if current in closed:
            continue

        if current_g!=g_score.get(current):
            continue

        closed.add(current)
        expanded+=1

        if on_expand is not None:
            on_expand(current)

        # check if the current node is a goal
        if current in goals:
            path=reconstruct_path(parent,current)

            return SearchResult(
                path=path,
                cost=current_g,
                expanded=expanded,
                generated=generated,
                runtime_s=time.perf_counter()-start_time,
                success=True
            )

        # check all neighbouring nodes
        for neighbour,step_cost in neighbours(current):
            new_g=current_g+step_cost

            # check for a better path
            if new_g<g_score.get(neighbour,float("inf")):
                g_score[neighbour]=new_g
                parent[neighbour]=current

                f_score=new_g+heuristic(neighbour)

                heapq.heappush(
                    open_list,
                    (f_score,-new_g,next(tie_breaker),neighbour,new_g)
                )

                generated+=1

    # no goal could be reached
    return SearchResult(
        path=None,
        cost=float("inf"),
        expanded=expanded,
        generated=generated,
        runtime_s=time.perf_counter()-start_time,
        success=False
    )


def dijkstra(start:Hashable,
             goals:Collection[Hashable],
             neighbours:Callable[[Hashable],Iterable[tuple[Hashable,float]]],
             on_expand:Callable[[Hashable],None]|None=None)->SearchResult:

    # Dijkstra uses zero heuristic
    return astar(
        start,
        goals,
        neighbours,
        lambda _:0.0,
        on_expand
    )


def dijkstra_all(start:Hashable,
                 neighbours:Callable[[Hashable],Iterable[tuple[Hashable,float]]],
                 targets:Collection[Hashable]|None=None
                 )->tuple[dict[Hashable,float],dict[Hashable,Hashable|None]]:

    # store distances from the start
    distances={start:0.0}

    # store the parent of each node
    parent={start:None}

    target_set=set(targets) if targets is not None else None

    if target_set is not None and not target_set:
        return distances,parent

    finished_targets=set()

    tie_breaker=count()
    open_list=[(0.0,next(tie_breaker),start)]

    while open_list:
        current_distance,_,current=heapq.heappop(open_list)

        if current_distance!=distances.get(current):
            continue

        # stop when all targets are reached
        if target_set is not None and current in target_set:
            finished_targets.add(current)

            if finished_targets==target_set:
                break

        # check neighbouring nodes
        for neighbour,step_cost in neighbours(current):
            new_distance=current_distance+step_cost

            if new_distance<distances.get(neighbour,float("inf")):
                distances[neighbour]=new_distance
                parent[neighbour]=current

                heapq.heappush(
                    open_list,
                    (new_distance,next(tie_breaker),neighbour)
                )

    return distances,parent


def reconstruct_path(parent:dict[Hashable,Hashable|None],
                     target:Hashable)->list[Hashable]:

    # rebuild path from target to start
    path=[]
    current=target

    while current is not None:
        path.append(current)
        current=parent[current]

    path.reverse()

    return path


class FlatPlanner:

    def __init__(self,algorithm:str="astar",heuristic:str="octile")->None:

        # check the selected algorithm and heuristic
        if algorithm not in ("astar","dijkstra"):
            raise ValueError("Unknown algorithm")

        if heuristic not in ("octile","manhattan"):
            raise ValueError("Unknown heuristic")

        self.algorithm=algorithm
        self.heuristic=heuristic
        self.name="A*" if algorithm=="astar" else "Dijkstra"
        self.grid:Grid|None=None

    def preprocess(self,grid:Grid)->float:

        # store the grid
        self.grid=grid
        return 0.0

    def query(self,start:Cell,goal:Cell,
              on_expand:Callable[[Cell],None]|None=None)->SearchResult:

        # check if preprocessing has been done
        if self.grid is None:
            raise RuntimeError("Call preprocess() before query()")

        # check if start and goal are valid
        if not self.grid.is_free(start):
            raise ValueError("Invalid start cell")

        if not self.grid.is_free(goal):
            raise ValueError("Invalid goal cell")

        # run the selected algorithm
        if self.algorithm=="dijkstra":
            return dijkstra(start,{goal},self.grid.neighbours,on_expand)

        if self.heuristic=="manhattan":
            heuristic_function=lambda cell: manhattan(cell,goal)
        else:
            heuristic_function=lambda cell: octile(cell,goal)

        return astar(start,{goal},self.grid.neighbours,heuristic_function,on_expand)

    def update(self,cell:Cell,blocked:bool)->float:
        raise NotImplementedError

    def stats(self)->dict:

        # return planner statistics
        return {"areas":0,"doorways":0,"preprocess_s":0.0}