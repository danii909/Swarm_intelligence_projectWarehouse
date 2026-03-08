"""
Algoritmi di pathfinding sulla griglia.

BFS è sufficiente per griglia 25x25 con ostacoli statici.
A* è disponibile per chi vuole ridurre il numero di nodi espansi.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from src.environment.grid import Grid, CellType

# Celle percorribili solo come destinazione finale (non come transito)
_DOOR_CELLS = frozenset({CellType.ENTRANCE, CellType.EXIT})


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: Optional[set] = None,
    allow_warehouse: bool = False,
) -> Optional[List[Tuple[int, int]]]:
    """
    BFS sulla griglia. Restituisce il percorso [start, ..., goal] oppure None.

    Parameters
    ----------
    grid            : Grid
    start           : (row, col) di partenza
    goal            : (row, col) di destinazione
    blocked         : celle aggiuntive da considerare non percorribili
    allow_warehouse : se True usa delivery_neighbors (include celle WAREHOUSE)
    """
    if start == goal:
        return [start]

    blocked = blocked or set()
    neighbor_fn = grid.delivery_neighbors if allow_warehouse else grid.walkable_neighbors

    queue: deque[Tuple[int, int]] = deque([start])
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    while queue:
        current = queue.popleft()
        for nb in neighbor_fn(*current):
            if nb in came_from:
                continue
            if nb in blocked and nb != goal:
                continue
            # Direzionalità: ingresso/uscita sono percorribili solo come
            # destinazione finale, mai come celle di transito intermedio
            # (solo in modalità normale, non dentro il magazzino).
            if not allow_warehouse and grid.cell(*nb) in _DOOR_CELLS and nb != goal:
                continue
            came_from[nb] = current
            if nb == goal:
                return _reconstruct(came_from, start, goal)
            queue.append(nb)

    return None  # percorso non trovato


def astar(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: Optional[set] = None,
    allow_warehouse: bool = False,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* sulla griglia con euristica Manhattan.
    Restituisce il percorso [start, ..., goal] oppure None.
    """
    import heapq

    if start == goal:
        return [start]

    blocked = blocked or set()
    neighbor_fn = grid.delivery_neighbors if allow_warehouse else grid.walkable_neighbors

    g_score: Dict[Tuple[int, int], int] = {start: 0}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    open_heap: list = [(manhattan(start, goal), 0, start)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, start, goal)
        if g > g_score.get(current, float("inf")):
            continue
        for nb in neighbor_fn(*current):
            if nb in blocked and nb != goal:
                continue
            # Direzionalità: ingresso/uscita sono percorribili solo come
            # destinazione finale, mai come celle di transito intermedio
            # (solo in modalità normale, non dentro il magazzino).
            if not allow_warehouse and grid.cell(*nb) in _DOOR_CELLS and nb != goal:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = current
                f = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_heap, (f, tentative_g, nb))

    return None


def _reconstruct(
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    path = []
    node: Optional[Tuple[int, int]] = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


class Pathfinder:
    """Piccola facade che mantiene in cache i percorsi recenti."""

    def __init__(self, grid: Grid, use_astar: bool = True) -> None:
        self.grid = grid
        self._fn = astar if use_astar else bfs

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: Optional[set] = None,
        allow_warehouse: bool = False,
    ) -> Optional[List[Tuple[int, int]]]:
        return self._fn(self.grid, start, goal, blocked, allow_warehouse)

    def next_step(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: Optional[set] = None,
        allow_warehouse: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """Restituisce solo il prossimo passo verso goal, oppure None.

        Se il percorso è bloccato da altri agenti, ritenta ignorandoli come
        ostacoli: il conflitto viene poi risolto da _apply_moves (deadlock
        testa-a-testa → swap; coda → scorrimento).
        """
        path = self.find_path(start, goal, blocked, allow_warehouse)
        if path and len(path) > 1:
            return path[1]
        # Fallback: ricalcola ignorando gli agenti (solo muri come ostacoli)
        if blocked:
            path = self.find_path(start, goal, blocked=None, allow_warehouse=allow_warehouse)
            if path and len(path) > 1:
                return path[1]
        return None
