"""
Strategia 2 — Frontier-Based Exploration.

L'agente mantiene un insieme di "frontiere": celle visitate adiacenti
a celle ancora non visitate. Si muove verso la frontiera più vicina (BFS).
Quando trova un oggetto, lo raccoglie e consegna.
Quando trasporta, si dirige al magazzino più vicino.

Nota: la mappa del terreno è completa fin dall'inizio. Le frontiere
rappresentano il confine tra aree cercate e non cercate per oggetti.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.environment.grid import CellType, DIRECTIONS

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class FrontierStrategy(ExplorationStrategy):
    """
    Esplorazione frontier-based: si muove sempre verso la frontiera
    più vicina dell'area non ancora visitata (non cercata per oggetti).
    """

    def next_move(
        self,
        agent: "Agent",
        env: "Environment",
        pathfinder: "Pathfinder",
        occupied: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:

        # --- Se sta trasportando, vai all'ingresso più vicino ---
        if agent.carrying_object:
            target = env.nearest_warehouse_entrance(*agent.pos)
            if target:
                step = pathfinder.next_step(agent.pos, target, occupied - {agent.pos})
                if step:
                    return step

        # --- Se conosce oggetti, vai al più vicino ---
        if agent.known_objects:
            nearest = min(
                agent.known_objects,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
            step = pathfinder.next_step(agent.pos, nearest, occupied - {agent.pos})
            if step:
                return step

        # --- Calcola frontiere nella mappa locale ---
        frontiers = self._find_frontiers(agent, env)
        if not frontiers:
            # Tutte le aree sono state visitate: cerca celle EMPTY non ancora visitate
            # (questo può accadere se le celle erano oscurate ma ora la mappa è nota)
            unvisited_empty = [
                (r, c) for (r, c), cell_type in agent.local_map.items()
                if cell_type == CellType.EMPTY and (r, c) not in agent.visited_cells
            ]
            if unvisited_empty:
                # Vai alla cella non visitata più vicina
                nearest = min(
                    unvisited_empty,
                    key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
                )
                step = pathfinder.next_step(agent.pos, nearest, occupied - {agent.pos})
                return step
            # Tutto visitato: random walk
            neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
            free = [n for n in neighbors if n not in occupied]
            return free[0] if free else (neighbors[0] if neighbors else None)

        # Vai alla frontiera più vicina
        best = min(
            frontiers,
            key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
        )
        step = pathfinder.next_step(agent.pos, best, occupied - {agent.pos})
        return step

    # ------------------------------------------------------------------

    def _find_frontiers(
        self, agent: "Agent", env: "Environment"
    ) -> Set[Tuple[int, int]]:
        """
        Una cella è una frontiera se:
        - è nella mappa locale (sempre true ora che la mappa è completa)
        - è visitata (in visited_cells)
        - è percorribile (solo EMPTY: le porte dei magazzini non sono esplorabili)
        - ha almeno un vicino EMPTY NON ancora visitato

        Questo identifica il confine tra aree cercate e non cercate per oggetti.
        """
        frontiers: Set[Tuple[int, int]] = set()
        for (r, c), cell_type in agent.local_map.items():
            # Skip non-walkable cells
            if cell_type != CellType.EMPTY:
                continue
            # Skip unvisited cells (can't be a frontier if we haven't been there)
            if (r, c) not in agent.visited_cells:
                continue
            # Check if any neighbor is unvisited EMPTY
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not env.grid.in_bounds(nr, nc):
                    continue
                # Is neighbor an unvisited empty cell?
                if (nr, nc) not in agent.visited_cells and agent.local_map.get((nr, nc)) == CellType.EMPTY:
                    frontiers.add((r, c))
                    break
        return frontiers
