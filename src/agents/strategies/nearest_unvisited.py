"""
Strategia — Nearest Unvisited (Greedy Puro).

L'agente si dirige sempre alla cella EMPTY non visitata più vicina
in termini di distanza Manhattan. Nessuna euristica complessa,
solo ottimizzazione greedy della distanza.

Vantaggi:
- Zero overhead computazionale
- Comportamento deterministico e prevedibile
- Baseline semplice per confronti

Svantaggi:
- Può fare backtracking
- Non garantisce copertura ottimale
- Ignora la cooperazione con altri agenti
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.environment.grid import CellType

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class NearestUnvisitedStrategy(ExplorationStrategy):
    """
    Greedy puro: si muove sempre verso la cella EMPTY non visitata
    più vicina (distanza Manhattan).
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

        # --- Trova tutte le celle EMPTY non visitate ---
        unvisited_empty = [
            (r, c) for (r, c), cell_type in agent.local_map.items()
            if cell_type == CellType.EMPTY and (r, c) not in agent.visited_cells
        ]

        if not unvisited_empty:
            # Tutta la mappa visitata: random walk
            neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
            free = [n for n in neighbors if n not in occupied]
            return free[0] if free else (neighbors[0] if neighbors else None)

        # Greedy: vai alla cella non visitata più vicina
        nearest_unvisited = min(
            unvisited_empty,
            key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
        )

        step = pathfinder.next_step(agent.pos, nearest_unvisited, occupied - {agent.pos})
        return step
