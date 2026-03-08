"""
Strategia 2 — Frontier-Based Exploration.

L'agente mantiene un insieme di "frontiere": celle esplorate adiacenti
a celle ancora non esplorate. Si muove verso la frontiera più vicina (BFS).
Quando trova un oggetto, lo raccoglie e consegna.
Quando trasporta, si dirige al magazzino più vicino.
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
    più vicina dell'area inesplorata nella propria mappa locale.
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
            # Mappa completamente esplorata: random walk
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
        - è nella mappa locale (esplorata)
        - è percorribile (solo EMPTY: le porte dei magazzini non sono esplorabili)
        - ha almeno un vicino NON presente nella mappa locale
        """
        frontiers: Set[Tuple[int, int]] = set()
        for (r, c), cell_type in agent.local_map.items():
            if cell_type != CellType.EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in agent.local_map and env.grid.in_bounds(nr, nc):
                    frontiers.add((r, c))
                    break
        return frontiers
