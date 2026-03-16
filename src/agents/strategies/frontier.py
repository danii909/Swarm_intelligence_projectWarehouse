"""
Strategia 2 — Frontier-Based Exploration.

L'agente mantiene un insieme di "frontiere": celle esplorate adiacenti
a celle ancora non esplorate. Si muove verso la frontiera più vicina (BFS).
Quando trova un oggetto, lo raccoglie e consegna.
Quando trasporta, si dirige al magazzino più vicino.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy

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

        move = self._priority_move(agent, env, pathfinder, occupied)
        if move:
            return move

        frontiers = self._find_frontiers(agent, env)
        if not frontiers:
            # Mappa completamente esplorata: random walk locale
            neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
            free = [n for n in neighbors if n not in occupied]
            return free[0] if free else (neighbors[0] if neighbors else None)

        best = min(
            frontiers,
            key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
        )
        return pathfinder.next_step(agent.pos, best, occupied - {agent.pos})
