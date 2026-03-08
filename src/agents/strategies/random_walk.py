"""
Strategia 1 — Random Walk con priorità oggetti noti.

L'agente si sposta in una direzione casuale tra quelle percorribili.
Se ha un oggetto noto nel campo visivo si dirige subito verso di esso.
Se sta trasportando un oggetto, si dirige all'ingresso del magazzino più vicino.
"""

from __future__ import annotations

import random
from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.environment.grid import CellType

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class RandomWalkStrategy(ExplorationStrategy):
    """
    Strategia di base: random walk con bias verso oggetti noti
    e consegna al magazzino più vicino.
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

        # --- Altrimenti random walk ---
        neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
        free = [n for n in neighbors if n not in occupied]
        if free:
            return random.choice(free)
        if neighbors:
            return random.choice(neighbors)
        return None
