"""
Strategia 1 — Random Walk con priorità oggetti noti.

L'agente si sposta in una direzione casuale tra quelle percorribili.
Se ha un oggetto noto si dirige subito verso di esso.
Se sta trasportando un oggetto, si dirige all'ingresso del magazzino più vicino.
"""

from __future__ import annotations

import random
from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy

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

        move = self._priority_move(agent, env, pathfinder, occupied)
        if move:
            return move

        neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
        free = [n for n in neighbors if n not in occupied]
        if free:
            return random.choice(free)
        return random.choice(neighbors) if neighbors else None
