"""
Strategia 5 — Greedy verso l'oggetto noto più vicino.

Se l'agente non conosce ancora nessun oggetto, esplora con frontier-based.
Appena rileva uno o più oggetti, si muove in modo greedy verso il più vicino.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.agents.strategies.frontier import FrontierStrategy

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class GreedyStrategy(ExplorationStrategy):
    """
    Greedy puro: raccoglie l'oggetto noto più vicino, poi consegna,
    poi torna a cercare. Se non conosce oggetti, delega al frontier.
    """

    def __init__(self) -> None:
        self._fallback = FrontierStrategy()

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

        # --- Fallback: frontier-based ---
        return self._fallback.next_move(agent, env, pathfinder, occupied)
