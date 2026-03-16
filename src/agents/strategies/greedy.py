"""
Strategia 5 — Greedy warehouse-centric.

In esplorazione sceglie la frontiera che minimizza:
  dist(agente → frontiera) + dist(frontiera → magazzino)

Questo concentra l'agente attorno ai magazzini: quando trova un oggetto,
la consegna è già breve. Gli altri agenti esplorano lontano; questo presidia
le zone di consegna.

Secondo meccanismo: se conosce la posizione di altri agenti (via visione
diretta o comunicazione), si avvicina a quelli fuori raggio per forzare
il merge delle mappe e ricevere eventuali oggetti da loro scoperti.

Tradeoff: copre male le zone lontane dai magazzini, ma ha il percorso
pickup→consegna più corto tra tutte le strategie.
"""

from __future__ import annotations

import random
from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class GreedyStrategy(ExplorationStrategy):
    """
    Esplorazione warehouse-centric: esplora frontiere vicine ai magazzini.
    Usa la comunicazione per ottenere la mappa degli altri agenti.
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

        nearest_wh = env.nearest_warehouse_entrance(*agent.pos)

        # --- Esplorazione: avvicinati ad agenti noti fuori raggio ---
        # Forza la comunicazione per ricevere la loro mappa/oggetti scoperti.
        if agent.known_agents:
            nearest_agent_pos = min(
                (pos for pos, _tick in agent.known_agents.values()),
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
            if abs(nearest_agent_pos[0] - agent.row) + abs(nearest_agent_pos[1] - agent.col) > agent.comm_radius:
                step = pathfinder.next_step(agent.pos, nearest_agent_pos, occupied - {agent.pos})
                if step:
                    return step

        # --- Esplorazione: frontiera più vicina al magazzino ---
        # Minimizza dist(me→frontiera) + dist(frontiera→magazzino):
        # l'agente presidia le zone attorno ai magazzini.
        frontiers = self._find_frontiers(agent, env)
        if frontiers:
            best = min(
                frontiers,
                key=lambda p: (
                    abs(p[0] - agent.row) + abs(p[1] - agent.col)
                    + (abs(p[0] - nearest_wh[0]) + abs(p[1] - nearest_wh[1])
                       if nearest_wh else 0)
                ),
            )
            step = pathfinder.next_step(agent.pos, best, occupied - {agent.pos})
            if step:
                return step

        # --- Fallback: random walk ---
        neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
        free = [n for n in neighbors if n not in occupied]
        return random.choice(free) if free else (random.choice(neighbors) if neighbors else None)
