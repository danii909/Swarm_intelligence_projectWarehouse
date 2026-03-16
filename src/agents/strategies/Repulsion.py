"""
Strategia 4 — Repulsion-Based Spreader.

L'agente esplora le frontiere più lontane dagli altri agenti noti
(via visibilità diretta o comunicazione). Non ha settori fissi: la
dispersione è emergente e si adatta dinamicamente alle posizioni
degli altri agenti, inclusi i casi in cui un agente si ferma o va ko.

Funzione di score per ogni frontiera:
  score = isolation − travel_cost
  isolation  = distanza media dagli agenti noti (più è alta, meglio)
  travel_cost = distanza da me (cost per raggiungerla)

Con comm_radius=2 le posizioni degli altri agenti si aggiornano solo
quando ci si avvicina; questo crea una dispersione "pigra" che tende
a coprire aree non ancora visitate da nessun agente recente.
"""

from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class RepulsionStrategy(ExplorationStrategy):
    """
    Dispersione emergente: si allontana dagli agenti noti scegliendo
    frontiere isolate. Si adatta automaticamente se un agente cade.
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

        known_positions: List[Tuple[int, int]] = [
            pos for pos, _tick in agent.known_agents.values()
        ]

        # --- Frontiere nella mappa locale ---
        frontiers = self._find_frontiers(agent, env)

        if not frontiers:
            # Nessuna frontiera locale: considera tutte le celle EMPTY inesplorate
            frontiers = self._unexplored_empty(agent, env)

        if not frontiers:
            # Tutto esplorato: random walk
            neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
            free = [n for n in neighbors if n not in occupied]
            return random.choice(free) if free else (random.choice(neighbors) if neighbors else None)

        row, col = agent.row, agent.col
        if known_positions:
            known_count = len(known_positions)

            def score(frontier: Tuple[int, int]) -> float:
                fr, fc = frontier
                isolation = sum(
                    abs(fr - pr) + abs(fc - pc)
                    for pr, pc in known_positions
                ) / known_count
                travel = abs(fr - row) + abs(fc - col)
                return isolation - travel
        else:
            # Con zero agenti noti: isolation=0, massimizzare score equivale
            # minimizzare la distanza da me (stesso comportamento della formula).
            def score(frontier: Tuple[int, int]) -> float:
                fr, fc = frontier
                return -(abs(fr - row) + abs(fc - col))

        best = max(frontiers, key=score)
        return pathfinder.next_step(agent.pos, best, occupied - {agent.pos})

    # ------------------------------------------------------------------

    def _score(
        self,
        frontier: Tuple[int, int],
        agent: "Agent",
        known_positions: List[Tuple[int, int]],
    ) -> float:
        """
        Punteggio higher = frontiera più isolata dai noti e più vicina a me.
        isolation: dist media dagli altri (vogliamo alta)
        travel:    dist da me  (vogliamo bassa → sottratta)
        """
        if known_positions:
            isolation = sum(
                abs(frontier[0] - p[0]) + abs(frontier[1] - p[1])
                for p in known_positions
            ) / len(known_positions)
        else:
            isolation = 0.0  # nessun agente noto: scegli solo per distanza

        travel = abs(frontier[0] - agent.row) + abs(frontier[1] - agent.col)
        return isolation - travel

# Alias per compatibilità con il codice esistente
SectorStrategy = RepulsionStrategy
