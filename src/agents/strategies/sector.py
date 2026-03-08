"""
Strategia 4 — Copertura per Settori.

La griglia viene suddivisa in settori assegnati agli agenti (in base al loro ID).
Ogni agente esplora il proprio settore prima di muoversi
verso settori adiacenti non esplorati.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.environment.grid import CellType, DIRECTIONS

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder

NUM_SECTORS_SIDE = 2   # divide la griglia in 2x2 = 4 settori + centro


class SectorStrategy(ExplorationStrategy):
    """
    Divide la griglia in settori; l'agente si specializza sul settore
    corrispondente al proprio ID (mod numero settori).
    """

    def __init__(self, num_agents: int = 5) -> None:
        self._num_agents = num_agents
        self._sector_cells: List[Tuple[int, int]] = []
        self._target_index: int = 0

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

        # --- Inizializza o aggiorna settore se necessario ---
        if not self._sector_cells:
            self._sector_cells = self._compute_sector(agent.id, env)

        # Scegli la cella del settore non ancora esplorata più vicina
        unexplored = [
            c for c in self._sector_cells
            if c not in agent.local_map
        ]
        if not unexplored:
            # Settore esplorato: esplora l'intera griglia come frontier
            unexplored = [
                (r, c)
                for r in range(env.grid.size)
                for c in range(env.grid.size)
                if (r, c) not in agent.local_map
                and env.grid.cell(r, c) == CellType.EMPTY
            ]

        if not unexplored:
            # Tutto esplorato
            return None

        best = min(
            unexplored,
            key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
        )
        return pathfinder.next_step(agent.pos, best, occupied - {agent.pos})

    # ------------------------------------------------------------------

    def _compute_sector(
        self, agent_id: int, env: "Environment"
    ) -> List[Tuple[int, int]]:
        """
        Suddivide la griglia in settori orizzontali uguali (uno per agente).
        """
        size = env.grid.size
        rows_per_sector = max(1, size // self._num_agents)
        sector_index = agent_id % self._num_agents
        r_start = sector_index * rows_per_sector
        r_end = r_start + rows_per_sector if sector_index < self._num_agents - 1 else size

        cells = []
        for r in range(r_start, r_end):
            for c in range(size):
                if env.grid.cell(r, c) == CellType.EMPTY:
                    cells.append((r, c))
        return cells
