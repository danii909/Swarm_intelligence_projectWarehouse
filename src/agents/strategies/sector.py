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

# Nota: la suddivisione è implementata come bande orizzontali
# (una per agente) nella funzione _compute_sector.


class SectorStrategy(ExplorationStrategy):
    """
    Divide la griglia in bande orizzontali; l'agente si specializza
    nella banda corrispondente al proprio ID (mod numero agenti).
    """

    def __init__(self, num_agents: int = 5) -> None:
        super().__init__()
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

        # --- Priorità universale: trasporto + oggetti noti ---
        move = self._priority_move(agent, env, pathfinder, occupied)
        if move:
            return move

        # --- Inizializza settore se necessario ---
        if not self._sector_cells:
            self._sector_cells = self._compute_sector(agent.id, env)

        # Usa map globale nota, ma filtra per il proprio settore
        targets = self._coverage_targets(agent, env)
        sector_targets = [c for c in targets if c in self._sector_cells]
        
        if sector_targets:
            # Prioritizza il proprio settore
            best = min(
                sector_targets,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
        elif targets:
            # Settore esaurito: esplora il resto della griglia
            best = min(
                targets,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
        else:
            # Tutto visitato
            return None

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
