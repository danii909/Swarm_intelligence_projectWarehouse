"""
Interfaccia astratta per le strategie di esplorazione.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.environment.grid import CellType, DIRECTIONS

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class ExplorationStrategy(ABC):
    """
    Ogni strategia implementa `next_move`, che restituisce la prossima
    cella (row, col) verso cui l'agente deve muoversi, oppure None se
    l'agente deve restare fermo per questo tick.

    La strategia può usare:
      - agent.local_map     — celle esplorate finora dall'agente
      - agent.known_objects — oggetti noti non ancora raccolti
      - agent.known_agents  — posizioni note di altri agenti
      - agent.pos           — posizione corrente
      - agent.carrying_object
      - env.grid            — mappa globale (solo per walkability)
      - pathfinder          — per calcolare percorsi
      - occupied            — posizioni occupate da altri agenti (anti-collisione)
    """

    @abstractmethod
    def next_move(
        self,
        agent: "Agent",
        env: "Environment",
        pathfinder: "Pathfinder",
        occupied: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Restituisce la prossima cella verso cui spostarsi, oppure None."""
        ...

    # ------------------------------------------------------------------
    # Metodi condivisi tra le strategie
    # ------------------------------------------------------------------

    def _priority_move(
        self,
        agent: "Agent",
        env: "Environment",
        pathfinder: "Pathfinder",
        occupied: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """
        Priorità universale applicata da tutte le strategie:
          1. Se l'agente trasporta un oggetto → vai all'ingresso più vicino.
          2. Se conosce oggetti → vai al più vicino.
        Restituisce None se nessuna delle due condizioni è attiva,
        delegando alla logica di esplorazione specifica.
        """
        if agent.carrying_object:
            target = env.nearest_warehouse_entrance(*agent.pos)
            if target:
                step = pathfinder.next_step(agent.pos, target, occupied - {agent.pos})
                if step:
                    return step

        if agent.known_objects:
            nearest = min(
                agent.known_objects,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
            step = pathfinder.next_step(agent.pos, nearest, occupied - {agent.pos})
            if step:
                return step

        return None

    def _find_frontiers(
        self,
        agent: "Agent",
        env: "Environment",
    ) -> Set[Tuple[int, int]]:
        """
        Celle EMPTY già esplorate nella mappa locale che hanno almeno
        un vicino non ancora esplorato. Usata da più strategie.
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

    def _unexplored_empty(
        self,
        agent: "Agent",
        env: "Environment",
    ) -> Set[Tuple[int, int]]:
        """
        Celle EMPTY globali non ancora presenti nella mappa locale.
        Helper mantenuto anche per compatibilita' con strategie legacy.
        """
        return {
            (r, c)
            for r in range(env.grid.size)
            for c in range(env.grid.size)
            if env.grid.cell(r, c) == CellType.EMPTY and (r, c) not in agent.local_map
        }

    @property
    def name(self) -> str:
        return self.__class__.__name__
