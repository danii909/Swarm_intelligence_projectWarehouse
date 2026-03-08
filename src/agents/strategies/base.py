"""
Interfaccia astratta per le strategie di esplorazione.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set, Tuple, TYPE_CHECKING

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
      - agent.local_map    — celle esplorate finora dall'agente
      - agent.known_objects — oggetti noti non ancora raccolti
      - agent.pos          — posizione corrente
      - agent.carrying_object
      - env.grid           — mappa globale (solo per walkability)
      - pathfinder         — per calcolare percorsi
      - occupied           — posizioni occupate da altri agenti (anti-collisione)
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

    @property
    def name(self) -> str:
        return self.__class__.__name__
