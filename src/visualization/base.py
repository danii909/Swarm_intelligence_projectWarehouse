"""
Interfaccia comune per i visualizzatori della simulazione.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment


class BaseVisualizer(ABC):
    """
    Interfaccia che ogni visualizzatore deve implementare.

    Il Simulator chiama:
      1. setup()  — una volta sola all'inizio
      2. update() — dopo ogni tick
      3. close()  — al termine
    """

    @abstractmethod
    def setup(self, env: "Environment", agents: List["Agent"]) -> None:
        """Inizializza la finestra/canvas. Chiamato una sola volta."""
        ...

    @abstractmethod
    def update(self, tick: int, agents: List["Agent"], env: "Environment") -> bool:
        """
        Aggiorna il frame corrente.

        Returns
        -------
        bool
            True  → continua la simulazione
            False → l'utente ha chiuso la finestra (termina)
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Rilascia le risorse grafiche."""
        ...
