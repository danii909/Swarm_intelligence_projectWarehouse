"""
Strategia 3 — Esplorazione Sistematica a Spirale / Zigzag.

L'agente attraversa a serpentina (riga per riga) la griglia, garantendo
massima copertura sistematica. Utile come benchmark di esplorazione completa.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.environment.grid import CellType

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class SpiralStrategy(ExplorationStrategy):
    """
    Percorre la griglia a zigzag (riga-per-riga, alternando direzione).
    Priorità a raccolta/consegna se coglie oggetti lungo il percorso.
    """

    def __init__(self) -> None:
        self._waypoints: list = []
        self._wp_index: int = 0

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

        # --- Genera waypoint se necessario ---
        if not self._waypoints:
            self._build_waypoints(env)

        # --- Segui i waypoint ---
        while self._wp_index < len(self._waypoints):
            wp = self._waypoints[self._wp_index]
            if agent.pos == wp:
                self._wp_index += 1
                continue
            step = pathfinder.next_step(agent.pos, wp, occupied - {agent.pos})
            if step:
                return step
            # waypoint irraggiungibile: salta
            self._wp_index += 1

        # Tutti i waypoint visitati: rally a [0,0]
        step = pathfinder.next_step(agent.pos, (0, 0), occupied - {agent.pos})
        return step

    # ------------------------------------------------------------------

    def _build_waypoints(self, env: "Environment") -> None:
        """
        Costruisce una lista di waypoint a zigzag sulla griglia.
        Considera solo celle EMPTY (non le porte dei magazzini),
        una ogni 2 righe come checkpoint.
        """
        size = env.grid.size
        wps = []
        for r in range(0, size, 2):
            cols = range(0, size) if r % 4 == 0 else range(size - 1, -1, -1)
            for c in cols:
                if env.grid.cell(r, c) == CellType.EMPTY:
                    wps.append((r, c))
        self._waypoints = wps
        self._wp_index = 0
