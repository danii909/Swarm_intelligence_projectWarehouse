"""
Strategia — Warehouse-Centric Exploration.

Esplora in onde concentriche partendo dai magazzini (prima le celle vicine,
poi quelle più lontane). L'ipotesi è che in scenari realistici gli oggetti
siano spesso vicini ai magazzini (zone di carico/scarico).

Vantaggi:
- Trova velocemente oggetti vicini ai warehouse
- Riduce distanza media di consegna
- Sfrutta la struttura tipica dei warehouse

Svantaggi:
- Pessimo se oggetti sono lontani dai magazzini
- Può creare congestione vicino ai warehouse
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.environment.grid import CellType

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class WarehouseCentricStrategy(ExplorationStrategy):
    """
    Esplora a cerchi concentrici espandendosi dai magazzini verso l'esterno.
    """

    def __init__(self) -> None:
        # Cache delle distanze: (r, c) → min_distance_to_warehouse
        self._distance_cache: Dict[Tuple[int, int], int] = {}

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

        # --- Inizializza cache distanze se necessario ---
        if not self._distance_cache:
            self._compute_warehouse_distances(agent, env)

        # --- Trova celle non visitate, ordinate per distanza da warehouse ---
        unvisited_empty = [
            (r, c) for (r, c), cell_type in agent.local_map.items()
            if cell_type == CellType.EMPTY and (r, c) not in agent.visited_cells
        ]

        if not unvisited_empty:
            # Tutta la mappa visitata
            neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
            free = [n for n in neighbors if n not in occupied]
            return free[0] if free else (neighbors[0] if neighbors else None)

        # Trova cella non visitata con minima distanza da warehouse
        # In caso di pareggio, scegli la più vicina all'agente
        target = min(
            unvisited_empty,
            key=lambda p: (
                self._distance_cache.get(p, 9999),  # Prima: vicinanza a warehouse
                abs(p[0] - agent.row) + abs(p[1] - agent.col)  # Poi: vicinanza ad agente
            )
        )

        step = pathfinder.next_step(agent.pos, target, occupied - {agent.pos})
        return step

    def _compute_warehouse_distances(
        self, agent: "Agent", env: "Environment"
    ) -> None:
        """
        Calcola la distanza Manhattan minima di ogni cella EMPTY
        dal warehouse più vicino (entrance o exit).
        """
        # Raccogli tutte le porte dei warehouse
        warehouse_doors: Set[Tuple[int, int]] = set()
        for wh in env.warehouses:
            warehouse_doors.add(wh.entrance)
            warehouse_doors.add(wh.exit)

        # Calcola per ogni cella EMPTY la distanza minima
        for (r, c), cell_type in agent.local_map.items():
            if cell_type != CellType.EMPTY:
                continue

            min_dist = min(
                abs(r - dr) + abs(c - dc)
                for (dr, dc) in warehouse_doors
            )
            self._distance_cache[(r, c)] = min_dist
