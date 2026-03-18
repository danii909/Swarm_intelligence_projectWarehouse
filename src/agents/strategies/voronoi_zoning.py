"""
Strategia — Voronoi Zoning (Divisione Cooperativa).

Divide dinamicamente lo spazio usando i diagrammi di Voronoi basati
sulle posizioni correnti degli agenti. Ogni agente esplora solo la propria
zona di Voronoi (celle più vicine a lui che ad altri agenti).

Vantaggi:
- Massima dispersione e copertura
- Zero overlap tra agenti
- Si adatta dinamicamente se un agente va KO

Svantaggi:
- Richiede comunicazione frequente (usa known_agents)
- Overhead computazionale per calcolo Voronoi
- Può creare zone sbilanciare se agenti sono mal distribuiti
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.environment.grid import CellType

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class VoronoiZoningStrategy(ExplorationStrategy):
    """
    Divide lo spazio in celle di Voronoi basate sulle posizioni degli agenti.
    Ogni agente esplora solo la propria cella di Voronoi.
    """

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

        # --- Calcola zona di Voronoi di questo agente ---
        my_voronoi_zone = self._compute_voronoi_zone(agent, env)

        # --- Trova celle non visitate nella mia zona ---
        unvisited_in_zone = [
            (r, c) for (r, c) in my_voronoi_zone
            if (r, c) not in agent.visited_cells
        ]

        if not unvisited_in_zone:
            # Zona completamente visitata: esplora fuori zona (nearest unvisited globale)
            unvisited_global = [
                (r, c) for (r, c), cell_type in agent.local_map.items()
                if cell_type == CellType.EMPTY and (r, c) not in agent.visited_cells
            ]

            if not unvisited_global:
                # Tutto visitato
                neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
                free = [n for n in neighbors if n not in occupied]
                return free[0] if free else (neighbors[0] if neighbors else None)

            # Vai alla più vicina fuori zona
            target = min(
                unvisited_global,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
        else:
            # Vai alla cella non visitata più vicina nella mia zona
            target = min(
                unvisited_in_zone,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )

        step = pathfinder.next_step(agent.pos, target, occupied - {agent.pos})
        return step

    def _compute_voronoi_zone(
        self, agent: "Agent", env: "Environment"
    ) -> Set[Tuple[int, int]]:
        """
        Calcola la cella di Voronoi per questo agente: tutte le celle EMPTY
        che sono più vicine a questo agente che ad altri agenti conosciuti.

        Usa agent.known_agents per le posizioni degli altri agenti.
        """
        voronoi_zone: Set[Tuple[int, int]] = set()

        # Posizioni di tutti gli agenti conosciuti (incluso questo)
        agent_positions: Set[Tuple[int, int]] = {agent.pos}

        for other_id, (other_pos, _tick) in agent.known_agents.items():
            if other_id != agent.id:
                agent_positions.add(other_pos)

        # Se non conosce altri agenti, tutta la griglia è sua zona
        if len(agent_positions) == 1:
            return {
                (r, c) for (r, c), cell_type in agent.local_map.items()
                if cell_type == CellType.EMPTY
            }

        # Per ogni cella EMPTY, verifica se è più vicina a questo agente
        for (r, c), cell_type in agent.local_map.items():
            if cell_type != CellType.EMPTY:
                continue

            # Distanza da questo agente
            my_dist = abs(r - agent.row) + abs(c - agent.col)

            # Distanze da altri agenti
            is_mine = True
            for other_pos in agent_positions:
                if other_pos == agent.pos:
                    continue
                other_dist = abs(r - other_pos[0]) + abs(c - other_pos[1])
                if other_dist < my_dist:
                    is_mine = False
                    break

            if is_mine:
                voronoi_zone.add((r, c))

        return voronoi_zone
