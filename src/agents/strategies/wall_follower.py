"""
Strategia — Wall Following (Perimetri e Bordi).

Segue i perimetri dei muri e i bordi della griglia. L'ipotesi è che in
warehouse reali gli oggetti siano spesso posizionati lungo scaffali,
pareti, o agli angoli.

Vantaggi:
- Molto efficace se oggetti sono ai bordi/scaffali
- Pattern di movimento sistematico e prevedibile
- Usa la mappa completa per evitare loop infiniti

Svantaggi:
- Lento se oggetti sono al centro di zone aperte
- Può ignorare aree interne ampie
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy
from src.environment.grid import CellType, DIRECTIONS

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class WallFollowerStrategy(ExplorationStrategy):
    """
    Segue i perimetri dei muri (celle EMPTY adiacenti a WALL).
    Dopo aver coperto i perimetri, esplora l'interno.
    """

    def __init__(self) -> None:
        # Cache delle celle perimetrali (adiacenti a muri)
        self._perimeter_cells: Set[Tuple[int, int]] = set()
        self._perimeter_computed: bool = False

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

        # --- Calcola celle perimetrali se necessario ---
        if not self._perimeter_computed:
            self._compute_perimeter_cells(agent, env)
            self._perimeter_computed = True

        # --- Priorità 1: celle perimetrali non visitate ---
        unvisited_perimeter = [
            (r, c) for (r, c) in self._perimeter_cells
            if (r, c) not in agent.visited_cells
        ]

        if unvisited_perimeter:
            # Vai alla cella perimetrale più vicina
            target = min(
                unvisited_perimeter,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
            step = pathfinder.next_step(agent.pos, target, occupied - {agent.pos})
            return step

        # --- Priorità 2: celle interne non visitate ---
        unvisited_interior = [
            (r, c) for (r, c), cell_type in agent.local_map.items()
            if cell_type == CellType.EMPTY
            and (r, c) not in agent.visited_cells
            and (r, c) not in self._perimeter_cells
        ]

        if unvisited_interior:
            # Vai alla cella interna più vicina
            target = min(
                unvisited_interior,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
            step = pathfinder.next_step(agent.pos, target, occupied - {agent.pos})
            return step

        # --- Tutto visitato ---
        neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
        free = [n for n in neighbors if n not in occupied]
        return free[0] if free else (neighbors[0] if neighbors else None)

    def _compute_perimeter_cells(
        self, agent: "Agent", env: "Environment"
    ) -> None:
        """
        Calcola tutte le celle EMPTY che sono adiacenti (4-connesso)
        a muri, bordi della griglia, o altre celle non percorribili.

        Queste sono le celle "perimetrali" da esplorare per prime.
        """
        size = env.grid.size

        for (r, c), cell_type in agent.local_map.items():
            if cell_type != CellType.EMPTY:
                continue

            # Verifica se è adiacente a muro o bordo
            is_perimeter = False

            # Bordi della griglia
            if r == 0 or r == size - 1 or c == 0 or c == size - 1:
                is_perimeter = True
            else:
                # Adiacente a muro o cella non walkable?
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not env.grid.in_bounds(nr, nc):
                        continue
                    neighbor_type = agent.local_map.get((nr, nc), CellType.EMPTY)
                    # WALL o WAREHOUSE sono considerati "muri"
                    if neighbor_type in (CellType.WALL, CellType.WAREHOUSE):
                        is_perimeter = True
                        break

            if is_perimeter:
                self._perimeter_cells.add((r, c))
