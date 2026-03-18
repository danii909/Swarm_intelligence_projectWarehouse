from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

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

    def __init__(self) -> None:
        # Cache frontiere di copertura: {agent_id: (frontier_set, seen_size_al_calcolo)}
        self._frontier_cache: Dict[int, Tuple[Set, int]] = {}

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
        Frontiere di copertura: celle EMPTY gia' viste che confinano con
        celle EMPTY non ancora viste. Cached per agent_id finche' seen_cells
        non cresce.
        """
        seen_size = len(agent.seen_cells)
        cached = self._frontier_cache.get(agent.id)
        if cached is not None and cached[1] == seen_size:
            return cached[0]

        frontiers: Set[Tuple[int, int]] = set()
        seen_cells = agent.seen_cells
        in_bounds = env.grid.in_bounds
        for r, c in seen_cells:
            if env.grid.cell(r, c) != CellType.EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue
                if env.grid.cell(nr, nc) != CellType.EMPTY:
                    continue
                if (nr, nc) not in seen_cells:
                    frontiers.add((r, c))
                    break

        self._frontier_cache[agent.id] = (frontiers, seen_size)
        return frontiers

    def _unexplored_empty(
        self,
        agent: "Agent",
        env: "Environment",
    ) -> Set[Tuple[int, int]]:
        """
        Celle candidate per la ricerca pacchi non ancora osservate.

        La mappa e' nota a priori, ma i pacchi no: la copertura si misura
        sulle celle EMPTY gia' scansionate via sensori.
        """
        return set(env.grid.empty_cells) - agent.seen_cells

    def _stale_empty(
        self,
        agent: "Agent",
        env: "Environment",
        min_age: int = 12,
    ) -> Set[Tuple[int, int]]:
        """
        Celle EMPTY osservate in passato ma non recentemente.

        Usata come fallback quando tutta la mappa e' stata vista almeno una volta.
        """
        tick = env.tick
        stale: Set[Tuple[int, int]] = set()
        for pos in env.grid.empty_cells:
            last_seen = agent.cell_last_seen.get(pos)
            if last_seen is None:
                continue
            if tick - last_seen >= min_age:
                stale.add(pos)
        return stale

    def _coverage_targets(
        self,
        agent: "Agent",
        env: "Environment",
    ) -> Set[Tuple[int, int]]:
        """
        Target di ricerca ordinati per priorita':
          1. Celle EMPTY mai osservate
          2. Celle EMPTY stale (non viste da un po')
          3. Tutte le celle EMPTY (patrolling)
        """
        unseen = self._unexplored_empty(agent, env)
        if unseen:
            return unseen

        stale = self._stale_empty(agent, env, min_age=max(8, env.grid.size // 2))
        if stale:
            return stale

        return set(env.grid.empty_cells)

    @property
    def name(self) -> str:
        return self.__class__.__name__
