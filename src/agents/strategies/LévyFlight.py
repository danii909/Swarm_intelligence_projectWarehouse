"""
Strategia 3 — Lévy Flight Explorer.

Alterna esplorazione locale (frontier-based) a salti lunghi verso
zone inesplorate distanti. La probabilità di salto aumenta al crescere
della densità locale di celle già esplorate nel raggio di visibilità.

Inspirazione: i pattern di ricerca ottimale in natura (voli di Lévy)
alternano fasi di exploit locale e exploit globale, risultando più
efficienti del random walk puro su spazi con risorse sparse.

Adattamento al raggio di visibilità:
  - radius=1 → 4 celle locali, riempimento rapido → salti frequenti
  - radius=3 → 24 celle locali, riempimento lento → salti rari
"""

from __future__ import annotations

import random
from typing import Optional, Set, Tuple, TYPE_CHECKING

from src.agents.strategies.base import ExplorationStrategy

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder


class LevyFlightStrategy(ExplorationStrategy):
    """
    Esplorazione adattiva: locale (frontier) quando c'è ancora da vedere
    nelle vicinanze, salto lungo quando la densità locale è satura.
    """

    _JUMP_THRESHOLD: float = 0.75   # salta se >=75% delle celle locali esplorata
    _JUMP_MIN_DIST:  int   = 8      # distanza Manhattan minima del salto

    def next_move(
        self,
        agent: "Agent",
        env: "Environment",
        pathfinder: "Pathfinder",
        occupied: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:

        move = self._priority_move(agent, env, pathfinder, occupied)
        if move:
            return move

        # --- Salta se la zona locale è già satura ---
        if self._local_density(agent, env) >= self._JUMP_THRESHOLD:
            target = self._long_jump_target(agent, env)
            if target:
                step = pathfinder.next_step(agent.pos, target, occupied - {agent.pos})
                if step:
                    return step

        # --- Esplorazione locale: target di copertura piu' vicino ---
        targets = self._coverage_targets(agent, env)
        if targets:
            best = min(
                targets,
                key=lambda p: abs(p[0] - agent.row) + abs(p[1] - agent.col),
            )
            step = pathfinder.next_step(agent.pos, best, occupied - {agent.pos})
            if step:
                return step

        # --- Fallback: random walk ---
        neighbors = env.grid.walkable_neighbors(agent.row, agent.col)
        free = [n for n in neighbors if n not in occupied]
        return random.choice(free) if free else (random.choice(neighbors) if neighbors else None)

    # ------------------------------------------------------------------

    def _local_density(self, agent: "Agent", env: "Environment") -> float:
        """
        Frazione di celle nel raggio di visibilita' gia' scansionate per oggetti.
        """
        r, c = agent.pos
        radius = agent.visibility_radius
        size = env.grid.size
        seen_cells = agent.seen_cells
        total = known = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    total += 1
                    if (nr, nc) in seen_cells:
                        known += 1
        return known / total if total > 0 else 1.0

    def _long_jump_target(
        self, agent: "Agent", env: "Environment"
    ) -> Optional[Tuple[int, int]]:
        """
        Sceglie un target di copertura distante a sufficienza.
        """
        r, c = agent.pos
        min_dist = self._JUMP_MIN_DIST
        targets = self._coverage_targets(agent, env)
        if not targets:
            return None
        far = [p for p in targets if abs(p[0] - r) + abs(p[1] - c) >= min_dist]
        if far:
            return random.choice(far)
        return random.choice(list(targets))


# Alias per compatibilità con il codice esistente
SpiralStrategy = LevyFlightStrategy
