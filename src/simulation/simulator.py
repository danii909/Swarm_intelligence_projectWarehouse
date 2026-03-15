"""
Loop principale della simulazione.

Ad ogni tick:
  1. Ogni agente percepisce l'ambiente (aggiorna mappa locale)
  2. Gli agenti comunicano tra loro se entro raggio
  3. Ogni agente decide la prossima mossa
  4. Le mosse vengono applicate (con gestione collisioni)
  5. Gli agenti raccolgono/consegnano oggetti
  6. Le metriche vengono aggiornate
  7. Si verifica la condizione di terminazione
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from src.environment.environment import Environment
from src.environment.grid import CellType
from src.agents.agent import Agent, AgentState
from src.agents.strategies.random_walk import RandomWalkStrategy
from src.agents.strategies.frontier import FrontierStrategy
from src.agents.strategies.spiral import SpiralStrategy
from src.agents.strategies.sector import SectorStrategy
from src.agents.strategies.greedy import GreedyStrategy
from src.communication.protocol import communicate_agents
from src.pathfinding.pathfinder import Pathfinder
from src.simulation.metrics import Metrics


# ---------------------------------------------------------------------------
# Factory agenti di default
# ---------------------------------------------------------------------------

def _create_default_agents(num_agents: int = 5) -> List[Agent]:
    """Crea 5 agenti, uno per strategia, con raggio visibilità variato."""
    strategies = [
        RandomWalkStrategy(),
        FrontierStrategy(),
        SpiralStrategy(),
        SectorStrategy(num_agents=num_agents),
        GreedyStrategy(),
    ]
    agents = []
    for i in range(num_agents):
        strategy = strategies[i % len(strategies)]
        # Raggio di visibilità varia tra 1 e 3
        vis_radius = 1 + (i % 3)
        agents.append(Agent(agent_id=i, strategy=strategy, visibility_radius=vis_radius))
    return agents


# ---------------------------------------------------------------------------
# Simulatore
# ---------------------------------------------------------------------------

class Simulator:
    """
    Coordina l'esecuzione della simulazione multi-agente.

    Parameters
    ----------
    env        : Environment      — ambiente già caricato da JSON
    agents     : List[Agent]      — se None, vengono creati 5 agenti di default
    max_ticks  : int              — numero massimo di tick (default 500)
    seed       : int | None       — seed per la riproducibilità
    verbose    : bool             — stampa log a schermo
    log_every  : int              — ogni quanti tick salvare snapshot nelle metriche
    visualizer : BaseVisualizer | None — se fornito, chiama update() ad ogni tick
    """

    def __init__(
        self,
        env: Environment,
        agents: Optional[List[Agent]] = None,
        max_ticks: int = 500,
        seed: Optional[int] = None,
        verbose: bool = True,
        log_every: int = 10,
        visualizer=None,
    ) -> None:
        if seed is not None:
            random.seed(seed)

        self.env = env
        self.agents: List[Agent] = agents or _create_default_agents()
        self.max_ticks = max_ticks
        self.verbose = verbose
        self.log_every = log_every
        self.visualizer = visualizer

        self.pathfinder = Pathfinder(env.grid)
        self.metrics = Metrics()
        self.metrics.initialize(self.agents, env.total_objects)

        self._tick: int = 0

    # ------------------------------------------------------------------
    # Esecuzione
    # ------------------------------------------------------------------

    def run(self) -> Metrics:
        """Esegue la simulazione completa. Restituisce le metriche finali."""
        if self.verbose:
            print(f"Avvio simulazione | agenti={len(self.agents)} "
                  f"| oggetti={self.env.total_objects} "
                  f"| max_tick={self.max_ticks}")

        # Inizializza visualizzatore (se presente)
        if self.visualizer is not None:
            self.visualizer.setup(self.env, self.agents)

        try:
            while self._tick < self.max_ticks:
                self._tick += 1
                self.env.advance_tick()

                # 1. Percezione
                for agent in self.agents:
                    if agent.is_active:
                        agent.perceive(self.env)

                # 1b. Aggiorna known_agents con agenti visibili direttamente
                for agent in self.agents:
                    if not agent.is_active:
                        continue
                    for other in self.agents:
                        if other.id == agent.id or not other.is_active:
                            continue
                        dist = abs(other.row - agent.row) + abs(other.col - agent.col)
                        if dist <= agent.visibility_radius:
                            agent.known_agents[other.id] = (other.pos, self._tick)

                # 2. Comunicazione (propaga anche known_agents transitivamente)
                communicate_agents(self.agents)

                # 3. Pianificazione mosse: ogni agente usa solo le posizioni
                #    che conosce (visione diretta o comunicazione), con scadenza
                _MAX_AGENT_INFO_AGE = 5
                moves: Dict[int, Optional[Tuple[int, int]]] = {}
                for agent in self.agents:
                    if agent.is_active:
                        locally_known = {
                            pos
                            for pos, tick in agent.known_agents.values()
                            if self._tick - tick <= _MAX_AGENT_INFO_AGE
                        } - {agent.pos}
                        moves[agent.id] = agent.decide_next_move(
                            self.env, self.pathfinder, locally_known
                        )

                # 4. Applica mosse (no-overlap: risolvi conflitti)
                self._apply_moves(moves)

                # 5. Raccolta / consegna oggetti
                for agent in self.agents:
                    if not agent.is_active:
                        continue
                    if agent.state == AgentState.EXITING_WAREHOUSE:
                        # Agente ha consegnato, sta uscendo: quando raggiunge EXIT
                        # torna in esplorazione
                        if self.env.grid.cell(agent.row, agent.col) == CellType.EXIT:
                            agent.state = AgentState.EXPLORING
                    elif not agent.carrying_object:
                        agent.pick_up(self.env)
                    else:
                        agent.deliver(self.env)

                # 6. Visualizzazione (se attiva)
                if self.visualizer is not None:
                    should_continue = self.visualizer.update(
                        self._tick, self.agents, self.env
                    )
                    if not should_continue:
                        break

                # 7. Metriche
                log_this_tick = (self._tick % self.log_every == 0)
                self.metrics.record_tick(
                    self._tick, self.agents, self.env, log=log_this_tick
                )

                # 8. Log verboso
                if self.verbose and log_this_tick:
                    print(
                        f"  Tick {self._tick:4d} | "
                        f"consegnati={self.env.delivered}/{self.env.total_objects} | "
                        f"rimanenti={self.env.remaining_objects}"
                    )

                # 9. Condizioni di terminazione
                if self.env.all_delivered:
                    if self.verbose:
                        print(f"\nTutti gli oggetti consegnati al tick {self._tick}!")
                    break
                # batteria esaurita: la simulazione continua fino a max_ticks

        finally:
            self.metrics.finalize(self.agents)
            if self.verbose:
                self.metrics.print_summary()
            if self.visualizer is not None:
                self.visualizer.close()
        return self.metrics

    # ------------------------------------------------------------------
    # Generatore tick-by-tick (per UI in tempo reale)
    # ------------------------------------------------------------------

    def step_gen(self):
        """
        Generatore che esegue la simulazione un tick alla volta.
        Ad ogni tick cede (tick, agents, env) per aggiornamenti UI.
        Le metriche vengono finalizzate automaticamente al termine.
        """
        try:
            while self._tick < self.max_ticks:
                self._tick += 1
                self.env.advance_tick()

                # 1. Percezione
                for agent in self.agents:
                    if agent.is_active:
                        agent.perceive(self.env)

                # 1b. Aggiorna known_agents con agenti visibili direttamente
                for agent in self.agents:
                    if not agent.is_active:
                        continue
                    for other in self.agents:
                        if other.id == agent.id or not other.is_active:
                            continue
                        dist = abs(other.row - agent.row) + abs(other.col - agent.col)
                        if dist <= agent.visibility_radius:
                            agent.known_agents[other.id] = (other.pos, self._tick)

                # 2. Comunicazione
                communicate_agents(self.agents)

                # 3. Pianificazione mosse
                _MAX_AGENT_INFO_AGE = 5
                moves: Dict[int, Optional[Tuple[int, int]]] = {}
                for agent in self.agents:
                    if agent.is_active:
                        locally_known = {
                            pos
                            for pos, tick in agent.known_agents.values()
                            if self._tick - tick <= _MAX_AGENT_INFO_AGE
                        } - {agent.pos}
                        moves[agent.id] = agent.decide_next_move(
                            self.env, self.pathfinder, locally_known
                        )

                # 4. Applica mosse
                self._apply_moves(moves)

                # 5. Raccolta / consegna oggetti
                for agent in self.agents:
                    if not agent.is_active:
                        continue
                    if agent.state == AgentState.EXITING_WAREHOUSE:
                        if self.env.grid.cell(agent.row, agent.col) == CellType.EXIT:
                            agent.state = AgentState.EXPLORING
                    elif not agent.carrying_object:
                        agent.pick_up(self.env)
                    else:
                        agent.deliver(self.env)

                # 6. Metriche
                log_this_tick = (self._tick % self.log_every == 0)
                self.metrics.record_tick(
                    self._tick, self.agents, self.env, log=log_this_tick
                )

                # 7. Cedi il controllo all'UI
                yield self._tick, self.agents, self.env

                # 8. Condizioni di terminazione
                if self.env.all_delivered:
                    break
                # batteria esaurita: la simulazione continua fino a max_ticks

        finally:
            self.metrics.finalize(self.agents)

    # ------------------------------------------------------------------
    # Gestione collisioni
    # ------------------------------------------------------------------

    def _apply_moves(
        self, moves: Dict[int, Optional[Tuple[int, int]]]
    ) -> None:
        """
        Applica le mosse evitando sovrapposizioni (progetto di gruppo).

        Regole di risoluzione conflitti:
        1. Un agente che non si muove (dest=None) occupa la sua cella corrente.
        2. Un agente in movimento non può spostarsi in una cella occupata da
           un agente stazionario.
        3. Se due agenti vogliono spostarsi nella stessa cella, si muove solo
           quello con ID minore; l'altro resta fermo.

        Nota: all'avvio tutti gli agenti partono da [0,0], quindi la
        sovrapposizione al tick 0 è inevitabile (consentita per progetto
        individuale dalla specifica).
        """
        agents_by_id = {a.id: a for a in self.agents}

        # Posizioni occupate da agenti fermi (dest=None)
        stationary_pos: Set[Tuple[int, int]] = set()
        for agent in self.agents:
            if agent.is_active and moves.get(agent.id) is None:
                stationary_pos.add(agent.pos)

        # Mappa destinazione → lista di agent_id (solo agenti in movimento)
        dest_map: Dict[Tuple[int, int], List[int]] = {}
        for agent_id, dest in moves.items():
            if dest is None:
                continue
            # Blocca se la destinazione è occupata da un agente fermo
            if dest in stationary_pos and dest != agents_by_id[agent_id].pos:
                continue
            dest_map.setdefault(dest, []).append(agent_id)

        # Risolvi conflitti tra agenti in movimento verso la stessa cella
        allowed: Set[int] = set()
        for dest, agent_ids in dest_map.items():
            if len(agent_ids) == 1:
                allowed.add(agent_ids[0])
            else:
                # Vince l'agente con ID minore
                allowed.add(min(agent_ids))

        # Esegui le mosse consentite
        for agent_id, dest in moves.items():
            if dest is not None and agent_id in allowed:
                agents_by_id[agent_id].move_to(*dest)
