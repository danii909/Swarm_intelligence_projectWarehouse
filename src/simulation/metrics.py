"""
Raccolta e calcolo delle metriche di simulazione.

Metriche richieste dalla specifica:
  1. Oggetti consegnati correttamente al magazzino
  2. Tempo totale (tick)
  3. Energia media consumata dagli agenti
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment


@dataclass
class TickSnapshot:
    tick: int
    delivered: int
    remaining: int
    agent_positions: List[tuple]
    agent_batteries: List[int]
    agent_states: List[str]


@dataclass
class Metrics:
    """Raccoglie e calcola le metriche di simulazione."""

    total_ticks: int = 0
    objects_delivered: int = 0
    total_objects: int = 0
    agent_steps: List[int] = field(default_factory=list)
    agent_initial_batteries: List[int] = field(default_factory=list)
    agent_final_batteries: List[int] = field(default_factory=list)
    history: List[TickSnapshot] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Inizializzazione
    # ------------------------------------------------------------------

    def initialize(self, agents: List["Agent"], total_objects: int) -> None:
        self.total_objects = total_objects
        self.agent_initial_batteries = [a.INITIAL_BATTERY for a in agents]

    # ------------------------------------------------------------------
    # Aggiornamento per-tick
    # ------------------------------------------------------------------

    def record_tick(
        self,
        tick: int,
        agents: List["Agent"],
        env: "Environment",
        log: bool = False,
    ) -> None:
        self.total_ticks = tick
        self.objects_delivered = env.delivered

        if log:
            snapshot = TickSnapshot(
                tick=tick,
                delivered=env.delivered,
                remaining=env.remaining_objects,
                agent_positions=[a.pos for a in agents],
                agent_batteries=[a.battery for a in agents],
                agent_states=[a.state.name for a in agents],
            )
            self.history.append(snapshot)

    # ------------------------------------------------------------------
    # Finalizzazione
    # ------------------------------------------------------------------

    def finalize(self, agents: List["Agent"]) -> None:
        self.agent_final_batteries = [a.battery for a in agents]
        self.agent_steps = [a.steps_taken for a in agents]

    # ------------------------------------------------------------------
    # Calcoli
    # ------------------------------------------------------------------

    @property
    def average_energy_consumed(self) -> float:
        if not self.agent_initial_batteries:
            return 0.0
        consumed = [
            init - final
            for init, final in zip(
                self.agent_initial_batteries, self.agent_final_batteries
            )
        ]
        return sum(consumed) / len(consumed)

    @property
    def delivery_rate(self) -> float:
        if self.total_objects == 0:
            return 0.0
        return self.objects_delivered / self.total_objects

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "objects_delivered": self.objects_delivered,
            "total_objects": self.total_objects,
            "delivery_rate": round(self.delivery_rate, 3),
            "total_ticks": self.total_ticks,
            "average_energy_consumed": round(self.average_energy_consumed, 2),
            "agent_steps": self.agent_steps,
            "agent_final_batteries": self.agent_final_batteries,
        }

    def print_summary(self) -> None:  # pragma: no cover
        s = self.summary()
        print("\n=== Risultati Simulazione ===")
        print(f"  Oggetti consegnati : {s['objects_delivered']} / {s['total_objects']}")
        print(f"  Tasso consegna     : {s['delivery_rate']*100:.1f}%")
        print(f"  Tick totali        : {s['total_ticks']}")
        print(f"  Energia media cons.: {s['average_energy_consumed']:.1f} unità")
        print(f"  Passi per agente   : {s['agent_steps']}")
        print("=" * 30)
