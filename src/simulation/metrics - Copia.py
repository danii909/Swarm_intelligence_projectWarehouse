"""
Raccolta e calcolo delle metriche di simulazione.

Metriche richieste dalla specifica:
  1. Oggetti consegnati correttamente al magazzino
  2. Tempo totale (tick)
  3. Energia media consumata dagli agenti
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, TYPE_CHECKING, Optional, Set, Tuple

import math
from statistics import median

from src.environment.grid import CellType

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
    n_communicating_pairs: int = 0
    n_move_requests: int = 0
    n_moves_executed: int = 0
    n_conflicts: int = 0
    n_unique_empty_cells_seen_global: int = 0


@dataclass
class Metrics:
    """Raccoglie e calcola le metriche di simulazione."""

    total_ticks: int = 0
    objects_delivered: int = 0
    total_objects: int = 0
    agent_steps: List[int] = field(default_factory=list)
    agent_delivered: List[int] = field(default_factory=list)
    agent_initial_batteries: List[int] = field(default_factory=list)
    agent_final_batteries: List[int] = field(default_factory=list)
    history: List[TickSnapshot] = field(default_factory=list)

    first_pickup_tick: Optional[int] = None
    first_delivery_tick: Optional[int] = None
    completion_tick: Optional[int] = None

    total_move_requests: int = 0
    total_moves_executed: int = 0
    total_conflicts: int = 0
    conflict_ticks: int = 0

    total_comm_pairs: int = 0
    total_network_density: float = 0.0
    network_density_samples: int = 0

    total_active_agent_slots: int = 0
    total_idle_agent_slots: int = 0

    state_time_counts: Dict[str, int] = field(default_factory=dict)

    empty_cells_total: int = 0
    empty_cells_seen_unique: Set[Tuple[int, int]] = field(default_factory=set)
    total_empty_visits: int = 0
    empty_visit_counts: Dict[str, int] = field(default_factory=dict)

    delivery_trip_times: List[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Inizializzazione
    # ------------------------------------------------------------------

    def initialize(
        self,
        agents: List["Agent"],
        total_objects: int,
        empty_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        self.total_objects = total_objects
        self.agent_initial_batteries = [a.INITIAL_BATTERY for a in agents]
        self.empty_cells_total = len(empty_cells or set())

    # ------------------------------------------------------------------
    # Aggiornamento per-tick
    # ------------------------------------------------------------------

    def record_tick(
        self,
        tick: int,
        agents: List["Agent"],
        env: "Environment",
        visible_by_agent: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        communicating_pairs: int = 0,
        move_requests: int = 0,
        moves_executed: int = 0,
        conflicts: int = 0,
        log: bool = False,
    ) -> None:
        self.total_ticks = tick
        self.objects_delivered = env.delivered

        if self.first_pickup_tick is None and any(a.carrying_object for a in agents):
            self.first_pickup_tick = tick
        if self.first_delivery_tick is None and env.delivered > 0:
            self.first_delivery_tick = tick
        if self.completion_tick is None and self.total_objects > 0 and env.delivered >= self.total_objects:
            self.completion_tick = tick

        self.total_move_requests += move_requests
        self.total_moves_executed += moves_executed
        self.total_conflicts += conflicts
        if conflicts > 0:
            self.conflict_ticks += 1

        self.total_comm_pairs += communicating_pairs
        active_agents = sum(1 for a in agents if a.is_active)
        self.total_active_agent_slots += active_agents
        self.total_idle_agent_slots += max(active_agents - moves_executed, 0)

        for agent in agents:
            state_name = agent.state.name
            self.state_time_counts[state_name] = self.state_time_counts.get(state_name, 0) + 1

        possible_edges = active_agents * (active_agents - 1) / 2
        density = (communicating_pairs / possible_edges) if possible_edges > 0 else 0.0
        self.total_network_density += density
        self.network_density_samples += 1

        if visible_by_agent:
            for cells in visible_by_agent.values():
                for (r, c) in cells:
                    if env.grid.cell(r, c) != CellType.EMPTY:
                        continue
                    self.empty_cells_seen_unique.add((r, c))
                    self.total_empty_visits += 1
                    key = f"{r},{c}"
                    self.empty_visit_counts[key] = self.empty_visit_counts.get(key, 0) + 1

        if log:
            snapshot = TickSnapshot(
                tick=tick,
                delivered=env.delivered,
                remaining=env.remaining_objects,
                agent_positions=[a.pos for a in agents],
                agent_batteries=[a.battery for a in agents],
                agent_states=[a.state.name for a in agents],
                n_communicating_pairs=communicating_pairs,
                n_move_requests=move_requests,
                n_moves_executed=moves_executed,
                n_conflicts=conflicts,
                n_unique_empty_cells_seen_global=len(self.empty_cells_seen_unique),
            )
            self.history.append(snapshot)

    # ------------------------------------------------------------------
    # Finalizzazione
    # ------------------------------------------------------------------

    def finalize(self, agents: List["Agent"]) -> None:
        self.agent_final_batteries = [a.battery for a in agents]
        self.agent_steps = [a.steps_taken for a in agents]
        self.agent_delivered = [a.objects_delivered for a in agents]

    def record_delivery_trip_time(self, trip_ticks: int) -> None:
        """Aggiunge la durata pickup->delivery per un oggetto consegnato."""
        if trip_ticks >= 0:
            self.delivery_trip_times.append(trip_ticks)

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
    def total_energy_consumed(self) -> float:
        if not self.agent_initial_batteries:
            return 0.0
        return float(
            sum(
                init - final
                for init, final in zip(
                    self.agent_initial_batteries, self.agent_final_batteries
                )
            )
        )

    @property
    def delivery_rate(self) -> float:
        if self.total_objects == 0:
            return 0.0
        return self.objects_delivered / self.total_objects

    @property
    def completion_rate(self) -> float:
        return self.delivery_rate

    @property
    def throughput(self) -> float:
        if self.total_ticks <= 0:
            return 0.0
        return self.objects_delivered / self.total_ticks

    @property
    def energy_per_object(self) -> float:
        return self.total_energy_consumed / max(1, self.objects_delivered)

    @property
    def coverage_final(self) -> float:
        if self.empty_cells_total <= 0:
            return 0.0
        return len(self.empty_cells_seen_unique) / self.empty_cells_total

    @property
    def redundancy_index(self) -> float:
        unique = len(self.empty_cells_seen_unique)
        if unique <= 0:
            return 0.0
        return self.total_empty_visits / unique

    @property
    def blocked_move_rate(self) -> float:
        if self.total_move_requests <= 0:
            return 0.0
        blocked = self.total_move_requests - self.total_moves_executed
        return max(blocked, 0) / self.total_move_requests

    @property
    def conflict_rate(self) -> float:
        if self.total_ticks <= 0:
            return 0.0
        return self.conflict_ticks / self.total_ticks

    @property
    def mean_pairs_communicating(self) -> float:
        if self.total_ticks <= 0:
            return 0.0
        return self.total_comm_pairs / self.total_ticks

    @property
    def network_density(self) -> float:
        if self.network_density_samples <= 0:
            return 0.0
        return self.total_network_density / self.network_density_samples

    @property
    def idle_ratio(self) -> float:
        if self.total_active_agent_slots <= 0:
            return 0.0
        return self.total_idle_agent_slots / self.total_active_agent_slots

    @property
    def average_delivery_trip_time(self) -> float:
        if not self.delivery_trip_times:
            return 0.0
        return float(sum(self.delivery_trip_times) / len(self.delivery_trip_times))

    @staticmethod
    def _cv(values: List[int]) -> float:
        valid = [float(v) for v in values if v is not None]
        if not valid:
            return 0.0
        m = sum(valid) / len(valid)
        if math.isclose(m, 0.0):
            return 0.0
        var = sum((v - m) ** 2 for v in valid) / len(valid)
        return math.sqrt(var) / m

    def state_occupancy(self) -> Dict[str, float]:
        """Ritorna la quota di tempo per stato agente, in [0,1]."""
        if self.total_ticks <= 0 or not self.state_time_counts:
            return {}
        denom = sum(self.state_time_counts.values())
        if denom <= 0:
            return {}
        return {
            state: count / denom
            for state, count in sorted(self.state_time_counts.items())
        }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        occupancy = self.state_occupancy()
        completion_time = self.completion_tick if self.completion_tick is not None else self.total_ticks
        return {
            "objects_delivered": self.objects_delivered,
            "total_objects": self.total_objects,
            "delivery_rate": round(self.delivery_rate, 3),
            "completion_rate": round(self.completion_rate, 3),
            "completed": bool(self.total_objects > 0 and self.objects_delivered >= self.total_objects),
            "completion_time": completion_time,
            "completion_time_censored": self.completion_tick is None,
            "total_ticks": self.total_ticks,
            "average_energy_consumed": round(self.average_energy_consumed, 2),
            "total_energy_consumed": round(self.total_energy_consumed, 2),
            "throughput": round(self.throughput, 4),
            "energy_per_object": round(self.energy_per_object, 4),
            "first_pickup_tick": self.first_pickup_tick,
            "first_delivery_tick": self.first_delivery_tick,
            "coverage_final": round(self.coverage_final, 4),
            "redundancy_index": round(self.redundancy_index, 4),
            "conflict_rate": round(self.conflict_rate, 4),
            "blocked_move_rate": round(self.blocked_move_rate, 4),
            "mean_pairs_communicating": round(self.mean_pairs_communicating, 4),
            "network_density": round(self.network_density, 4),
            "idle_ratio": round(self.idle_ratio, 4),
            "delivery_trip_time_avg": round(self.average_delivery_trip_time, 3),
            "cv_steps": round(self._cv(self.agent_steps), 4),
            "cv_delivered": round(self._cv(self.agent_delivered), 4),
            "agent_steps": self.agent_steps,
            "agent_delivered": self.agent_delivered,
            "agent_final_batteries": self.agent_final_batteries,
            "state_occupancy": {k: round(v, 4) for k, v in occupancy.items()},
            "empty_cells_total": self.empty_cells_total,
            "empty_cells_unique_seen": len(self.empty_cells_seen_unique),
            "total_empty_visits": self.total_empty_visits,
            "median_delivery_trip_time": round(median(self.delivery_trip_times), 3)
            if self.delivery_trip_times else 0.0,
            "empty_visit_counts": self.empty_visit_counts,
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
