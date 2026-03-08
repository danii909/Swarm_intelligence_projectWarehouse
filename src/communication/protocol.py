"""
Protocollo di comunicazione tra agenti.

Due agenti si scambiano informazioni quando la distanza Manhattan
tra le loro posizioni è ≤ raggio di comunicazione.

Informazioni condivise:
  1. Mappa locale esplorata (merge bidirezionale)
  2. Oggetti rilevati ma non ancora raccolti
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from src.agents.sensors import can_communicate

if TYPE_CHECKING:
    from src.agents.agent import Agent


def communicate_agents(agents: List["Agent"]) -> int:
    """
    Esegue lo scambio di informazioni tra tutti i coppie di agenti
    che si trovano entro il raggio di comunicazione reciproco.

    Restituisce il numero di coppie che hanno comunicato in questo tick.
    """
    n = len(agents)
    pairs_communicated = 0

    for i in range(n):
        a = agents[i]
        if not a.is_active:
            continue
        for j in range(i + 1, n):
            b = agents[j]
            if not b.is_active:
                continue
            comm_radius = min(a.comm_radius, b.comm_radius)
            if can_communicate(a.pos, b.pos, comm_radius):
                _exchange(a, b)
                pairs_communicated += 1

    return pairs_communicated


def _exchange(a: "Agent", b: "Agent") -> None:
    """Merge bidirezionale di mappa locale, oggetti noti e posizioni agenti."""
    # Merge mappa locale
    merged_map = {**b.local_map, **a.local_map}
    a.local_map = merged_map.copy()
    b.local_map = merged_map.copy()

    # Merge oggetti noti
    merged_objects = a.known_objects | b.known_objects
    a.known_objects = merged_objects.copy()
    b.known_objects = merged_objects.copy()

    # Merge posizioni agenti noti: per ogni ID teniamo la voce più recente
    merged_agents = dict(a.known_agents)
    for agent_id, entry in b.known_agents.items():
        if agent_id not in merged_agents or merged_agents[agent_id][1] < entry[1]:
            merged_agents[agent_id] = entry
    a.known_agents = merged_agents.copy()
    b.known_agents = merged_agents.copy()
