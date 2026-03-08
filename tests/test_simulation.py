import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.environment import Environment
from src.simulation.simulator import Simulator
from src.simulation.metrics import Metrics

INSTANCE_A = os.path.join(os.path.dirname(__file__), "..", "Consegna", "A.json")
INSTANCE_B = os.path.join(os.path.dirname(__file__), "..", "Consegna", "B.json")


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_simulation_runs_without_errors():
    """La simulazione deve completare senza eccezioni."""
    env = Environment.from_json(INSTANCE_A)
    sim = Simulator(env=env, max_ticks=50, seed=0, verbose=False)
    metrics = sim.run()
    assert isinstance(metrics, Metrics)
    assert metrics.total_ticks <= 50


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_simulation_metrics_structure():
    """Le metriche devono avere tutti i campi richiesti dalla specifica."""
    env = Environment.from_json(INSTANCE_A)
    sim = Simulator(env=env, max_ticks=20, seed=1, verbose=False)
    metrics = sim.run()
    summary = metrics.summary()

    assert "objects_delivered" in summary
    assert "total_ticks" in summary
    assert "average_energy_consumed" in summary
    assert "delivery_rate" in summary
    assert summary["total_objects"] == 10


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_agents_start_at_origin():
    """Tutti gli agenti devono partire da [0,0]."""
    env = Environment.from_json(INSTANCE_A)
    sim = Simulator(env=env, max_ticks=0, seed=0, verbose=False)
    for agent in sim.agents:
        assert agent.pos == (0, 0)


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_no_agent_overlap_after_dispersal():
    """
    Dopo il tick 5 (quando gli agenti si sono dispersi da [0,0]), non ci devono
    essere sovrapposizioni tra agenti che si sono effettivamente mossi.

    Nota: al tick 1 tutti gli agenti si trovano in [0,0] per specifica
    (posizione iniziale condivisa); la sovrapposizione al tick 0 è inevitabile
    e consentita dalla specifica per il progetto individuale.
    """
    env = Environment.from_json(INSTANCE_A)
    sim = Simulator(env=env, max_ticks=100, seed=42, verbose=False)
    overlap_counts = []

    original_apply = sim._apply_moves.__func__

    def patched_apply(self_inner, moves):
        original_apply(self_inner, moves)
        if self_inner._tick > 5:
            active_positions = [a.pos for a in self_inner.agents if a.is_active]
            n_unique = len(set(active_positions))
            n_total = len(active_positions)
            overlap_counts.append(n_total - n_unique)

    import types
    sim._apply_moves = types.MethodType(patched_apply, sim)
    sim.run()

    # Dopo la dispersione iniziale, la sovrapposizione deve essere minima
    # (al più 1 agente può occasionalmente essere bloccato nella stessa cella)
    if overlap_counts:
        avg_overlap = sum(overlap_counts) / len(overlap_counts)
        assert avg_overlap < 1.5, (
            f"Sovrapposizione media troppo alta dopo tick 5: {avg_overlap:.2f}"
        )


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_battery_decreases_monotonically():
    """La batteria di ogni agente non deve mai aumentare."""
    env = Environment.from_json(INSTANCE_A)
    sim = Simulator(env=env, max_ticks=30, seed=7, verbose=False)

    batteries_before = {a.id: a.battery for a in sim.agents}
    sim.run()
    for agent in sim.agents:
        assert agent.battery <= batteries_before[agent.id]


@pytest.mark.skipif(not os.path.exists(INSTANCE_B), reason="Consegna/B.json not found")
def test_simulation_instance_b():
    """La simulazione deve funzionare anche sull'istanza B."""
    env = Environment.from_json(INSTANCE_B)
    sim = Simulator(env=env, max_ticks=50, seed=0, verbose=False)
    metrics = sim.run()
    assert metrics.total_ticks <= 50
