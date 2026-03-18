"""
Test rapido delle nuove strategie.
"""
import sys
sys.path.insert(0, ".")

from src.environment.environment import Environment
from src.simulation.simulator import Simulator

# Carica ambiente
env = Environment.from_json("Consegna/A.json")

# Crea simulatore con agenti di default (usa le 5 nuove strategie)
sim = Simulator(env, agents=None, max_ticks=50, verbose=True)

print("\n=== Agenti creati ===")
for agent in sim.agents:
    print(f"Agent {agent.id}: {agent.strategy.name} (vis={agent.visibility_radius})")

print("\n=== Running simulation ===")
metrics = sim.run()

print("\n=== Risultati ===")
summary = metrics.summary()
print(f"Oggetti consegnati: {summary['objects_delivered']}/{env.total_objects}")
print(f"Tick totali: {summary['total_ticks']}")
print(f"Energia media consumata: {summary['avg_energy_consumed']:.1f}")
print(f"Distanza media percorsa: {summary['avg_distance_traveled']:.1f}")

print("\n✅ Test completato con successo!")
