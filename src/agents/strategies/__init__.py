from src.agents.strategies.base import ExplorationStrategy
from src.agents.strategies.random_walk import RandomWalkStrategy
from src.agents.strategies.frontier import FrontierStrategy
from src.agents.strategies.LévyFlight import LevyFlightStrategy, SpiralStrategy
from src.agents.strategies.Repulsion import RepulsionStrategy
from src.agents.strategies.greedy import GreedyStrategy

# Backward-compatible alias: SectorStrategy maps to RepulsionStrategy.
SectorStrategy = RepulsionStrategy

__all__ = [
    "ExplorationStrategy",
    "RandomWalkStrategy",
    "FrontierStrategy",
    "LevyFlightStrategy",
    "SpiralStrategy",        # alias → LevyFlightStrategy
    "RepulsionStrategy",
    "SectorStrategy",        # alias → RepulsionStrategy
    "GreedyStrategy",
]
