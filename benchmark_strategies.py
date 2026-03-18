"""
Benchmark comparativo delle strategie di esplorazione.

Esegue N simulazioni per ogni strategia su ogni istanza, con raggio visivo
fisso e stesso seed per eliminare la varianza casuale. Stampa una tabella
di confronto finale con le metriche chiave.

Uso:
    python benchmark_strategies.py
    python benchmark_strategies.py --instances Consegna/A.json Consegna/B.json
    python benchmark_strategies.py --runs 10 --seed 0 --visibility 2
    python benchmark_strategies.py --save-results results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from src.environment.environment import Environment
from src.agents.agent import Agent
from src.agents.strategies.frontier import FrontierStrategy
from src.agents.strategies.nearest_unvisited import NearestUnvisitedStrategy
from src.agents.strategies.warehouse_centric import WarehouseCentricStrategy
from src.agents.strategies.voronoi_zoning import VoronoiZoningStrategy
from src.agents.strategies.wall_follower import WallFollowerStrategy
from src.simulation.simulator import Simulator


# ---------------------------------------------------------------------------
# Configurazione strategie
# ---------------------------------------------------------------------------

STRATEGIES = {
    "Frontier":          lambda n: [FrontierStrategy()          for _ in range(n)],
    "NearestUnvisited":  lambda n: [NearestUnvisitedStrategy()  for _ in range(n)],
    "WarehouseCentric":  lambda n: [WarehouseCentricStrategy()  for _ in range(n)],
    "VoronoiZoning":     lambda n: [VoronoiZoningStrategy()     for _ in range(n)],
    "WallFollower":      lambda n: [WallFollowerStrategy()      for _ in range(n)],
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_once(
    instance_path: str,
    strategy_name: str,
    strategy_factories,
    num_agents: int,
    visibility_radius: int,
    seed: int,
    max_ticks: int,
) -> Dict:
    env = Environment.from_json(instance_path)
    strategies = strategy_factories(num_agents)
    agents = [
        Agent(
            agent_id=i,
            strategy=strategies[i],
            grid=env.grid,
            visibility_radius=visibility_radius,
        )
        for i in range(num_agents)
    ]
    metrics = Simulator(env, agents, max_ticks=max_ticks, seed=seed, verbose=False).run()
    return metrics.summary()


def benchmark(
    instances: List[str],
    num_runs: int,
    num_agents: int,
    visibility_radius: int,
    base_seed: int,
    max_ticks: int,
) -> Dict:
    results = {}

    for instance in instances:
        instance_label = instance.split("/")[-1].replace(".json", "")
        results[instance_label] = {}

        print(f"\n{'='*60}")
        print(f"  Istanza: {instance}")
        print(f"{'='*60}")
        print(f"  {'Strategia':<14} {'Consegnati':>10} {'Tick medi':>10} {'Energia':>10} {'Comp.%':>8}")
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        for strategy_name, factory in STRATEGIES.items():
            run_results = []
            for run_i in range(num_runs):
                seed = base_seed + run_i
                r = run_once(
                    instance, strategy_name, factory,
                    num_agents, visibility_radius, seed, max_ticks,
                )
                run_results.append(r)

            # Medie tra le run
            avg_delivered  = sum(r["objects_delivered"] for r in run_results) / num_runs
            total_objects  = run_results[0]["total_objects"]
            avg_ticks      = sum(r["total_ticks"] for r in run_results) / num_runs
            avg_energy     = sum(r["average_energy_consumed"] for r in run_results) / num_runs
            completion_pct = (avg_delivered / total_objects * 100) if total_objects else 0

            results[instance_label][strategy_name] = {
                "avg_delivered":   round(avg_delivered, 2),
                "total_objects":   total_objects,
                "avg_ticks":       round(avg_ticks, 1),
                "avg_energy":      round(avg_energy, 1),
                "completion_pct":  round(completion_pct, 1),
                "runs":            num_runs,
            }

            print(
                f"  {strategy_name:<14} "
                f"{avg_delivered:>8.1f}/{total_objects:<1} "
                f"{avg_ticks:>10.1f} "
                f"{avg_energy:>10.1f} "
                f"{completion_pct:>7.1f}%"
            )

        # Ranking per tick medi (solo strategie che completano al 100%)
        strats = results[instance_label]
        full_completion = {k: v for k, v in strats.items() if v["completion_pct"] == 100.0}
        partial         = {k: v for k, v in strats.items() if v["completion_pct"] <  100.0}

        ranked_full    = sorted(full_completion.items(), key=lambda x: x[1]["avg_ticks"])
        ranked_partial = sorted(partial.items(), key=lambda x: x[1]["completion_pct"], reverse=True)
        ranked = ranked_full + ranked_partial

        print(f"\n  Ranking (1=migliore):")
        for pos, (name, _) in enumerate(ranked, 1):
            tag = " ✓ completamento 100%" if name in full_completion else ""
            print(f"    {pos}. {name}{tag}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark comparativo delle strategie di esplorazione",
    )
    parser.add_argument(
        "--instances", nargs="+",
        default=["Consegna/A.json", "Consegna/B.json"],
        help="Percorsi ai file JSON delle istanze",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Numero di run per strategia per istanza (default: 5)",
    )
    parser.add_argument(
        "--agents", type=int, default=5,
        help="Numero di agenti (default: 5)",
    )
    parser.add_argument(
        "--visibility", type=int, default=2,
        help="Raggio di visibilità fisso per tutti gli agenti (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed base (ogni run usa seed+i) (default: 42)",
    )
    parser.add_argument(
        "--max-ticks", type=int, default=500,
        help="Tick massimi per simulazione (default: 500)",
    )
    parser.add_argument(
        "--save-results", type=str, default=None, metavar="PATH",
        help="Salva i risultati in JSON nel percorso specificato",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nBenchmark strategie — {args.runs} run/strategia — "
          f"{args.agents} agenti — visibilità {args.visibility} — seed base {args.seed}")

    results = benchmark(
        instances=args.instances,
        num_runs=args.runs,
        num_agents=args.agents,
        visibility_radius=args.visibility,
        base_seed=args.seed,
        max_ticks=args.max_ticks,
    )

    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results) or ".", exist_ok=True)
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nRisultati salvati in: {args.save_results}")


if __name__ == "__main__":
    import os
    main()
