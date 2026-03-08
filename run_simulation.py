"""
Entry point della simulazione.

Uso:
    # Senza visualizzazione
    python run_simulation.py --instance Consegna/A.json

    # Visualizzazione matplotlib (live, interattiva)
    python run_simulation.py --instance Consegna/A.json --visualize

    # Visualizzazione pygame (interattiva, con controlli tastiera)
    python run_simulation.py --instance Consegna/B.json --visualize-pygame

    # Opzioni avanzate
    python run_simulation.py --instance Consegna/A.json \\
        --visualize --tick-delay 0.1 --no-fog --seed 42 --max-ticks 750

    # Salva risultati
    python run_simulation.py --instance Consegna/A.json --save-results results/run_A.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulazione Multi-Agente Swarm Intelligence — Recupero Oggetti",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Istanza ---
    parser.add_argument(
        "--instance", type=str, default="Consegna/A.json",
        help="Percorso al file JSON dell'istanza (default: Consegna/A.json)",
    )
    parser.add_argument(
        "--max-ticks", type=int, default=500,
        help="Numero massimo di tick (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed per la riproducibilità (default: nessuno)",
    )

    # --- Visualizzazione ---
    viz_group = parser.add_argument_group("visualizzazione")
    viz_exclusive = viz_group.add_mutually_exclusive_group()
    viz_exclusive.add_argument(
        "--visualize", action="store_true",
        help="Visualizzazione live con Matplotlib",
    )
    viz_exclusive.add_argument(
        "--visualize-pygame", action="store_true",
        help="Visualizzazione interattiva con Pygame (richiede: pip install pygame)",
    )

    viz_group.add_argument(
        "--tick-delay", type=float, default=0.05, metavar="SEC",
        help="Secondi di pausa tra un tick e l'altro durante la visualizzazione (default: 0.05)",
    )
    viz_group.add_argument(
        "--update-every", type=int, default=1, metavar="N",
        help="Aggiorna il grafico ogni N tick (default: 1). Usare >1 per simulazioni veloci.",
    )
    viz_group.add_argument(
        "--no-fog", action="store_true",
        help="Disabilita il fog of war (mostra tutta la griglia da subito)",
    )
    viz_group.add_argument(
        "--no-vision", action="store_true",
        help="Nasconde i cerchi del raggio di visione degli agenti",
    )
    viz_group.add_argument(
        "--no-comm", action="store_true",
        help="Nasconde le linee di comunicazione tra agenti",
    )
    viz_group.add_argument(
        "--debug", action="store_true",
        help="Mostra tutti gli oggetti (anche non ancora scoperti dagli agenti)",
    )

    # --- Output ---
    parser.add_argument(
        "--save-results", type=str, default=None, metavar="PATH",
        help="Salva i risultati in formato JSON nel percorso specificato",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Disabilita l'output testuale durante la simulazione",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.instance):
        print(f"Errore: file istanza non trovato: {args.instance}", file=sys.stderr)
        sys.exit(1)

    from src.environment.environment import Environment
    from src.simulation.simulator import Simulator

    print(f"Caricamento istanza: {args.instance}")
    env = Environment.from_json(args.instance)
    print(
        f"  Griglia {env.grid.size}x{env.grid.size} | "
        f"Magazzini: {len(env.warehouses)} | "
        f"Oggetti: {env.total_objects}"
    )

    # --- Costruisce il visualizzatore richiesto ---
    visualizer = None

    if args.visualize:
        from src.visualization.matplotlib_viz import MatplotlibVisualizer
        visualizer = MatplotlibVisualizer(
            tick_delay=args.tick_delay,
            show_vision=not args.no_vision,
            show_comm=not args.no_comm,
            show_fog=not args.no_fog,
            update_every=args.update_every,
        )
        print(
            f"  Visualizzatore: Matplotlib "
            f"| delay={args.tick_delay}s "
            f"| fog={'OFF' if args.no_fog else 'ON'}"
        )

    elif args.visualize_pygame:
        from src.visualization.pygame_viz import PygameVisualizer
        visualizer = PygameVisualizer(
            tick_delay=args.tick_delay,
            show_fog=not args.no_fog,
            show_vision=not args.no_vision,
            show_comm=not args.no_comm,
            show_debug=args.debug,
        )
        print(
            f"  Visualizzatore: Pygame "
            f"| delay={args.tick_delay}s "
            f"| fog={'OFF' if args.no_fog else 'ON'} "
            f"| debug={'ON' if args.debug else 'OFF'}"
        )

    # --- Avvia simulatore ---
    sim = Simulator(
        env=env,
        agents=None,
        max_ticks=args.max_ticks,
        seed=args.seed,
        verbose=not args.quiet,
        visualizer=visualizer,
    )
    metrics = sim.run()

    # --- Salva risultati ---
    if args.save_results:
        _save_results(metrics, args)


def _save_results(metrics, args: argparse.Namespace) -> None:
    summary = metrics.summary()
    summary["instance"] = args.instance
    summary["max_ticks"] = args.max_ticks
    summary["seed"] = args.seed

    out_dir = os.path.dirname(args.save_results) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.save_results, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Risultati salvati in: {args.save_results}")


if __name__ == "__main__":
    main()
