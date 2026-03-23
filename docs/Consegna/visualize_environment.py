"""
Visualizza un ambiente da un file JSON e salva come immagine PNG.

Uso:
    python visualize_environment.py input.json output.png
    python visualize_environment.py input.json              # salva come input.png
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


# Costanti cella
EMPTY = 0
WALL = 1
WAREHOUSE = 2
ENTRANCE = 3
EXIT = 4


def visualize(input_json: str, output_png: str) -> None:
    """Legge il JSON dell'ambiente e salva la visualizzazione come PNG."""

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = data["grid"]
    warehouses = data["warehouses"]
    objects = data.get("objects", [])
    n = data["metadata"]["grid_size"]

    fig, ax = plt.subplots(figsize=(10, 10))

    color_map = {
        EMPTY: "white",
        WALL: "#404040",
        WAREHOUSE: "#4a90d9",
        ENTRANCE: "#2ecc71",
        EXIT: "#e74c3c",
    }

    # Mappa frecce direzionali per entrate/uscite di ogni magazzino
    arrow_map = {}
    for w in warehouses:
        side = w["side"]
        er, ec = w["entrance"]
        xr, xc = w["exit"]
        if side == "top":
            arrow_map[(er, ec)] = "\u25B2"
            arrow_map[(xr, xc)] = "\u25BC"
        elif side == "bottom":
            arrow_map[(er, ec)] = "\u25BC"
            arrow_map[(xr, xc)] = "\u25B2"
        elif side == "left":
            arrow_map[(er, ec)] = "\u25C0"
            arrow_map[(xr, xc)] = "\u25B6"
        else:
            arrow_map[(er, ec)] = "\u25B6"
            arrow_map[(xr, xc)] = "\u25C0"

    # Disegna cella per cella
    for r in range(n):
        for c in range(n):
            val = grid[r][c]
            color = color_map.get(val, "white")
            rect = Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=color,
                edgecolor="lightgrey",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            if (r, c) in arrow_map:
                ax.text(
                    c, r, arrow_map[(r, c)], fontsize=8, color="white",
                    ha="center", va="center", fontweight="bold",
                )

    # Oggetti - cerchi arancioni
    for obj in objects:
        obj_r, obj_c = obj
        circle = plt.Circle(
            (obj_c, obj_r), 0.3,
            color="orange", zorder=5,
        )
        ax.add_patch(circle)

    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor="#404040", edgecolor="black",
                       label="Muro / Scaffale"),
        mpatches.Patch(facecolor="#4a90d9", edgecolor="black",
                       label="Magazzino"),
        mpatches.Patch(facecolor="#2ecc71", edgecolor="black",
                       label="Entrata"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black",
                       label="Uscita"),
        mpatches.Patch(facecolor="white", edgecolor="black",
                       label="Corridoio"),
        mpatches.Patch(facecolor="orange", edgecolor="black",
                       label="Oggetto"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=9, framealpha=0.9,
    )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"MAPD Logistics Grid {n}\u00d7{n}", fontsize=14)
    ax.set_xlabel("Colonna")
    ax.set_ylabel("Riga")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Immagine salvata in '{output_png}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualizza un ambiente MAPD da un file JSON e salva come PNG."
    )
    parser.add_argument(
        "input_json",
        help="percorso del file JSON con lo stato dell'ambiente",
    )
    parser.add_argument(
        "output_png",
        nargs="?",
        default=None,
        help="percorso del file PNG di output (default: <input>.png)",
    )

    args = parser.parse_args()
    output = args.output_png or os.path.splitext(args.input_json)[0] + ".png"

    visualize(args.input_json, output)
