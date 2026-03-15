"""
Visualizzatore live basato su Matplotlib.

Layout
------
  ┌─────────────────────────────┬──────────────────┐
  │                             │  Radar agenti    │
  │      Griglia 25x25          │  Batteria bar    │
  │  (griglia + agenti +        │  Oggetti score   │
  │   oggetti + comunicazione)  │  Legenda         │
  └─────────────────────────────┴──────────────────┘

Controlli (finestra aperta):
  - Chiudi la finestra per fermare la simulazione
  - Velocità regolabile via flag `tick_delay`
"""

from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from src.visualization.base import BaseVisualizer
from src.environment.grid import CellType
from src.agents.sensors import can_communicate

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment


# ---------------------------------------------------------------------------
# Palette colori
# ---------------------------------------------------------------------------

# Celle
CELL_COLORS = {
    CellType.EMPTY:     np.array([1.00, 1.00, 1.00]),   # bianco
    CellType.WALL:      np.array([0.22, 0.22, 0.22]),   # grigio scuro
    CellType.WAREHOUSE: np.array([0.29, 0.56, 0.85]),   # blu
    CellType.ENTRANCE:  np.array([0.18, 0.80, 0.44]),   # verde
    CellType.EXIT:      np.array([0.91, 0.30, 0.24]),   # rosso
}

# Uno per agente (fino a 10 agenti)
AGENT_PALETTE = [
    "#FF6B35",  # arancione
    "#4ECDC4",  # teal
    "#FFD166",  # giallo
    "#A8E6CF",  # verde menta
    "#C879FF",  # viola
    "#FF8B94",  # rosa
    "#06D6A0",  # verde acqua
    "#118AB2",  # blu scuro
    "#EF476F",  # rosso rosa
    "#073B4C",  # blu navy
]


# ---------------------------------------------------------------------------
# Visualizzatore matplotlib
# ---------------------------------------------------------------------------

class MatplotlibVisualizer(BaseVisualizer):
    """
    Visualizzatore live con matplotlib.

    Parameters
    ----------
    tick_delay   : float  — secondi di pausa tra un tick e l'altro (default 0.05)
    show_vision  : bool   — mostra il raggio di visione di ogni agente
    show_comm    : bool   — mostra i link di comunicazione attivi tra agenti
    show_fog     : bool   — oscura le celle non ancora esplorate da nessun agente
    update_every : int    — aggiorna il grafico ogni N tick (1 = ogni tick)
    """

    def __init__(
        self,
        tick_delay: float = 0.05,
        show_vision: bool = True,
        show_comm: bool = True,
        show_fog: bool = True,
        update_every: int = 1,
    ) -> None:
        self.tick_delay = tick_delay
        self.show_vision = show_vision
        self.show_comm = show_comm
        self.show_fog = show_fog
        self.update_every = update_every

        # Riferimenti agli artisti matplotlib (inizializzati in setup)
        self._fig: Optional[plt.Figure] = None
        self._ax_grid: Optional[plt.Axes] = None
        self._ax_stats: Optional[plt.Axes] = None
        self._base_img: Optional[np.ndarray] = None

        # Artisti aggiornabili
        self._img_handle = None
        self._fog_handle = None
        self._agent_circles: list = []
        self._agent_labels: list = []
        self._agent_carry_dots: list = []
        self._obj_scatter = None
        self._comm_lines: list = []
        self._comm_rects: list = []
        self._vision_circles: list = []
        self._battery_bars: list = []
        self._stats_texts: list = []

        self._env: Optional["Environment"] = None
        self._agents: Optional[List["Agent"]] = None
        self._size: int = 0
        self._tick: int = 0

    # ------------------------------------------------------------------
    # Setup iniziale
    # ------------------------------------------------------------------

    def setup(self, env: "Environment", agents: List["Agent"]) -> None:
        self._env = env
        self._agents = agents
        self._size = env.grid.size

        matplotlib.use("TkAgg") if _has_tkinter() else None

        self._fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
        self._fig.canvas.manager.set_window_title("Swarm Intelligence — Simulazione Live")

        # Layout: griglia (sinistra 70%) + stats (destra 30%)
        gs = self._fig.add_gridspec(1, 2, width_ratios=[7, 3], wspace=0.05)
        self._ax_grid = self._fig.add_subplot(gs[0])
        self._ax_stats = self._fig.add_subplot(gs[1])

        self._setup_grid_axes()
        self._setup_stats_axes(agents)

        plt.ion()
        plt.tight_layout(pad=1.5)
        plt.show(block=False)
        plt.pause(0.1)

    def _setup_grid_axes(self) -> None:
        ax = self._ax_grid
        size = self._size
        env = self._env

        # Immagine base della griglia (statica)
        self._base_img = np.zeros((size, size, 3), dtype=float)
        for r in range(size):
            for c in range(size):
                ct = CellType(env.grid.data[r][c])
                self._base_img[r, c] = CELL_COLORS[ct]

        # Immagine con fog of war
        self._fog_layer = np.zeros((size, size, 4), dtype=float)  # RGBA
        if self.show_fog:
            self._fog_layer[:, :, :3] = 0.05   # quasi nero
            self._fog_layer[:, :, 3] = 0.75    # semi-opaco

        self._img_handle = ax.imshow(
            self._base_img, interpolation="nearest", origin="upper",
            extent=[-0.5, size - 0.5, size - 0.5, -0.5],
        )
        self._fog_handle = ax.imshow(
            self._fog_layer, interpolation="nearest", origin="upper",
            extent=[-0.5, size - 0.5, size - 0.5, -0.5],
            zorder=2,
        )

        # Griglia leggera
        for i in range(size + 1):
            ax.axhline(i - 0.5, color="#555", lw=0.2, zorder=1)
            ax.axvline(i - 0.5, color="#555", lw=0.2, zorder=1)

        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.set_title("Ambiente", color="white", fontsize=11)
        ax.spines[:].set_color("#555")

        # Oggetti (scatter aggiornabile)
        self._obj_scatter = ax.scatter(
            [], [], c="#FFD700", marker="*", s=150, zorder=6, label="Oggetti",
            edgecolors="#FFA500", linewidths=0.8,
        )

        # Cerchi agenti + label
        for i, agent in enumerate(self._agents):
            color = AGENT_PALETTE[i % len(AGENT_PALETTE)]
            circle = plt.Circle(
                (agent.col, agent.row), 0.38,
                color=color, zorder=8, label=f"A{i}",
            )
            ax.add_patch(circle)
            lbl = ax.text(
                agent.col, agent.row, str(i),
                ha="center", va="center", fontsize=7,
                fontweight="bold", color="black", zorder=9,
            )
            carry_dot = plt.Circle(
                (agent.col, agent.row), 0.14,
                color="white", zorder=10, visible=False,
            )
            ax.add_patch(carry_dot)
            self._agent_circles.append(circle)
            self._agent_labels.append(lbl)
            self._agent_carry_dots.append(carry_dot)

        # Cerchi visione (inizialmente nascosti)
        if self.show_vision:
            for i, agent in enumerate(self._agents):
                color = AGENT_PALETTE[i % len(AGENT_PALETTE)]
                vc = plt.Circle(
                    (agent.col, agent.row), agent.visibility_radius + 0.5,
                    fill=False, linestyle="--", color=color, alpha=0.3,
                    zorder=7, linewidth=0.8,
                )
                ax.add_patch(vc)
                self._vision_circles.append(vc)

    def _setup_stats_axes(self, agents: List["Agent"]) -> None:
        ax = self._ax_stats
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Statistiche", color="white", fontsize=11)

        # Placeholder testo stats
        self._tick_text = ax.text(
            0.05, 0.97, "Tick: 0", color="white", fontsize=10,
            fontweight="bold", va="top", transform=ax.transAxes,
        )
        self._score_text = ax.text(
            0.05, 0.91, "Consegnati: 0/0", color="#FFD700",
            fontsize=10, va="top", transform=ax.transAxes,
        )
        self._remaining_text = ax.text(
            0.05, 0.85, "Rimanenti: 0", color="#FF8B94",
            fontsize=9, va="top", transform=ax.transAxes,
        )

        # Batteria per agente
        bar_top = 0.72
        for i, agent in enumerate(agents):
            color = AGENT_PALETTE[i % len(AGENT_PALETTE)]
            y = bar_top - i * 0.10
            # Sfondo barra
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.05, y - 0.015), 0.9, 0.03,
                boxstyle="round,pad=0.005",
                facecolor="#333", edgecolor="#555", linewidth=0.5,
                transform=ax.transAxes,
            ))
            # Barra batteria
            bar = mpatches.FancyBboxPatch(
                (0.05, y - 0.015), 0.9, 0.03,
                boxstyle="round,pad=0.005",
                facecolor=color, edgecolor="none",
                transform=ax.transAxes,
            )
            ax.add_patch(bar)
            self._battery_bars.append(bar)
            # Label agente
            ax.text(
                0.02, y, f"A{i}", color=color, fontsize=7,
                va="center", ha="right", transform=ax.transAxes,
                fontweight="bold",
            )

        # Legenda strategie
        leg_y = bar_top - len(agents) * 0.10 - 0.04
        ax.text(
            0.05, leg_y, "Strategie:", color="#AAA", fontsize=8,
            va="top", transform=ax.transAxes,
        )
        for i, agent in enumerate(agents):
            color = AGENT_PALETTE[i % len(AGENT_PALETTE)]
            ax.text(
                0.08, leg_y - 0.06 - i * 0.055,
                f"A{i}: {agent.strategy.name}",
                color=color, fontsize=7, va="top",
                transform=ax.transAxes,
            )

    # ------------------------------------------------------------------
    # Update frame
    # ------------------------------------------------------------------

    def update(self, tick: int, agents: List["Agent"], env: "Environment") -> bool:
        self._tick = tick

        # Verifica che la finestra sia ancora aperta
        if not plt.fignum_exists(self._fig.number):
            return False

        if tick % self.update_every != 0:
            return True

        self._update_fog(agents, env)
        self._update_objects(env)
        self._update_agents(agents)
        self._update_comm_lines(agents)
        self._update_stats(tick, agents, env)

        self._ax_grid.set_title(
            f"Tick {tick:4d}  |  Consegnati {env.delivered}/{env.total_objects}  "
            f"|  Rimanenti {env.remaining_objects}",
            color="white", fontsize=10,
        )

        self._fig.canvas.draw_idle()
        plt.pause(self.tick_delay)
        return True

    def _update_fog(self, agents: List["Agent"], env: "Environment") -> None:
        if not self.show_fog:
            return
        # Accumula tutte le celle viste da almeno un agente
        all_seen = set()
        for agent in agents:
            all_seen.update(agent.local_map.keys())
        # Riduci l'opacità delle celle viste
        fog = self._fog_layer.copy()
        for (r, c) in all_seen:
            fog[r, c, 3] = 0.0   # completamente trasparente
        self._fog_handle.set_data(fog)

    def _update_objects(self, env: "Environment") -> None:
        obj_positions = list(env._objects)
        if obj_positions:
            self._obj_scatter.set_offsets([[c, r] for r, c in obj_positions])
            self._obj_scatter.set_sizes([150] * len(obj_positions))
        else:
            self._obj_scatter.set_offsets(np.empty((0, 2)))

    def _update_agents(self, agents: List["Agent"]) -> None:
        for i, agent in enumerate(agents):
            circle = self._agent_circles[i]
            lbl = self._agent_labels[i]
            carry_dot = self._agent_carry_dots[i]

            if agent.is_active:
                circle.set_center((agent.col, agent.row))
                circle.set_visible(True)
                lbl.set_position((agent.col, agent.row))
                lbl.set_visible(True)
                carry_dot.set_center((agent.col, agent.row))
                carry_dot.set_visible(agent.carrying_object)
            else:
                # Agente esaurito: mostra in grigio
                circle.set_facecolor("#555")
                circle.set_center((agent.col, agent.row))

            if self.show_vision and i < len(self._vision_circles):
                vc = self._vision_circles[i]
                vc.set_center((agent.col, agent.row))
                vc.set_visible(agent.is_active)

    def _update_comm_lines(self, agents: List["Agent"]) -> None:
        # Rimuovi gli highlight precedenti
        for ln in self._comm_lines:
            ln.remove()
        self._comm_lines.clear()
        for rect in self._comm_rects:
            rect.remove()
        self._comm_rects.clear()

        if not self.show_comm:
            return

        # Disegna linee tra agenti che comunicano questo tick
        n = len(agents)
        for i in range(n):
            a = agents[i]
            if not a.is_active:
                continue
            for j in range(i + 1, n):
                b = agents[j]
                if not b.is_active:
                    continue
                if can_communicate(a.pos, b.pos, min(a.comm_radius, b.comm_radius)):
                    rect = mpatches.Rectangle(
                        (min(a.col, b.col) - 0.5, min(a.row, b.row) - 0.5),
                        abs(a.col - b.col) + 1,
                        abs(a.row - b.row) + 1,
                        facecolor="#4DD0E1",
                        edgecolor="#00BCD4",
                        linewidth=1.2,
                        alpha=0.18,
                        zorder=6.5,
                    )
                    self._ax_grid.add_patch(rect)
                    self._comm_rects.append(rect)
                    ln = self._ax_grid.add_line(mlines.Line2D(
                        [a.col, b.col], [a.row, b.row],
                        color="cyan", alpha=0.4, linewidth=0.8,
                        linestyle=":", zorder=7,
                    ))
                    self._comm_lines.append(ln)

    def _update_stats(
        self, tick: int, agents: List["Agent"], env: "Environment"
    ) -> None:
        self._tick_text.set_text(f"Tick: {tick}")
        self._score_text.set_text(
            f"Consegnati: {env.delivered}/{env.total_objects}"
        )
        self._remaining_text.set_text(f"Rimanenti: {env.remaining_objects}")

        max_batt = agents[0].INITIAL_BATTERY if agents else 500
        for i, agent in enumerate(agents):
            if i >= len(self._battery_bars):
                break
            bar = self._battery_bars[i]
            ratio = max(0.0, agent.battery / max_batt)
            bar.set_width(0.9 * ratio)
            # Cambia colore della barra in base al livello batteria
            if ratio > 0.5:
                bar.set_facecolor(AGENT_PALETTE[i % len(AGENT_PALETTE)])
            elif ratio > 0.2:
                bar.set_facecolor("#FFA500")
            else:
                bar.set_facecolor("#FF4444")

    # ------------------------------------------------------------------
    # Chiusura
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._fig and plt.fignum_exists(self._fig.number):
            plt.ioff()
            self._ax_grid.set_title(
                f"Simulazione terminata — Tick {self._tick}",
                color="white", fontsize=11,
            )
            self._fig.canvas.draw()
            plt.show(block=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_tkinter() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except ImportError:
        return False
