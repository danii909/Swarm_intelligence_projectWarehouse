"""
Interfaccia Streamlit per configurare ed eseguire la simulazione Swarm Intelligence.

    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import math
import sys
import os
import time
import random as _random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend non-interattivo per Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configurazione pagina
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Swarm Intelligence Project",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Costanti strategia
# ---------------------------------------------------------------------------

STRATEGIES = {
    0: ("Frontier",    "Esplorazione frontier-based sistematica"),
    1: ("Greedy",      "Esplorazione warehouse-centric"),
    2: ("Sector",      "Suddivide la griglia in settori assegnati"),
    3: ("Repulsion",   "Dispersione emergente dagli altri agenti"),
    4: ("Smart Random", "Random walk guidato da info gain, stale e separazione"),
    5: ("Ant-Colony",  "Coverage con feromone evaporativo condiviso"),
}

DEFAULT_RADIUS = {0: 2, 1: 3, 2: 2, 3: 2, 4: 1, 5: 2}

STRATEGY_COLORS = {
    "Frontier":    "#4C72B0",
    "Greedy":      "#DD8452",
    "Sector":      "#55A868",
    "Repulsion":   "#C44E52",
    "Smart Random": "#8172B2",
    "Ant-Colony":  "#1F9D8A",
}

# ---------------------------------------------------------------------------
# Palette celle (uguale a matplotlib_viz.py)
# ---------------------------------------------------------------------------

from src.environment.grid import CellType

_CELL_RGB = {
    CellType.EMPTY:     (1.00, 1.00, 1.00),
    CellType.WALL:      (0.22, 0.22, 0.22),
    CellType.WAREHOUSE: (0.29, 0.56, 0.85),
    CellType.ENTRANCE:  (0.18, 0.80, 0.44),
    CellType.EXIT:      (0.91, 0.30, 0.24),
}

_AGENT_PALETTE = [
    "#FF1F1F", "#0057FF", "#00B050", "#000000", "#FFD400",
    "#FF8B94", "#06D6A0", "#118AB2", "#EF476F", "#7A1CAC",
]

_AGENT_RGB_PALETTE = [
    (255, 31, 31), (0, 87, 255), (0, 176, 80), (0, 0, 0), (255, 212, 0),
    (255, 139, 148), (6, 214, 160), (17, 138, 178), (239, 71, 111), (122, 28, 172),
]

INITIAL_BATTERY = 500  # uguale a Agent.INITIAL_BATTERY
AGENT_ICON_DEFAULT_PATH = "assets/agent.png"
PACKAGE_ICON_DEFAULT_PATH = "assets/package.png"
DEFAULT_FRAME_DELAY = 0.15

_PYGAME = None
_PYGAME_FONT = None
_STREAMLIT_CELL_PX = 28
_STREAMLIT_BG = (15, 15, 40)
_STREAMLIT_GRID = (100, 100, 100)
_STREAMLIT_COMM_FILL = (30, 144, 255, 50)
_STREAMLIT_COMM_BORDER = (0, 0, 139, 200)
_STREAMLIT_COMM_LINE = (0, 0, 139, 200)


# ---------------------------------------------------------------------------
# Render frame griglia → PNG bytes
# ---------------------------------------------------------------------------

def _get_pygame():
    """Inizializza pygame in modo lazy per il rendering offscreen."""
    global _PYGAME, _PYGAME_FONT
    if _PYGAME is None:
        import pygame
        pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        _PYGAME = pygame
        _PYGAME_FONT = pygame.font.SysFont("monospace", 12, bold=True)
    return _PYGAME


def _normalize_pygame_surface(surface):
    """Converte con alpha se possibile, senza fallire in assenza di display attivo."""
    try:
        return surface.convert_alpha()
    except Exception:
        return surface


def _agent_label_rgb(color_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Sceglie un colore testo leggibile sopra il colore agente."""
    luminance = (0.299 * color_rgb[0]) + (0.587 * color_rgb[1]) + (0.114 * color_rgb[2])
    return (255, 255, 255) if luminance < 140 else (0, 0, 0)


def _agent_label_hex(color_hex: str) -> str:
    """Versione hex del colore di contrasto per etichette HTML."""
    color_hex = color_hex.lstrip("#")
    rgb = tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
    label_rgb = _agent_label_rgb(rgb)
    return "#FFFFFF" if label_rgb == (255, 255, 255) else "#000000"


def _load_icon_image(path_str: str):
    """Carica un'immagine icona da path locale. Ritorna None se non valida."""
    path_str = (path_str or "").strip()
    if not path_str:
        return None

    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.is_file():
        return None

    try:
        return plt.imread(str(path))
    except Exception:
        return None


def _load_pygame_icon(path_str: str):
    """Carica un'icona pygame con alpha, senza aprire una finestra."""
    path_str = (path_str or "").strip()
    if not path_str:
        return None

    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.is_file():
        return None

    try:
        pygame = _get_pygame()
        surface = pygame.image.load(str(path))
        return _normalize_pygame_surface(surface)
    except Exception:
        return None


def _load_uploaded_pygame_icon(uploaded_file):
    """Carica un'immagine caricata via Streamlit come Surface pygame."""
    if uploaded_file is None:
        return None

    try:
        pygame = _get_pygame()
        surface = pygame.image.load(io.BytesIO(uploaded_file.getvalue()), uploaded_file.name)
        return _normalize_pygame_surface(surface)
    except Exception:
        return None


def _draw_star_pygame(screen, cx: int, cy: int, r: int, color) -> None:
    """Disegna una stella a 5 punte sulla superficie pygame."""
    import math as _math

    pygame = _get_pygame()
    points = []
    r_inner = r * 0.45
    for i in range(10):
        angle = _math.radians(-90 + i * 36)
        radius = r if i % 2 == 0 else r_inner
        points.append((cx + radius * _math.cos(angle), cy + radius * _math.sin(angle)))

    pygame.draw.polygon(screen, color, points)
    pygame.draw.polygon(screen, (200, 160, 0), points, 1)


def _draw_package_pygame(screen, cx: int, cy: int, size: int) -> None:
    """Disegna un pacco stilizzato (fallback quando manca l'icona custom)."""
    pygame = _get_pygame()

    box = max(8, size)
    front_w = box
    front_h = max(6, int(box * 0.72))
    top_h = max(3, int(box * 0.28))

    x = cx - front_w // 2
    y = cy - (front_h + top_h) // 2

    front = pygame.Rect(x, y + top_h, front_w, front_h)
    top = [
        (x, y + top_h),
        (x + max(2, front_w // 5), y),
        (x + front_w + max(2, front_w // 5), y),
        (x + front_w, y + top_h),
    ]

    front_color = (233, 179, 82)
    edge_color = (140, 94, 32)
    top_color = (245, 201, 120)
    tape_color = (190, 132, 53)

    pygame.draw.polygon(screen, top_color, top)
    pygame.draw.polygon(screen, edge_color, top, 1)
    pygame.draw.rect(screen, front_color, front, border_radius=2)
    pygame.draw.rect(screen, edge_color, front, 1, border_radius=2)

    tape_x = front.centerx
    pygame.draw.line(screen, tape_color, (tape_x, y + 1), (tape_x, front.bottom - 1), 2)
    pygame.draw.line(screen, tape_color, (front.left + 2, front.top + 2), (front.right - 2, front.top + 2), 2)


def _draw_aa_circle_pygame(screen, cx: int, cy: int, radius: int, color, border_color=None, border_width: int = 0) -> None:
    """Disegna un cerchio anti-aliased quando possibile, con fallback standard."""
    pygame = _get_pygame()
    try:
        import pygame.gfxdraw as gfxdraw

        gfxdraw.filled_circle(screen, cx, cy, radius, color)
        gfxdraw.aacircle(screen, cx, cy, radius, color)
        if border_color is not None and border_width > 0:
            for rr in range(radius, max(0, radius - border_width), -1):
                gfxdraw.aacircle(screen, cx, cy, rr, border_color)
    except Exception:
        pygame.draw.circle(screen, color, (cx, cy), radius)
        if border_color is not None and border_width > 0:
            pygame.draw.circle(screen, border_color, (cx, cy), radius, border_width)


def _render_empty_grid_pygame(env) -> np.ndarray:
    """Render di anteprima a griglia vuota (senza agenti/oggetti) per stato idle."""
    pygame = _get_pygame()
    size = env.grid.size
    px = _STREAMLIT_CELL_PX

    screen = pygame.Surface((size * px, size * px), pygame.SRCALPHA)
    screen.fill(_STREAMLIT_BG)

    for r in range(size):
        for c in range(size):
            ct = CellType(env.grid.data[r][c])
            color = tuple(int(channel * 255) for channel in _CELL_RGB[ct])
            rect = pygame.Rect(c * px, r * px, px, px)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, _STREAMLIT_GRID, rect, 1)

    return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))


def _render_frame_pygame(
    tick: int,
    agents,
    env,
    show_fog: bool = True,
    agent_icon_img=None,
    package_icon_img=None,
) -> np.ndarray:
    """Renderizza lo stato corrente usando pygame offscreen e ritorna un array RGB."""
    from src.agents.sensors import can_communicate

    pygame = _get_pygame()
    size = env.grid.size
    px = _STREAMLIT_CELL_PX
    half = px // 2

    screen = pygame.Surface((size * px, size * px), pygame.SRCALPHA)
    screen.fill(_STREAMLIT_BG)

    for r in range(size):
        for c in range(size):
            ct = CellType(env.grid.data[r][c])
            color = tuple(int(channel * 255) for channel in _CELL_RGB[ct])
            rect = pygame.Rect(c * px, r * px, px, px)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, _STREAMLIT_GRID, rect, 1)

    all_seen: set = set()
    for agent in agents:
        all_seen.update(agent.local_map.keys())

    if show_fog:
        fog = pygame.Surface((size * px, size * px), pygame.SRCALPHA)
        fog.fill((15, 15, 40, 200))
        for (r, c) in all_seen:
            pygame.draw.rect(fog, (0, 0, 0, 0), pygame.Rect(c * px, r * px, px, px))
        screen.blit(fog, (0, 0))

    dir_to_triangle = {
        (-1, 0): ((0.5, 0.15), (0.2, 0.8), (0.8, 0.8)),
        (1, 0): ((0.2, 0.2), (0.8, 0.2), (0.5, 0.85)),
        (0, -1): ((0.15, 0.5), (0.8, 0.2), (0.8, 0.8)),
        (0, 1): ((0.2, 0.2), (0.85, 0.5), (0.2, 0.8)),
    }
    opposite = {(-1, 0): (1, 0), (1, 0): (-1, 0), (0, -1): (0, 1), (0, 1): (0, -1)}
    for r in range(size):
        for c in range(size):
            ct = CellType(env.grid.data[r][c])
            if ct not in (CellType.ENTRANCE, CellType.EXIT):
                continue
            direction = (-1, 0)
            for dr, dc in dir_to_triangle:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if CellType(env.grid.data[nr][nc]) == CellType.WAREHOUSE:
                        direction = (dr, dc) if ct == CellType.ENTRANCE else opposite[(dr, dc)]
                        break

            color = (255, 255, 255) if (r, c) in all_seen else (68, 68, 68)
            points = [
                (c * px + int(px * x), r * px + int(px * y))
                for x, y in dir_to_triangle[direction]
            ]
            pygame.draw.polygon(screen, color, points)

    for r, c in env._objects:
        cx = c * px + half
        cy = r * px + half
        if package_icon_img is not None:
            icon = pygame.transform.smoothscale(package_icon_img, (int(px * 0.72), int(px * 0.72)))
            screen.blit(icon, (cx - icon.get_width() // 2, cy - icon.get_height() // 2))
        else:
            _draw_package_pygame(screen, cx, cy, int(px * 0.62))

    for i, agent in enumerate(agents):
        if not agent.is_active:
            continue
        color = _AGENT_RGB_PALETTE[i % len(_AGENT_RGB_PALETTE)]
        cx = agent.col * px + half
        cy = agent.row * px + half
        radius = int((agent.visibility_radius + 0.5) * px)
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, 30), (radius, radius), radius)
        pygame.draw.circle(surf, (*color, 80), (radius, radius), radius, 1)
        screen.blit(surf, (cx - radius, cy - radius))

    comm_surf = pygame.Surface((size * px, size * px), pygame.SRCALPHA)
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
                x1 = a.col * px + half
                y1 = a.row * px + half
                x2 = b.col * px + half
                y2 = b.row * px + half
                rect = pygame.Rect(
                    min(a.col, b.col) * px,
                    min(a.row, b.row) * px,
                    (abs(a.col - b.col) + 1) * px,
                    (abs(a.row - b.row) + 1) * px,
                )
                pygame.draw.rect(comm_surf, _STREAMLIT_COMM_FILL, rect)
                pygame.draw.rect(comm_surf, _STREAMLIT_COMM_BORDER, rect, 2)
                pygame.draw.line(comm_surf, _STREAMLIT_COMM_LINE, (x1, y1), (x2, y2), 1)
    screen.blit(comm_surf, (0, 0))

    agent_icon_scaled = None
    if agent_icon_img is not None:
        icon_size = int(px * 0.8)
        agent_icon_scaled = pygame.transform.smoothscale(agent_icon_img, (icon_size, icon_size))

    for i, agent in enumerate(agents):
        color = _AGENT_RGB_PALETTE[i % len(_AGENT_RGB_PALETTE)] if agent.is_active else (85, 85, 85)
        label_color = _agent_label_rgb(color)
        cx = agent.col * px + half
        cy = agent.row * px + half
        radius = max(4, half - 4)

        if agent_icon_scaled is not None:
            pygame.draw.circle(screen, (0, 0, 0), (cx, cy), radius + 2, 2)
            icon = agent_icon_scaled.copy()
            if not agent.is_active:
                icon = icon.copy()
                icon.fill((255, 255, 255, 110), special_flags=pygame.BLEND_RGBA_MULT)
            screen.blit(icon, (cx - icon.get_width() // 2, cy - icon.get_height() // 2))
        else:
            base_color = color if agent.is_active else (95, 95, 95)
            edge_color = (10, 10, 10) if agent.is_active else (55, 55, 55)
            _draw_aa_circle_pygame(screen, cx, cy, radius, base_color, edge_color, border_width=2)
            if agent.is_active:
                hi = tuple(min(255, int(ch * 1.2) + 18) for ch in base_color)
                _draw_aa_circle_pygame(
                    screen,
                    cx - max(1, radius // 3),
                    cy - max(1, radius // 3),
                    max(2, radius // 3),
                    hi,
                )
            if not agent.is_active:
                off = radius // 2
                pygame.draw.line(screen, (60, 60, 60), (cx - off, cy - off), (cx + off, cy + off), 2)
                pygame.draw.line(screen, (60, 60, 60), (cx + off, cy - off), (cx - off, cy + off), 2)

        if agent.carrying_object:
            pulse = 2 + int(abs(np.sin(tick * 0.35)) * 2)
            pygame.draw.circle(screen, (255, 215, 0), (cx, cy), radius + pulse, 3)
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), max(2, radius // 3))

            badge_size = max(8, int(px * 0.34))
            badge_rect = pygame.Rect(0, 0, badge_size, badge_size)
            badge_rect.center = (cx + radius - badge_size // 4, cy - radius + badge_size // 4)
            pygame.draw.rect(screen, (255, 196, 0), badge_rect, border_radius=2)
            pygame.draw.rect(screen, (110, 78, 0), badge_rect, 1, border_radius=2)
            pygame.draw.line(
                screen,
                (130, 92, 0),
                (badge_rect.centerx, badge_rect.top + 1),
                (badge_rect.centerx, badge_rect.bottom - 2),
                1,
            )
            pygame.draw.line(
                screen,
                (130, 92, 0),
                (badge_rect.left + 1, badge_rect.centery),
                (badge_rect.right - 2, badge_rect.centery),
                1,
            )

        if agent_icon_scaled is None:
            label = _PYGAME_FONT.render(str(i + 1), True, label_color)
            screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

    return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))


def _render_frame(
    tick: int,
    agents,
    env,
    show_fog: bool = True,
    agent_icon_img=None,
    package_icon_img=None,
) -> np.ndarray:
    """Renderizza lo stato corrente della simulazione per Streamlit."""
    return _render_frame_pygame(
        tick,
        agents,
        env,
        show_fog=show_fog,
        agent_icon_img=agent_icon_img,
        package_icon_img=package_icon_img,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_agents(agent_configs: list, num_agents: int):
    """Costruisce la lista di Agent dalla configurazione UI."""
    from src.agents.agent import Agent
    from src.agents.strategies.frontier import FrontierStrategy
    from src.agents.strategies.greedy import GreedyStrategy
    from src.agents.strategies.sector import SectorStrategy
    from src.agents.strategies.Repulsion import RepulsionStrategy
    from src.agents.strategies.random_walk import RandomWalkStrategy
    from src.agents.strategies.ant_colony_lite import AntColonyLiteStrategy

    factories = {
        0: lambda: FrontierStrategy(),
        1: lambda: GreedyStrategy(),
        2: lambda: SectorStrategy(num_agents=num_agents),
        3: lambda: RepulsionStrategy(),
        4: lambda: RandomWalkStrategy(),
        5: lambda: AntColonyLiteStrategy(),
    }

    agents = []
    for cfg in agent_configs:
        strategy = factories[cfg["strategy_id"]]()
        agents.append(Agent(
            agent_id=cfg["agent_id"],
            strategy=strategy,
            visibility_radius=cfg["radius"],
            comm_radius=cfg.get("comm_radius", 2),
        ))
    return agents


def _render_battery_html(agents, agent_configs) -> str:
    """Genera HTML per le barre di batteria degli agenti."""
    strat_by_id = {cfg["agent_id"]: STRATEGIES[cfg["strategy_id"]][0] for cfg in agent_configs}

    html_parts = []

    for i, agent in enumerate(agents):
        pct = max(agent.battery / INITIAL_BATTERY, 0.0)
        pct_display = pct * 100
        if pct > 0.5:
            bar_color = "#55A868"
        elif pct > 0.2:
            bar_color = "#DD8452"
        else:
            bar_color = "#C44E52"

        agent_color = _AGENT_PALETTE[i % len(_AGENT_PALETTE)]
        agent_text_color = _agent_label_hex(agent_color)
        state_label = agent.state.name.replace("_", " ").title()
        strat_name = strat_by_id.get(agent.id, "?")
        radii_label = f"(v{agent.visibility_radius}, c{agent.comm_radius})"

        html_parts.append(
            f"<div style='margin-bottom:8px; padding:8px; border:1px solid #2e3342; border-radius:8px; background:#161b28;'>"
            f"  <div style='display:flex; justify-content:space-between; "
            f"       align-items:center; margin-bottom:6px;'>"
            f"    <span style='display:inline-flex; align-items:center; gap:6px;'>"
            f"      <span style='background:{agent_color}; color:{agent_text_color}; border:1px solid #000; border-radius:999px; padding:2px 8px; font-size:0.78em; font-weight:bold;'>A{agent.id + 1}</span>"
            f"      <span style='color:#cfd3d8; font-size:0.78em;'>{strat_name}</span>"
            f"      <span style='color:#8ea2b3; font-size:0.72em;'>{radii_label}</span>"
            f"    </span>"
            f"    <span style='color:#999; font-size:0.7em;'>"
            f"      🔋{agent.battery}/{INITIAL_BATTERY} — {state_label}</span>"
            f"  </div>"
            f"  <div style='background:#333; border-radius:4px; height:10px; overflow:hidden; margin-top:2px;'>"
            f"    <div style='background:{bar_color}; width:{pct_display:.1f}%; "
            f"         height:100%; border-radius:4px; transition: width 0.3s ease;'></div>"
            f"  </div>"
            f"</div>"
        )

    html_parts.append("</div>")
    return "".join(html_parts)


def _render_status_card_html(title: str, value: str, accent: str) -> str:
    """Genera una card HTML per metriche compatte (Tick/Consegnati)."""
    return (
        "<div style='"
        "padding:10px; border:1px solid #2e3342; border-radius:8px; "
        "background:#161b28; min-height:50px; display:flex; flex-direction:column; "
        "justify-content:center; margin-bottom:12px;'>"
        f"<div style='font-size:0.78em; color:#8ea2b3; margin-bottom:4px;'>{title}</div>"
        f"<div style='font-size:1.35em; font-weight:700; color:{accent}; line-height:1.1;'>{value}</div>"
        "</div>"
    )


def _style_dark_chart(ax):
    """Applica stile dark coerente a un asse matplotlib."""
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#aaa", labelsize=7)


def _safe_norm(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.isna().all():
        out = pd.Series([0.0] * len(s), index=s.index)
    else:
        s_min = s.min()
        s_max = s.max()
        if math.isclose(s_max, s_min):
            out = pd.Series([0.0] * len(s), index=s.index)
        else:
            out = (s - s_min) / (s_max - s_min)
    return 1.0 - out if invert else out


def _compute_composite_score(
    df: pd.DataFrame,
    w_delivery: float = 0.55,
    w_ticks: float = 0.25,
    w_energy: float = 0.20,
) -> pd.Series:
    delivery_term = _safe_norm(df["delivery_rate"], invert=False)
    ticks_term = _safe_norm(df["total_ticks"], invert=True)
    energy_term = _safe_norm(df["average_energy"], invert=True)
    score = (w_delivery * delivery_term) + (w_ticks * ticks_term) + (w_energy * energy_term)
    return score.round(4)


def _grouped_stats(df: pd.DataFrame, group_col: str, metric_cols: list[str]) -> pd.DataFrame:
    """Statistiche raggruppate sui preset unici che contengono il fattore."""
    if df.empty:
        return pd.DataFrame()

    agg_map = {"preset_name": pd.Series.nunique}
    for col in metric_cols:
        agg_map[col] = ["mean", "median", "std"]

    grouped = df.groupby(group_col).agg(agg_map)
    grouped.columns = [
        "N preset" if a == "preset_name" else f"{a} {b}"
        for a, b in grouped.columns
    ]
    grouped = grouped.reset_index()

    for col in grouped.columns:
        if pd.api.types.is_numeric_dtype(grouped[col]):
            grouped[col] = grouped[col].round(3)

    return grouped.sort_values("N preset", ascending=False)


def _plot_dark_boxplot(ax, data_groups, labels, colors=None, ylabel=""):
    _style_dark_chart(ax)

    bp = ax.boxplot(
        data_groups,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="#FFD700", linewidth=2),
        whiskerprops=dict(color="#bbbbbb"),
        capprops=dict(color="#bbbbbb"),
        flierprops=dict(markerfacecolor="#C44E52", marker="o", markersize=4),
    )

    if colors is None:
        colors = ["#4ECDC4"] * len(data_groups)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)

    ax.set_xticklabels(labels, color="#cccccc", fontsize=8)
    ax.set_ylabel(ylabel, color="#cccccc", fontsize=9)


def _plot_time_curves(top_rows: pd.DataFrame, title: str, y_key: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#0e1117")
    _style_dark_chart(ax)

    plotted = False
    for _, row in top_rows.iterrows():
        series = row.get(y_key)
        if isinstance(series, (list, tuple, np.ndarray)) and len(series) > 0:
            ax.plot(series, linewidth=2, label=row["preset_name"])
            plotted = True

    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel("Tick", color="#cccccc", fontsize=9)
    ax.set_ylabel(ylabel, color="#cccccc", fontsize=9)

    if plotted:
        ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
        plt.tight_layout()
        return fig

    plt.close(fig)
    return None


def _build_strategy_synergy_matrix(all_results, strategies_dict, metric: str = "score") -> pd.DataFrame:
    strategy_names = [v[0] for _, v in sorted(strategies_dict.items())]
    matrix = pd.DataFrame(index=strategy_names, columns=strategy_names, dtype=float)

    for s1 in strategy_names:
        for s2 in strategy_names:
            vals = []
            for r in all_results:
                team_strats = [strategies_dict[sid][0] for sid, _, _ in r["preset_raw"]]
                if s1 in team_strats and s2 in team_strats and metric in r and r[metric] is not None:
                    vals.append(r[metric])
            matrix.loc[s1, s2] = np.mean(vals) if vals else np.nan

    return matrix


# ---------------------------------------------------------------------------
# Opzioni strategia (definite qui per essere disponibili ovunque)
# ---------------------------------------------------------------------------

strategy_options = [f"{sid} — {name}  ({desc})" for sid, (name, desc) in STRATEGIES.items()]
strategy_name_options = [name for sid, (name, _) in STRATEGIES.items()]
strategy_ids = list(STRATEGIES.keys())

# ---------------------------------------------------------------------------
# Sidebar — parametri globali
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configurazione globale")
    st.divider()

    st.subheader("📁 Istanza")
    instances_found = sorted(
        str(p) for p in Path(".").glob("**/*.json")
        if "Consegna" in str(p) or "instances" in str(p).lower()
    )
    if not instances_found:
        instances_found = ["Consegna/A.json", "Consegna/B.json"]

    instance_path = st.selectbox(
        "File istanza",
        options=instances_found,
        index=0,
    )

    st.subheader("⚙️ Seed")
    seed = st.number_input("(−1 = casuale)", min_value=-1, max_value=9999, value=42)

    st.divider()

# ---------------------------------------------------------------------------
# Pannello principale — Tab layout
# ---------------------------------------------------------------------------
_logo_b64 = base64.b64encode(Path("assets/2.png").read_bytes()).decode()
st.markdown(
    f"""<div style="display:flex; align-items:center; gap:12px; margin-top:-1rem; margin-bottom:1.5rem;">
        <img src="data:image/png;base64,{_logo_b64}" width="72" style="border-radius:8px;">
        <div>
            <h1 style="margin:0; font-size:3.5rem; padding:0; line-height:1.1;">E.L.B.E.R.R.</h1>
            <p style="margin:0; color:#9aa0a6; font-size:0.95rem;">
                Efficient Logistics by Exploration with Robotic Retrieval
            </p>
        </div>
    </div>""",
    unsafe_allow_html=True,
)
tab_sim, tab_bench = st.tabs(["🎮 Simulazione", "🔬 Benchmark"])

# ===================================================================
# TAB 1: SIMULAZIONE SINGOLA (+ battery monitoring + comm_radius)
# ===================================================================

with tab_sim:
    st.markdown(
        """
        <style>
        div[data-testid="stSlider"] {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        .element-container:has(div[data-testid="stSlider"]) {
            margin-top: 0rem;
            margin-bottom: -0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_cfg, col_sim, col_status = st.columns([2, 4, 2])

    with col_cfg:
        if "_apply_preset" in st.session_state:
            _p = st.session_state.pop("_apply_preset")
            st.session_state["num_agents_val"] = _p["num_agents"]
            for _a in _p["agents"]:
                _aid = _a["agent_id"]
                st.session_state[f"strat_{_aid}"] = strategy_name_options[_a["strategy_id"]]
                st.session_state[f"radius_{_aid}"] = _a["radius"]
                st.session_state[f"comm_{_aid}"] = _a.get("comm_radius", 2)

        run_clicked = st.button("▶ Avvia", type="primary", width='stretch')
        st.markdown("##### Configurazione")

        with st.expander(f"Parametri generali", expanded=False):
            num_agents = st.slider("N. agenti", min_value=1, max_value=10, value=5, key="num_agents_val")
            max_ticks = st.slider("Tick massimi", min_value=100, max_value=750, value=500, step=50)
            update_every = st.slider("Aggiorna ogni N tick", min_value=1, max_value=20, value=1)
            frame_delay = st.slider(
                "Delay tra frame (secondi)",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_FRAME_DELAY,
                step=0.01,
                help="Piu alto = simulazione visivamente piu lenta.",
            )
            agent_icon_upload = st.file_uploader(
                "Immagine agente personalizzata",
                type=["png", "jpg", "jpeg", "webp"],
                key="agent_icon_upload",
                help="Se caricata, sostituisce l'icona agente di default nella simulazione.",
            )
            package_icon_upload = st.file_uploader(
                "Immagine pacco personalizzata",
                type=["png", "jpg", "jpeg", "webp"],
                key="package_icon_upload",
                help="Se caricata, sostituisce l'icona pacco di default nella simulazione.",
            )

        uploaded_preset = st.file_uploader("📂 Carica preset config. agenti", type=["json"], key="upload_preset_col")
        if uploaded_preset is not None:
            try:
                p = json.loads(uploaded_preset.read().decode("utf-8"))
                st.session_state["_apply_preset"] = p
            except Exception as _e:
                st.error(f"Errore lettura preset: {_e}")

        agent_configs = []

        for agent_id in range(num_agents):
            default_sid = agent_id % len(STRATEGIES)
            default_r = DEFAULT_RADIUS[default_sid]

            with st.expander(f"Agente {agent_id + 1}", expanded=False):
                chosen_name = st.selectbox(
                    "Strategia",
                    options=strategy_name_options,
                    index=default_sid,
                    key=f"strat_{agent_id}",
                )
                chosen_sid = strategy_ids[strategy_name_options.index(chosen_name)]
                radius = st.slider(
                    " Raggio Visione", min_value=1, max_value=3, value=default_r,
                    key=f"radius_{agent_id}",
                )
                comm_r = st.slider(
                    " Raggio Comunicazione", min_value=1, max_value=2, value=2,
                    key=f"comm_{agent_id}",
                )

            agent_configs.append({
                "agent_id": agent_id,
                "strategy_id": chosen_sid,
                "radius": radius,
                "comm_radius": comm_r,
            })

    with col_sim:
        frame_ph = st.empty()

    with col_status:
        top_left, top_right = st.columns(2)
        with top_left:
            tick_ph = st.empty()
        with top_right:
            stats_ph = st.empty()
        prog_ph = st.empty()
        battery_ph = st.empty()

    if not run_clicked:
        preview_agents = []
        total_objects_preview = "?"

        if os.path.isfile(instance_path):
            try:
                from src.environment.environment import Environment

                env_preview = Environment.from_json(instance_path)
                preview_agents = _build_agents(agent_configs, len(agent_configs))
                total_objects_preview = str(env_preview.total_objects)

                preview_agent_icon = _load_uploaded_pygame_icon(agent_icon_upload)
                preview_package_icon = _load_uploaded_pygame_icon(package_icon_upload)
                if preview_package_icon is None:
                    preview_package_icon = _load_pygame_icon(PACKAGE_ICON_DEFAULT_PATH)

                preview_png = _render_frame(
                    0,
                    preview_agents,
                    env_preview,
                    show_fog=False,
                    agent_icon_img=preview_agent_icon,
                    package_icon_img=preview_package_icon,
                )
                frame_ph.image(preview_png, width='stretch')
            except Exception as preview_exc:
                frame_ph.warning(f"Anteprima non disponibile: {preview_exc}")
        else:
            frame_ph.info("Seleziona un file istanza valido per vedere l'anteprima.")

        tick_ph.markdown(
            _render_status_card_html("Tick", "0", "#4C72B0"),
            unsafe_allow_html=True,
        )
        stats_ph.markdown(
            _render_status_card_html("Consegnati", f"0 / {total_objects_preview}", "#55A868"),
            unsafe_allow_html=True,
        )

        prog_ph.progress(0.0, text=f"Tick 0/{max_ticks}")

        if preview_agents:
            battery_ph.markdown(
                _render_battery_html(preview_agents, agent_configs),
                unsafe_allow_html=True,
            )
        else:
            battery_ph.info("Configura gli agenti e premi Avvia.")

    if run_clicked:
        if not os.path.isfile(instance_path):
            st.error(f"File istanza non trovato: `{instance_path}`")
            st.stop()

        from src.environment.environment import Environment
        from src.simulation.simulator import Simulator

        env_obj = Environment.from_json(instance_path)
        built_agents = _build_agents(agent_configs, len(agent_configs))
        _seed = seed if seed >= 0 else None

        sim = Simulator(
            env=env_obj,
            agents=built_agents,
            max_ticks=max_ticks,
            seed=_seed,
            verbose=False,
            log_every=1,
        )

        agent_icon_img = _load_uploaded_pygame_icon(agent_icon_upload)

        package_icon_img = _load_uploaded_pygame_icon(package_icon_upload)
        if package_icon_img is None:
            package_icon_img = _load_pygame_icon(PACKAGE_ICON_DEFAULT_PATH)

        t0 = time.perf_counter()

        try:
            for tick, cur_agents, cur_env in sim.step_gen():
                if tick % update_every == 0 or cur_env.all_delivered:
                    png = _render_frame(
                        tick,
                        cur_agents,
                        cur_env,
                        show_fog=True,
                        agent_icon_img=agent_icon_img,
                        package_icon_img=package_icon_img,
                    )
                    frame_ph.image(png, width='stretch')
                    prog_ph.progress(
                        min(tick / max_ticks, 1.0),
                        text=f"Tick {tick}/{max_ticks}",
                    )
                    tick_ph.markdown(
                        _render_status_card_html("Tick", str(tick), "#4C72B0"),
                        unsafe_allow_html=True,
                    )
                    stats_ph.markdown(
                        _render_status_card_html(
                            "Consegnati",
                            f"{cur_env.delivered} / {cur_env.total_objects}",
                            "#55A868",
                        ),
                        unsafe_allow_html=True,
                    )
                    battery_ph.markdown(
                        _render_battery_html(cur_agents, agent_configs),
                        unsafe_allow_html=True,
                    )
                    if frame_delay > 0:
                        time.sleep(frame_delay)
        except Exception as exc:
            st.error(f"Errore durante la simulazione: {exc}")
            st.exception(exc)
            st.stop()

        elapsed = time.perf_counter() - t0
        metrics = sim.metrics
        summary = metrics.summary()

        if "history_runs" not in st.session_state:
            st.session_state["history_runs"] = []
        st.session_state["history_runs"].append({"summary": summary, "configs": list(agent_configs)})

        st.divider()
        st.subheader("📊 Risultati")

        r1, r2, r3, r4, r5 = st.columns(5)
        delivered = summary["objects_delivered"]
        total = summary["total_objects"]
        rate_pct = summary["delivery_rate"] * 100
        ticks_done = summary["total_ticks"]
        energy = summary["average_energy_consumed"]

        r1.metric("Oggetti consegnati", f"{delivered} / {total}")
        r2.metric("Completamento", f"{rate_pct:.1f}%")
        r3.metric("Tick totali", ticks_done)
        r4.metric("Energia media", f"{energy:.1f}")
        r5.metric("Tempo CPU", f"{elapsed:.2f}s")

        st.subheader("🤖 Dettaglio agenti")
        steps_list = summary.get("agent_steps", [])
        batteries_list = summary.get("agent_final_batteries", [])

        agent_rows = []
        for cfg in agent_configs:
            i = cfg["agent_id"]
            agent_rows.append({
                "Agente": f"A{i + 1}",
                "Strategia": STRATEGIES[cfg["strategy_id"]][0],
                "Raggio vis.": cfg["radius"],
                "Raggio Comunicazione": cfg["comm_radius"],
                "Passi": steps_list[i] if i < len(steps_list) else "—",
                "Batteria finale": batteries_list[i] if i < len(batteries_list) else "—",
            })

        df_agents = pd.DataFrame(agent_rows)

        def _color_strategy(val):
            color = STRATEGY_COLORS.get(val, "#888")
            return f"background-color: {color}22; color: {color}; font-weight: bold;"

        styled = df_agents.style.map(_color_strategy, subset=["Strategia"])
        st.dataframe(styled, width='stretch', hide_index=True)

        if metrics.history:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.subheader("📈 Consegne nel tempo")
                ticks_list_h = [s.tick for s in metrics.history]
                delivered_list = [s.delivered for s in metrics.history]
                remaining_list = [s.remaining for s in metrics.history]
                df_hist = pd.DataFrame({
                    "Tick": ticks_list_h,
                    "Consegnati": delivered_list,
                    "Rimanenti": remaining_list,
                }).set_index("Tick")
                st.line_chart(df_hist, color=["#55A868", "#C44E52"])

            with gc2:
                st.subheader("🔋 Batteria nel tempo")
                batt_data = {"Tick": [s.tick for s in metrics.history]}
                for i, cfg in enumerate(agent_configs):
                    batt_data[f"A{cfg['agent_id'] + 1}"] = [
                        s.agent_batteries[i] if i < len(s.agent_batteries) else 0
                        for s in metrics.history
                    ]
                df_batt = pd.DataFrame(batt_data).set_index("Tick")
                colors_b = [_AGENT_PALETTE[i % len(_AGENT_PALETTE)] for i in range(len(agent_configs))]
                st.line_chart(df_batt, color=colors_b)

        preset_data = {
            "name": "preset",
            "num_agents": len(agent_configs),
            "agents": agent_configs,
        }
        st.download_button(
            "⬇ Scarica preset corrente",
            data=json.dumps(preset_data, indent=4),
            file_name="preset.json",
            mime="application/json",
        )

    if "history_runs" in st.session_state and len(st.session_state["history_runs"]) > 1:
        st.divider()
        st.subheader("🕑 Storico simulazioni")

        hist_rows = []
        for idx, run in enumerate(st.session_state["history_runs"]):
            s = run["summary"]
            cfg_str = ", ".join(
                f"A{c['agent_id'] + 1}:{STRATEGIES[c['strategy_id']][0]}"
                f"(v{c['radius']} c{c.get('comm_radius', 2)})"
                for c in run["configs"]
            )
            hist_rows.append({
                "Run": idx + 1,
                "Consegnati": f"{s['objects_delivered']}/{s['total_objects']}",
                "Completamento": f"{s['delivery_rate']*100:.1f}%",
                "Tick": s["total_ticks"],
                "Energia media": s["average_energy_consumed"],
                "Configurazione": cfg_str,
            })

        df_hist_runs = pd.DataFrame(hist_rows)
        st.dataframe(df_hist_runs, width='stretch', hide_index=True)

        if st.button("🗑 Azzera storico"):
            st.session_state["history_runs"] = []
            st.rerun()


# ===================================================================
# TAB 2: BENCHMARK ESPLORATIVO
# ===================================================================

with tab_bench:
    left_col, right_col = st.columns([1.6, 1], gap="large")

    with left_col:
        st.markdown("### Configurazione")

        with st.container(border=True):
            st.markdown("#### Parametri generali")
            g1, g2 = st.columns(2)

            with g1:
                bench_num_agents = st.slider(
                    "N. agenti",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="bench_num_agents"
                )

            with g2:
                bench_max_ticks = st.slider(
                    "Tick massimi",
                    min_value=100,
                    max_value=750,
                    value=500,
                    step=50,
                    key="bench_max_ticks"
                )

        st.markdown("")

        with st.container(border=True):
            st.markdown("#### Strategie")

            strat_mode = st.segmented_control(
                "Modalità strategia",
                options=["Casuale", "Fissa"],
                default="Casuale",
                key="bench_strat_mode"
            )

            if strat_mode == "Casuale":
                bench_strategies = st.multiselect(
                    "Strategie possibili",
                    options=[f"{sid} — {name}" for sid, (name, _) in STRATEGIES.items()],
                    default=[f"{sid} — {name}" for sid, (name, _) in STRATEGIES.items()],
                    key="bench_strats",
                    placeholder="Seleziona almeno una strategia"
                )
                bench_strategy_ids = [int(s.split(" — ")[0]) for s in bench_strategies]
            else:
                fixed_strat_label = st.selectbox(
                    "Strategia fissa",
                    options=[f"{sid} — {name}" for sid, (name, _) in STRATEGIES.items()],
                    index=1,
                    key="bench_fixed_strat",
                )
                bench_strategy_ids = [int(fixed_strat_label.split(" — ")[0])]

        st.markdown("")

        with st.container(border=True):
            st.markdown("#### Raggi")

            r1, r2 = st.columns(2, gap="large")

            with r1:
                st.markdown("**Visione**")
                vis_mode = st.segmented_control(
                    "Modalità visione",
                    options=["Casuale", "Fissa"],
                    default="Casuale",
                    key="bench_vis_mode"
                )

                if vis_mode == "Casuale":
                    bench_vis_range = st.slider(
                        "Range visione",
                        min_value=1,
                        max_value=5,
                        value=(1, 3),
                        key="bench_vis_range"
                    )
                    vis_values = list(range(bench_vis_range[0], bench_vis_range[1] + 1))
                else:
                    fixed_vis = st.slider(
                        "Visione fissa",
                        min_value=1,
                        max_value=5,
                        value=2,
                        key="bench_fixed_vis"
                    )
                    vis_values = [fixed_vis]

            with r2:
                st.markdown("**Comunicazione**")
                comm_mode = st.segmented_control(
                    "Modalità comunicazione",
                    options=["Casuale", "Fissa"],
                    default="Casuale",
                    key="bench_comm_mode"
                )

                if comm_mode == "Casuale":
                    bench_comm_range = st.slider(
                        "Range comunicazione",
                        min_value=1,
                        max_value=2,
                        value=(1, 2),
                        key="bench_comm_range"
                    )
                    comm_values = list(range(bench_comm_range[0], bench_comm_range[1] + 1))
                else:
                    fixed_comm = st.slider(
                        "Comunicazione fissa",
                        min_value=1,
                        max_value=2,
                        value=2,
                        key="bench_fixed_comm"
                    )
                    comm_values = [fixed_comm]

    choices_per_agent = len(bench_strategy_ids) * len(vis_values) * len(comm_values)
    max_unique_presets = choices_per_agent ** bench_num_agents if choices_per_agent > 0 else 0

    with right_col:
        st.markdown("### Esecuzione")

        with st.container(border=True):
            st.markdown("#### Riepilogo spazio di ricerca")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Scelte per agente", choices_per_agent)
            with m2:
                st.metric("Preset unici max", max_unique_presets)

            st.markdown("")

            default_bench_n = min(20, max_unique_presets) if max_unique_presets > 0 else 1

            bench_n = st.number_input(
                "N. preset da testare",
                min_value=1,
                max_value=max_unique_presets if max_unique_presets > 0 else 1,
                value=default_bench_n,
                step=1,
                key="bench_n",
                help="Numero di configurazioni casuali da eseguire senza duplicati, fin dove possibile."
            )

            st.markdown("")

            bench_clicked = st.button(
                "▶ Avvia benchmark",
                type="primary",
                use_container_width=True,
                key="bench_run"
            )

            if bench_strategy_ids:
                selected_names = [STRATEGIES[sid][0] for sid in bench_strategy_ids]
                st.caption(
                    f"Strategie selezionate: {', '.join(selected_names)} · "
                    f"Visione: {vis_values} · Comunicazione: {comm_values}"
                )
            else:
                st.caption("Nessuna strategia selezionata.")

    if bench_clicked:
        if not bench_strategy_ids:
            st.error("Seleziona almeno una strategia.")
            st.stop()

        if not os.path.isfile(instance_path):
            st.error(f"File istanza non trovato: `{instance_path}`")
            st.stop()

        from src.environment.environment import Environment
        from src.simulation.simulator import Simulator

        actual_n = min(bench_n, max_unique_presets)
        rng = _random.Random(seed if seed >= 0 else None)

        per_agent_space = list(itertools.product(bench_strategy_ids, vis_values, comm_values))
        generated_presets = []
        seen_signatures = set()

        if actual_n >= max_unique_presets:
            for combo in itertools.product(per_agent_space, repeat=bench_num_agents):
                generated_presets.append(list(combo))
        else:
            attempts = 0
            max_attempts = actual_n * 50
            while len(generated_presets) < actual_n and attempts < max_attempts:
                preset = tuple(rng.choice(per_agent_space) for _ in range(bench_num_agents))
                if preset not in seen_signatures:
                    seen_signatures.add(preset)
                    generated_presets.append(list(preset))
                attempts += 1

        actual_n = len(generated_presets)

        st.markdown("---")
        st.markdown("### Avanzamento benchmark")

        bench_progress = st.progress(0.0, text="Avvio benchmark... 0.0%")
        bench_status = st.empty()

        all_results = []
        t0_bench = time.perf_counter()

        for sim_i, preset in enumerate(generated_presets):
            preset_name = f"Preset {sim_i + 1}"

            agent_cfgs = []
            for ai, (strat_id, vis_r, comm_r) in enumerate(preset):
                agent_cfgs.append({
                    "agent_id": ai,
                    "strategy_id": strat_id,
                    "radius": vis_r,
                    "comm_radius": comm_r,
                })

            env_obj = Environment.from_json(instance_path)
            built_agents = _build_agents(agent_cfgs, bench_num_agents)

            sim = Simulator(
                env=env_obj,
                agents=built_agents,
                max_ticks=bench_max_ticks,
                seed=42,
                verbose=False,
                log_every=5,
            )

            t0_sim = time.perf_counter()
            for _ in sim.step_gen():
                pass
            elapsed_sim = time.perf_counter() - t0_sim

            m = sim.metrics
            s = m.summary()

            config_parts = []
            for ai, (strat_id, vis_r, comm_r) in enumerate(preset):
                sn = STRATEGIES[strat_id][0][:4]
                config_parts.append(f"A{ai + 1}:{sn}(v{vis_r},c{comm_r})")
            config_str = " ".join(config_parts)

            strat_counts = {}
            for strat_id, _, _ in preset:
                sname = STRATEGIES[strat_id][0]
                strat_counts[sname] = strat_counts.get(sname, 0) + 1
            team_desc = " + ".join(f"{cnt}×{sn}" for sn, cnt in sorted(strat_counts.items()))

            avg_vis = np.mean([vis_r for _, vis_r, _ in preset])
            avg_comm = np.mean([comm_r for _, _, comm_r in preset])
            dominant_strategy = max(strat_counts, key=strat_counts.get) if strat_counts else None

            all_results.append({
                "preset_name": preset_name,
                "config_str": config_str,
                "team_desc": team_desc,
                "agent_configs": agent_cfgs,
                "preset_raw": preset,
                "avg_vis": round(avg_vis, 2),
                "avg_comm": round(avg_comm, 2),
                "objects_delivered": s["objects_delivered"],
                "total_objects": s["total_objects"],
                "delivery_rate": s["delivery_rate"],
                "total_ticks": s["total_ticks"],
                "average_energy": s["average_energy_consumed"],
                "cpu_time": round(elapsed_sim, 3),
                "dominant_strategy": dominant_strategy,
                "detected_over_time": s.get("detected_over_time"),
                "delivered_over_time": s.get("delivered_over_time"),
            })

            pct = (sim_i + 1) / actual_n
            bench_progress.progress(
                pct,
                text=f"{pct * 100:.0f}% - Preset {sim_i + 1}/{actual_n}",
            )
            bench_status.info(
                f"{preset_name} · {team_desc} · "
                f"{s['objects_delivered']}/{s['total_objects']} oggetti · "
                f"{s['total_ticks']} tick"
            )

        total_bench_time = time.perf_counter() - t0_bench
        bench_progress.progress(1.0, text="Benchmark completato (100.0%)")
        bench_status.empty()

        st.session_state["bench_results"] = {
            "all_results": all_results,
            "actual_n": actual_n,
            "total_bench_time": total_bench_time,
            "bench_strategy_ids": bench_strategy_ids,
            "vis_values": vis_values,
        }

    if "bench_results" in st.session_state:
        _br = st.session_state["bench_results"]
        all_results = _br["all_results"]
        actual_n = _br["actual_n"]
        total_bench_time = _br["total_bench_time"]

        if not all_results:
            st.warning("Nessun risultato benchmark disponibile.")
            st.stop()

        df = pd.DataFrame(all_results).copy()
        total_obj = int(df["total_objects"].iloc[0]) if "total_objects" in df.columns else 0
        df["score"] = _compute_composite_score(df)

        for i, row in df.iterrows():
            all_results[i]["score"] = row["score"]

        st.divider()
        st.subheader("📊 Risultati benchmark")

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Preset testati", actual_n)
        mc2.metric("Completion medio", f"{df['delivery_rate'].mean() * 100:.1f}%")
        mc3.metric("Tick medi", f"{df['total_ticks'].mean():.1f}")
        mc4.metric("Energia media", f"{df['average_energy'].mean():.1f}")
        mc5.metric("Tempo totale CPU", f"{total_bench_time:.2f}s")

        export_cols = [
            "preset_name", "config_str", "team_desc", "dominant_strategy", "avg_vis", "avg_comm",
            "objects_delivered", "total_objects", "delivery_rate",
            "total_ticks", "average_energy", "score", "cpu_time"
        ]
        st.download_button(
            "💾 Scarica tutti i risultati (CSV)",
            data=df[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="benchmark_results.csv",
            mime="text/csv",
            key="dl_all_csv",
            help="Salva subito i risultati su disco. Sì, persino Streamlit riesce a farlo.",
        )

        st.divider()
        st.markdown("#### 🏆 Classifica preset")
        st.caption("Ordinata per score composito, poi completion, tick, energia")

        df_rank = df.sort_values(
            by=["score", "delivery_rate", "total_ticks", "average_energy"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

        df_rank_display = pd.DataFrame({
            "Pos.": range(1, len(df_rank) + 1),
            "Preset": df_rank["preset_name"],
            "Team": df_rank["team_desc"],
            "Dominante": df_rank["dominant_strategy"],
            "Configurazione": df_rank["config_str"],
            "Consegnati": df_rank["objects_delivered"].astype(int).astype(str) + "/" + df_rank["total_objects"].astype(int).astype(str),
            "Completion": (df_rank["delivery_rate"] * 100).round(1).astype(str) + "%",
            "Tick": df_rank["total_ticks"].round(0),
            "Energia": df_rank["average_energy"].round(2),
            "Score": df_rank["score"].round(4),
        })
        st.dataframe(df_rank_display, width='stretch', hide_index=True)

        if len(df_rank) >= 1:
            st.markdown("**Top 3:**")
            medals = ["🥇", "🥈", "🥉"]
            for i in range(min(3, len(df_rank))):
                row = df_rank.iloc[i]
                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(
                        f"{medals[i]} **{row['preset_name']}** · {row['team_desc']} · "
                        f"completion **{row['delivery_rate'] * 100:.1f}%** · "
                        f"**{row['total_ticks']} tick** · energia **{row['average_energy']:.1f}** · "
                        f"score **{row['score']:.4f}**"
                    )
                    st.caption(f"Dettaglio: {row['config_str']}")
                with col_btn:
                    dl_data = {
                        "name": row["preset_name"],
                        "num_agents": len(row["agent_configs"]),
                        "agents": row["agent_configs"],
                    }
                    st.download_button(
                        label="⬇ Scarica",
                        data=json.dumps(dl_data, indent=4),
                        file_name=f"{row['preset_name'].replace(' ', '_')}.json",
                        mime="application/json",
                        key=f"dl_top_{i}",
                    )

        st.divider()
        st.markdown("#### 📋 Tutti i preset")

        df_all_display = pd.DataFrame({
            "Preset": df["preset_name"],
            "Team": df["team_desc"],
            "Dominante": df["dominant_strategy"],
            "Configurazione": df["config_str"],
            "Completion": (df["delivery_rate"] * 100).round(1).astype(str) + "%",
            "Consegnati": df["objects_delivered"],
            "Tick": df["total_ticks"],
            "Energia media": df["average_energy"].round(1),
            "Score": df["score"].round(4),
            "CPU (s)": df["cpu_time"],
        })
        st.dataframe(df_all_display, width='stretch', hide_index=True)

        agent_level_rows = []
        for r in all_results:
            for strat_id, vis_r, comm_r in r["preset_raw"]:
                agent_level_rows.append({
                    "preset_name": r["preset_name"],
                    "strategy": STRATEGIES[strat_id][0],
                    "vis_radius": vis_r,
                    "comm_radius": comm_r,
                    "delivery_rate": r["delivery_rate"],
                    "total_ticks": r["total_ticks"],
                    "average_energy": r["average_energy"],
                    "objects_delivered": r["objects_delivered"],
                    "score": r["score"],
                })
        df_agents_flat = pd.DataFrame(agent_level_rows)

        metric_cols = ["delivery_rate", "total_ticks", "average_energy", "objects_delivered", "score"]

        st.divider()
        st.markdown("#### 📊 Raggruppamento per strategia")
        st.caption("Performance media dei preset che contengono ciascuna strategia")
        df_by_strat = _grouped_stats(df_agents_flat, "strategy", metric_cols)
        st.dataframe(df_by_strat, width='stretch', hide_index=True)

        st.markdown("#### 📊 Raggruppamento per raggio visione")
        st.caption("Performance media dei preset che contengono agenti con ciascun raggio")
        df_by_vis = _grouped_stats(df_agents_flat, "vis_radius", metric_cols)
        st.dataframe(df_by_vis, width='stretch', hide_index=True)

        st.markdown("#### 📊 Raggruppamento per raggio comunicazione")
        df_by_comm = _grouped_stats(df_agents_flat, "comm_radius", metric_cols)
        st.dataframe(df_by_comm, width='stretch', hide_index=True)

        st.divider()
        st.subheader("📈 Grafici di analisi")

        dominant_colors = [STRATEGY_COLORS.get(s, "#888888") for s in df["dominant_strategy"]]

        st.markdown("#### Pareto scatter: tick vs energia")
        st.caption("Colore = strategia dominante, dimensione = completion rate")
        fig_pareto, ax_pareto = plt.subplots(figsize=(10, 5.5), facecolor="#0e1117")
        _style_dark_chart(ax_pareto)
        sizes = 60 + (df["delivery_rate"].fillna(0) * 180)
        ax_pareto.scatter(
            df["total_ticks"],
            df["average_energy"],
            s=sizes,
            c=dominant_colors,
            edgecolors="#444444",
            linewidths=0.6,
            alpha=0.9,
        )
        ax_pareto.set_xlabel("Tick totali", color="#cccccc", fontsize=9)
        ax_pareto.set_ylabel("Energia media", color="#cccccc", fontsize=9)
        legend_handles = [
            mpatches.Patch(color=STRATEGY_COLORS.get(sn, "#888888"), label=sn)
            for sn in sorted(df["dominant_strategy"].dropna().unique())
        ]
        if legend_handles:
            ax_pareto.legend(
                handles=legend_handles,
                fontsize=8,
                facecolor="#1a1a2e",
                edgecolor="#555",
                labelcolor="white"
            )
        plt.tight_layout()
        st.pyplot(fig_pareto)
        plt.close(fig_pareto)

        st.markdown("#### Completion vs tick")
        st.caption("I preset forti stanno in alto a sinistra. Incredibile, la geometria funziona ancora.")
        fig_ct, ax_ct = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
        _style_dark_chart(ax_ct)
        ax_ct.scatter(
            df["total_ticks"],
            df["delivery_rate"] * 100,
            s=80,
            c=dominant_colors,
            edgecolors="#444444",
            linewidths=0.6,
        )
        ax_ct.set_xlabel("Tick totali", color="#cccccc", fontsize=9)
        ax_ct.set_ylabel("Completion rate (%)", color="#cccccc", fontsize=9)
        ax_ct.axhline(df["delivery_rate"].mean() * 100, linestyle="--", linewidth=1.2, color="#FFD700")
        ax_ct.axvline(df["total_ticks"].mean(), linestyle="--", linewidth=1.2, color="#FFD700")
        plt.tight_layout()
        st.pyplot(fig_ct)
        plt.close(fig_ct)

        top_k = min(5, len(df_rank))
        top_rows = df_rank.head(top_k)
        if top_rows["detected_over_time"].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
            st.markdown("#### Curve cumulative: oggetti rilevati nel tempo")
            fig_detected = _plot_time_curves(
                top_rows,
                "Oggetti rilevati nel tempo - Top preset",
                "detected_over_time",
                "Oggetti rilevati",
            )
            if fig_detected is not None:
                st.pyplot(fig_detected)
                plt.close(fig_detected)

        if top_rows["delivered_over_time"].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
            st.markdown("#### Curve cumulative: oggetti consegnati nel tempo")
            fig_delivered = _plot_time_curves(
                top_rows,
                "Oggetti consegnati nel tempo - Top preset",
                "delivered_over_time",
                "Oggetti consegnati",
            )
            if fig_delivered is not None:
                st.pyplot(fig_delivered)
                plt.close(fig_delivered)

        strat_names_present = sorted(df_agents_flat["strategy"].dropna().unique().tolist())
        if len(strat_names_present) > 1:
            st.markdown("#### Distribuzione score per strategia")
            fig_bs, ax_bs = plt.subplots(figsize=(10, 4.8), facecolor="#0e1117")
            data_groups = []
            colors = []
            labels = []
            for sname in strat_names_present:
                vals = (
                    df_agents_flat[df_agents_flat["strategy"] == sname][["preset_name", "score"]]
                    .drop_duplicates()["score"]
                    .dropna()
                    .values
                )
                if len(vals) > 0:
                    data_groups.append(vals)
                    labels.append(sname)
                    colors.append(STRATEGY_COLORS.get(sname, "#888888"))
            if len(data_groups) > 1:
                _plot_dark_boxplot(ax_bs, data_groups, labels, colors=colors, ylabel="Score")
                plt.tight_layout()
                st.pyplot(fig_bs)
            plt.close(fig_bs)

        vis_unique = sorted(df_agents_flat["vis_radius"].dropna().unique())
        if len(vis_unique) > 1:
            st.markdown("#### Distribuzione score per raggio visione")
            fig_bv, ax_bv = plt.subplots(figsize=(9, 4.4), facecolor="#0e1117")
            data_groups = []
            labels = []
            for v in vis_unique:
                vals = (
                    df_agents_flat[df_agents_flat["vis_radius"] == v][["preset_name", "score"]]
                    .drop_duplicates()["score"]
                    .dropna()
                    .values
                )
                if len(vals) > 0:
                    data_groups.append(vals)
                    labels.append(str(v))
            if len(data_groups) > 1:
                _plot_dark_boxplot(ax_bv, data_groups, labels, ylabel="Score")
                ax_bv.set_xlabel("Raggio visione", color="#cccccc", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_bv)
            plt.close(fig_bv)

        comm_unique = sorted(df_agents_flat["comm_radius"].dropna().unique())
        if len(comm_unique) > 1:
            st.markdown("#### Distribuzione score per raggio comunicazione")
            fig_bc, ax_bc = plt.subplots(figsize=(8, 4.4), facecolor="#0e1117")
            data_groups = []
            labels = []
            for c in comm_unique:
                vals = (
                    df_agents_flat[df_agents_flat["comm_radius"] == c][["preset_name", "score"]]
                    .drop_duplicates()["score"]
                    .dropna()
                    .values
                )
                if len(vals) > 0:
                    data_groups.append(vals)
                    labels.append(str(c))
            if len(data_groups) > 1:
                _plot_dark_boxplot(ax_bc, data_groups, labels, ylabel="Score")
                ax_bc.set_xlabel("Raggio comunicazione", color="#cccccc", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_bc)
            plt.close(fig_bc)

        st.markdown("#### Heatmap sinergia strategie")
        st.caption("Score medio dei preset che contengono entrambe le strategie")
        synergy = _build_strategy_synergy_matrix(all_results, STRATEGIES, metric="score")
        if not synergy.empty and synergy.notna().sum().sum() > 0:
            fig_sy, ax_sy = plt.subplots(
                figsize=(8.5, max(4, len(synergy.index) * 0.7)),
                facecolor="#0e1117"
            )
            _style_dark_chart(ax_sy)
            im = ax_sy.imshow(synergy.values, cmap="viridis", aspect="auto")
            ax_sy.set_xticks(range(len(synergy.columns)))
            ax_sy.set_xticklabels(synergy.columns, rotation=45, ha="right", color="#cccccc", fontsize=8)
            ax_sy.set_yticks(range(len(synergy.index)))
            ax_sy.set_yticklabels(synergy.index, color="#cccccc", fontsize=8)
            for i in range(len(synergy.index)):
                for j in range(len(synergy.columns)):
                    val = synergy.iloc[i, j]
                    if pd.notna(val):
                        ax_sy.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)
            cbar = fig_sy.colorbar(im, ax=ax_sy)
            cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
            for label in cbar.ax.yaxis.get_ticklabels():
                label.set_color("#aaaaaa")
            plt.tight_layout()
            st.pyplot(fig_sy)
            plt.close(fig_sy)

        st.markdown("#### Effetto marginale dei fattori sullo score")
        col_me1, col_me2, col_me3 = st.columns(3)
        with col_me1:
            strat_effect = (
                df_agents_flat.groupby("strategy")["score"]
                .mean()
                .sort_values(ascending=False)
                .round(3)
                .reset_index()
            )
            st.dataframe(strat_effect, width='stretch', hide_index=True)
        with col_me2:
            vis_effect = (
                df_agents_flat.groupby("vis_radius")["score"]
                .mean()
                .sort_index()
                .round(3)
                .reset_index()
            )
            st.dataframe(vis_effect, width='stretch', hide_index=True)
        with col_me3:
            comm_effect = (
                df_agents_flat.groupby("comm_radius")["score"]
                .mean()
                .sort_index()
                .round(3)
                .reset_index()
            )
            st.dataframe(comm_effect, width='stretch', hide_index=True)
