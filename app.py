"""
Interfaccia Streamlit per configurare ed eseguire la simulazione Swarm Intelligence.

    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import itertools
import json
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
    0: ("Frontier",       "Esplorazione frontier-based sistematica"),
    1: ("FrontierGreedy", "Greedy verso oggetti noti, fallback frontier"),
    2: ("Sector",         "Suddivide la griglia in settori assegnati"),
    3: ("Spiral",         "Movimento a spirale verso l'esterno"),
    4: ("Random",         "Passeggiata casuale"),
}

DEFAULT_RADIUS = {0: 2, 1: 3, 2: 2, 3: 1, 4: 1}

STRATEGY_COLORS = {
    "Frontier":       "#4C72B0",
    "FrontierGreedy": "#DD8452",
    "Sector":         "#55A868",
    "Spiral":         "#C44E52",
    "Random":         "#8172B2",
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
_STREAMLIT_COMM_FILL = (77, 208, 225, 45)
_STREAMLIT_COMM_BORDER = (0, 188, 212, 180)
_STREAMLIT_COMM_LINE = (0, 255, 255, 120)


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
    from src.agents.strategies.spiral import SpiralStrategy
    from src.agents.strategies.random_walk import RandomWalkStrategy

    factories = {
        0: lambda: FrontierStrategy(),
        1: lambda: GreedyStrategy(),
        2: lambda: SectorStrategy(num_agents=num_agents),
        3: lambda: SpiralStrategy(),
        4: lambda: RandomWalkStrategy(),
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
    html_parts.append(
        "<div style='background:#1a1a2e; border-radius:8px; padding:10px; "
        "border:1px solid #333;'>"
        "<div style='color:#ccc; font-size:0.85em; font-weight:bold; "
        "margin-bottom:8px;'>🔋 Batteria agenti</div>"
    )
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
        strat_name  = strat_by_id.get(agent.id, "?")
        radii_label = f"(v{agent.visibility_radius}, c{agent.comm_radius})"

        html_parts.append(
            f"<div style='margin-bottom:6px;'>"
            f"  <div style='display:flex; justify-content:space-between; "
            f"       align-items:center; margin-bottom:2px;'>"
            f"    <span style='display:inline-flex; align-items:center; gap:6px;'>"
            f"      <span style='background:{agent_color}; color:{agent_text_color}; border:1px solid #000; border-radius:999px; padding:2px 8px; font-size:0.78em; font-weight:bold;'>A{agent.id + 1}</span>"
            f"      <span style='color:#cfd3d8; font-size:0.78em;'>{strat_name}</span>"
            f"      <span style='color:#8ea2b3; font-size:0.72em;'>{radii_label}</span>"
            f"    </span>"
            f"    <span style='color:#999; font-size:0.7em;'>"
            f"      {agent.battery}/{INITIAL_BATTERY} — {state_label}</span>"
            f"  </div>"
            f"  <div style='background:#333; border-radius:4px; height:10px; overflow:hidden;'>"
            f"    <div style='background:{bar_color}; width:{pct_display:.1f}%; "
            f"         height:100%; border-radius:4px; transition: width 0.3s ease;'></div>"
            f"  </div>"
            f"</div>"
        )

    html_parts.append("</div>")
    return "".join(html_parts)


def _style_dark_chart(ax):
    """Applica stile dark coerente a un asse matplotlib."""
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#aaa", labelsize=7)


def _grouped_stats(df: pd.DataFrame, group_col: str, metric_cols: list) -> pd.DataFrame:
    """Calcola mean/min/max/std raggruppando per group_col."""
    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = ["mean", "min", "max", "std"]
    grouped = df.groupby(group_col).agg(agg_dict)
    grouped.columns = [f"{col} {stat}" for col, stat in grouped.columns]
    # Round
    for c in grouped.columns:
        grouped[c] = grouped[c].apply(lambda x: round(x, 2))
    grouped.insert(0, "N simulazioni", df.groupby(group_col).size())
    return grouped


# ---------------------------------------------------------------------------
# Opzioni strategia (definite qui per essere disponibili ovunque)
# ---------------------------------------------------------------------------

strategy_options      = [f"{sid} — {name}  ({desc})" for sid, (name, desc) in STRATEGIES.items()]
strategy_name_options = [name for sid, (name, _) in STRATEGIES.items()]
strategy_ids          = list(STRATEGIES.keys())

# ---------------------------------------------------------------------------
# Sidebar — parametri globali
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configurazione globale")
    st.divider()

    # Istanza
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

    # Parametri simulazione
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
        /* Riduce spazio verticale attorno a ogni slider */
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

    # -------------------------------------------------------------------
    # Layout 3 colonne: [config agenti | simulazione | barre stato]
    # -------------------------------------------------------------------
    col_cfg, col_sim, col_status = st.columns([2, 3, 2])

    # ---- Colonna sinistra: configurazione agenti ----
    with col_cfg:
        # Apply pending preset BEFORE any widget is instantiated
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

        #st.markdown("##### 🧩 Agenti")
        agent_configs = []

        for agent_id in range(num_agents):
            default_sid = agent_id % len(STRATEGIES)
            default_r   = DEFAULT_RADIUS[default_sid]
            #current_strat = st.session_state.get(f"strat_{agent_id}", strategy_name_options[default_sid])

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

    # ---- Colonna centrale: frame simulazione ----
    with col_sim:
        frame_ph = st.empty()

    # ---- Colonna destra: metriche live e batterie ----
    with col_status:
        tick_ph    = st.empty()
        prog_ph    = st.empty()
        stats_ph   = st.empty()
        battery_ph = st.empty()

    # -------------------------------------------------------------------
    # Esecuzione simulazione (fuori dai context delle colonne,
    # aggiorna i placeholder definiti sopra)
    # -------------------------------------------------------------------
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

        tick_ph.metric("Tick", 0)
        prog_ph.progress(0.0, text=f"Tick 0/{max_ticks}")
        stats_ph.metric("Consegnati", f"0 / {total_objects_preview}")
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

        # Senza upload usiamo il fallback vettoriale (piu nitido a piccole dimensioni).
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
                    tick_ph.metric("Tick", tick)
                    stats_ph.metric(
                        "Consegnati",
                        f"{cur_env.delivered} / {cur_env.total_objects}",
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

        # --------------------------------------------------------------
        # Risultati (larghezza piena, sotto le 3 colonne)
        # --------------------------------------------------------------
        st.divider()
        st.subheader("📊 Risultati")

        r1, r2, r3, r4, r5 = st.columns(5)
        delivered  = summary["objects_delivered"]
        total      = summary["total_objects"]
        rate_pct   = summary["delivery_rate"] * 100
        ticks_done = summary["total_ticks"]
        energy     = summary["average_energy_consumed"]

        r1.metric("Oggetti consegnati", f"{delivered} / {total}")
        r2.metric("Completamento", f"{rate_pct:.1f}%")
        r3.metric("Tick totali", ticks_done)
        r4.metric("Energia media", f"{energy:.1f}")
        r5.metric("Tempo CPU", f"{elapsed:.2f}s")

        #st.progress(summary["delivery_rate"], text=f"Completamento: {rate_pct:.1f}%")

        # Dettaglio per agente
        st.subheader("🤖 Dettaglio agenti")
        steps_list     = summary.get("agent_steps", [])
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

        # Grafici
        if metrics.history:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.subheader("📈 Consegne nel tempo")
                ticks_list_h   = [s.tick      for s in metrics.history]
                delivered_list = [s.delivered for s in metrics.history]
                remaining_list = [s.remaining for s in metrics.history]
                df_hist = pd.DataFrame({
                    "Tick": ticks_list_h,
                    "Consegnati": delivered_list,
                    "Rimanenti":  remaining_list,
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

        # Download preset corrente
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

    # -------------------------------------------------------------------
    # Storico run
    # -------------------------------------------------------------------
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
    st.subheader("🔬 Benchmark esplorativo")
    st.caption("Genera N preset casuali variando il comportamento di ogni agente.")
    st.divider()


    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        with st.expander("⚙️ Parametri generali", expanded=False):
            bench_num_agents   = st.slider("N. agenti",    min_value=1,   max_value=10,  value=5,      key="bench_num_agents")
            bench_max_ticks    = st.slider("Tick massimi", min_value=100, max_value=750, value=500, step=50, key="bench_max_ticks")
            
    with col_mid:
        with st.expander("🎲 Randomizzazione", expanded=False):
            st.markdown("**Strategia**")
            rand_strat = st.toggle("Randomizza strategia", value=True, key="rand_strat")
            if rand_strat:
                bench_strategies   = st.multiselect(
                    "Strategie possibili",
                    options=[f"{sid} — {name}" for sid, (name, _) in STRATEGIES.items()],
                    default=[f"{sid} — {name}" for sid, (name, _) in STRATEGIES.items()],
                    key="bench_strats",
                )
                bench_strategy_ids = [int(s.split(" — ")[0]) for s in bench_strategies]
            else:
                fixed_strat_label  = st.selectbox(
                    "Strategia fissa",
                    options=[f"{sid} — {name}" for sid, (name, _) in STRATEGIES.items()],
                    index=1, key="bench_fixed_strat",
                )
                bench_strategy_ids = [int(fixed_strat_label.split(" — ")[0])]

            st.markdown("**Raggi**")
            rand_vis = st.toggle("Raggio visione casuale", value=True, key="rand_vis")
            if rand_vis:
                bench_vis_range = st.slider("Range visione", min_value=1, max_value=5, value=(1, 3), key="bench_vis_range")
                vis_values = list(range(bench_vis_range[0], bench_vis_range[1] + 1))
            else:
                vis_values = [st.slider("Visione fissa", min_value=1, max_value=5, value=2, key="bench_fixed_vis")]

            rand_comm = st.toggle("Raggio comunicazione casuale", value=True, key="rand_comm")
            if rand_comm:
                bench_comm_range = st.slider("Range comunicazione", min_value=1, max_value=2, value=(1, 2), key="bench_comm_range")
                comm_values = list(range(bench_comm_range[0], bench_comm_range[1] + 1))
            else:
                comm_values = [st.slider("Comunicazione fissa", min_value=1, max_value=2, value=2, key="bench_fixed_comm")]

    # ---- Colonna destra: esecuzione ----
    choices_per_agent  = len(bench_strategy_ids) * len(vis_values) * len(comm_values)
    max_unique_presets = choices_per_agent ** bench_num_agents

    with col_right:
        st.markdown(f"**Preset generati**: fino a {max_unique_presets} combinazioni uniche.")

        bench_n = st.number_input(
            "N. preset da testare",
            min_value=1,
            max_value=max_unique_presets,
            value=min(20, max_unique_presets),
            step=1,
            key="bench_n"
        )

        bench_clicked = st.button(
            "▶ Avvia benchmark",
            type="primary",
            width="stretch",
            key="bench_run"
        )


    if bench_clicked:
        if not bench_strategy_ids:
            st.error("Seleziona almeno una strategia.")
            st.stop()

        if not os.path.isfile(instance_path):
            st.error(f"File istanza non trovato: `{instance_path}`")
            st.stop()

        from src.environment.environment import Environment
        from src.simulation.simulator import Simulator

        # --- Genera preset casuali ---
        actual_n = min(bench_n, max_unique_presets)
        rng = _random.Random(seed if seed >= 0 else None)

        # Ogni preset è una tupla di (strategy_id, vis, comm) per ogni agente
        per_agent_space = list(itertools.product(bench_strategy_ids, vis_values, comm_values))
        generated_presets = []
        seen_signatures = set()

        if actual_n >= max_unique_presets:
            # Enumera tutti
            for combo in itertools.product(per_agent_space, repeat=bench_num_agents):
                generated_presets.append(list(combo))
        else:
            # Campiona senza duplicati
            attempts = 0
            max_attempts = actual_n * 50
            while len(generated_presets) < actual_n and attempts < max_attempts:
                preset = tuple(rng.choice(per_agent_space) for _ in range(bench_num_agents))
                sig = preset  # tuple of tuples, hashable
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    generated_presets.append(list(preset))
                attempts += 1

        actual_n = len(generated_presets)

        bench_progress = st.progress(0.0, text="Avvio benchmark...")
        bench_status = st.empty()

        all_results = []
        t0_bench = time.perf_counter()

        for sim_i, preset in enumerate(generated_presets):
            preset_name = f"Preset {sim_i + 1}"

            # Costruisci configurazione agenti da questo preset
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

            # Descrizione compatta del preset
            config_parts = []
            for ai, (strat_id, vis_r, comm_r) in enumerate(preset):
                sn = STRATEGIES[strat_id][0][:4]
                config_parts.append(f"A{ai + 1}:{sn}(v{vis_r},c{comm_r})")
            config_str = " ".join(config_parts)

            # Conteggio strategie nel team
            strat_counts = {}
            for strat_id, _, _ in preset:
                sname = STRATEGIES[strat_id][0]
                strat_counts[sname] = strat_counts.get(sname, 0) + 1
            team_desc = " + ".join(f"{cnt}×{sn}" for sn, cnt in sorted(strat_counts.items()))

            # Raggi medi del team
            avg_vis = np.mean([vis_r for _, vis_r, _ in preset])
            avg_comm = np.mean([comm_r for _, _, comm_r in preset])

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
            })

            pct = (sim_i + 1) / actual_n
            bench_progress.progress(pct, text=f"Preset {sim_i+1}/{actual_n}")
            bench_status.markdown(
                f"**{preset_name}** — {team_desc} — "
                f"{s['objects_delivered']}/{s['total_objects']} in {s['total_ticks']} tick"
            )

        total_bench_time = time.perf_counter() - t0_bench
        bench_progress.progress(1.0, text="Benchmark completato!")
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
        bench_strategy_ids = _br["bench_strategy_ids"]
        vis_values = _br["vis_values"]

        # ==============================================================
        # RISULTATI BENCHMARK
        # ==============================================================

        df = pd.DataFrame(all_results)
        total_obj = df["total_objects"].iloc[0]

        st.divider()
        st.subheader("📊 Risultati benchmark")

        # --- Metriche aggregate ---
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Preset testati", actual_n)
        mc2.metric("Completamento medio", f"{df['delivery_rate'].mean()*100:.1f}%")
        mc3.metric("Tick medi", f"{df['total_ticks'].mean():.1f}")
        mc4.metric("Tempo totale CPU", f"{total_bench_time:.2f}s")

        _csv_cols = ["preset_name", "config_str", "team_desc", "avg_vis", "avg_comm",
                     "objects_delivered", "total_objects", "delivery_rate",
                     "total_ticks", "average_energy", "cpu_time"]
        _csv_bytes = df[_csv_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Scarica tutti i risultati (CSV)",
            data=_csv_bytes,
            file_name="benchmark_results.csv",
            mime="text/csv",
            key="dl_all_csv",
            help="Salva subito i risultati su disco — resistono a qualsiasi ricaricamento della pagina",
        )

        # ==============================================================
        # CLASSIFICA PRESET MIGLIORI
        # ==============================================================

        st.divider()
        st.markdown("#### 🏆 Classifica preset migliori")
        st.caption("Ordinata per: completamento (desc) → tick (asc) → energia (asc)")

        df_rank = df.sort_values(
            by=["delivery_rate", "total_ticks", "average_energy"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

        df_rank_display = pd.DataFrame({
            "Pos.": range(1, len(df_rank) + 1),
            "Preset": df_rank["preset_name"],
            "Team": df_rank["team_desc"],
            "Configurazione": df_rank["config_str"],
            "Consegnati": [f"{d}/{total_obj}" for d in df_rank["objects_delivered"]],
            "Completamento": [f"{r*100:.1f}%" for r in df_rank["delivery_rate"]],
            "Tick": df_rank["total_ticks"],
            "Energia media": df_rank["average_energy"].round(1),
        })
        st.dataframe(df_rank_display, width='stretch', hide_index=True)

        # Top 3 dettaglio
        if len(df_rank) >= 1:
            st.markdown("**Top 3:**")
            medals = ["🥇", "🥈", "🥉"]
            for i in range(min(3, len(df_rank))):
                row = df_rank.iloc[i]
                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(
                        f"{medals[i]} **{row['preset_name']}** — {row['team_desc']} — "
                        f"consegnati {row['objects_delivered']}/{total_obj} "
                        f"in **{row['total_ticks']} tick** — "
                        f"energia {row['average_energy']:.1f}"
                    )
                    st.caption(f"    Dettaglio: {row['config_str']}")
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

        # ==============================================================
        # TABELLA COMPLETA
        # ==============================================================

        st.divider()
        st.markdown("#### 📋 Tutti i preset")

        df_all_display = pd.DataFrame({
            "Preset": df["preset_name"],
            "Team": df["team_desc"],
            "Configurazione": df["config_str"],
            "Consegnati": df["objects_delivered"],
            "Tick": df["total_ticks"],
            "Energia media": df["average_energy"].round(1),
            "CPU (s)": df["cpu_time"],
        })
        st.dataframe(df_all_display, width='stretch', hide_index=True)

        # ==============================================================
        # RAGGRUPPAMENTI
        # ==============================================================

        # Esplodi le configurazioni per-agente per analisi per strategia
        agent_level_rows = []
        for r in all_results:
            for strat_id, vis_r, comm_r in r["preset_raw"]:
                agent_level_rows.append({
                    "preset_name": r["preset_name"],
                    "strategy": STRATEGIES[strat_id][0],
                    "vis_radius": vis_r,
                    "comm_radius": comm_r,
                    "total_ticks": r["total_ticks"],
                    "average_energy": r["average_energy"],
                    "objects_delivered": r["objects_delivered"],
                })
        df_agents_flat = pd.DataFrame(agent_level_rows)

        metric_cols = ["total_ticks", "average_energy", "objects_delivered"]

        # --- Per strategia (contando quante volte compare e performance media dei team che la usano) ---
        st.divider()
        st.markdown("#### 📊 Raggruppamento per strategia")
        st.caption("Performance media dei preset che contengono ciascuna strategia")
        df_by_strat = _grouped_stats(df_agents_flat, "strategy", metric_cols)
        st.dataframe(df_by_strat, width='stretch')

        # --- Per raggio visione ---
        st.markdown("#### 📊 Raggruppamento per raggio visione")
        st.caption("Performance media dei preset che contengono agenti con ciascun raggio")
        df_by_vis = _grouped_stats(df_agents_flat, "vis_radius", metric_cols)
        st.dataframe(df_by_vis, width='stretch')

        # --- Per raggio Comunicazione ---
        st.markdown("#### 📊 Raggruppamento per raggio Comunicazione")
        df_by_comm = _grouped_stats(df_agents_flat, "comm_radius", metric_cols)
        st.dataframe(df_by_comm, width='stretch')

        # ==============================================================
        # GRAFICI
        # ==============================================================

        st.divider()
        st.subheader("📈 Grafici di analisi")

        # --- 1. Tick per preset (bar chart) ---
        st.markdown("#### Tick per preset")
        _fig_w = min(max(10, actual_n * 0.35), 38)
        fig1, ax1 = plt.subplots(figsize=(_fig_w, 4), facecolor="#0e1117")
        _style_dark_chart(ax1)

        x_labels_bar = [f"P{i+1}" for i in range(actual_n)]
        # Colore basato sulla strategia dominante nel team
        dominant_colors = []
        for r in all_results:
            strat_counts = {}
            for sid, _, _ in r["preset_raw"]:
                sn = STRATEGIES[sid][0]
                strat_counts[sn] = strat_counts.get(sn, 0) + 1
            dominant = max(strat_counts, key=strat_counts.get)
            dominant_colors.append(STRATEGY_COLORS.get(dominant, "#888"))

        ax1.bar(range(actual_n), df["total_ticks"], color=dominant_colors,
                edgecolor="#444", linewidth=0.5)
        ax1.axhline(y=df["total_ticks"].mean(), color="#FFD700", linestyle="--",
                    linewidth=1.5, label=f"Media: {df['total_ticks'].mean():.1f}")
        if actual_n <= 60:
            ax1.set_xticks(range(actual_n))
            ax1.set_xticklabels(x_labels_bar, fontsize=6, color="#aaa")
        else:
            ax1.set_xticks([])
        ax1.set_ylabel("Tick", color="#ccc", fontsize=9)
        ax1.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # --- 2. Consegne per preset (bar chart) ---
        st.markdown("#### Oggetti consegnati per preset")
        fig2, ax2 = plt.subplots(figsize=(_fig_w, 4), facecolor="#0e1117")
        _style_dark_chart(ax2)

        delivered_vals = df["objects_delivered"].values
        colors_bar2 = [
            "#55A868" if d == total_obj else "#DD8452" if d > total_obj * 0.5 else "#C44E52"
            for d in delivered_vals
        ]
        ax2.bar(range(actual_n), delivered_vals, color=colors_bar2,
                edgecolor="#444", linewidth=0.5)
        ax2.axhline(y=total_obj, color="#FFD700", linestyle=":", linewidth=1,
                    label=f"Totale: {total_obj}")
        if actual_n <= 60:
            ax2.set_xticks(range(actual_n))
            ax2.set_xticklabels(x_labels_bar, fontsize=6, color="#aaa")
        else:
            ax2.set_xticks([])
        ax2.set_ylabel("Consegnati", color="#ccc", fontsize=9)
        ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # --- 3. Box plot per strategia ---
        strat_names_present = df_agents_flat["strategy"].unique().tolist()
        if len(strat_names_present) > 1:
            st.markdown("#### Box plot tick per strategia (team che la contengono)")
            fig3, ax3 = plt.subplots(figsize=(10, 4), facecolor="#0e1117")
            _style_dark_chart(ax3)
            # Deduplica: per ogni strategia, prendi i tick dei preset unici che la contengono
            data_per_strat = []
            for sname in strat_names_present:
                preset_ticks = df_agents_flat[
                    df_agents_flat["strategy"] == sname
                ]["total_ticks"].drop_duplicates().values
                data_per_strat.append(preset_ticks if len(preset_ticks) > 0 else [0])
            bp = ax3.boxplot(
                data_per_strat, patch_artist=True, widths=0.5,
                medianprops=dict(color="#FFD700", linewidth=2),
                whiskerprops=dict(color="#aaa"),
                capprops=dict(color="#aaa"),
                flierprops=dict(markerfacecolor="#C44E52", marker="o", markersize=5),
            )
            for patch, sname in zip(bp["boxes"], strat_names_present):
                patch.set_facecolor(STRATEGY_COLORS.get(sname, "#888"))
                patch.set_edgecolor(STRATEGY_COLORS.get(sname, "#888"))
            ax3.set_xticklabels(strat_names_present, color="#ccc", fontsize=8)
            ax3.set_ylabel("Tick", color="#ccc", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        # --- 4. Box plot per raggio visione ---
        vis_unique = sorted(df_agents_flat["vis_radius"].unique())
        if len(vis_unique) > 1:
            st.markdown("#### Box plot tick per raggio visione")
            fig4, ax4 = plt.subplots(figsize=(10, 4), facecolor="#0e1117")
            _style_dark_chart(ax4)
            data_per_vis = [
                df_agents_flat[df_agents_flat["vis_radius"] == v]["total_ticks"].drop_duplicates().values
                for v in vis_unique
            ]
            bp4 = ax4.boxplot(
                data_per_vis, patch_artist=True, widths=0.5,
                boxprops=dict(facecolor="#4ECDC4", color="#4ECDC4"),
                medianprops=dict(color="#FFD700", linewidth=2),
                whiskerprops=dict(color="#aaa"),
                capprops=dict(color="#aaa"),
                flierprops=dict(markerfacecolor="#C44E52", marker="o", markersize=5),
            )
            ax4.set_xticklabels([str(v) for v in vis_unique], color="#ccc", fontsize=8)
            ax4.set_xlabel("Raggio visione", color="#ccc", fontsize=9)
            ax4.set_ylabel("Tick", color="#ccc", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

        # --- 5. Box plot per raggio Comunicazione ---
        comm_unique = sorted(df_agents_flat["comm_radius"].unique())
        if len(comm_unique) > 1:
            st.markdown("#### Box plot tick per raggio Comunicazione")
            fig5, ax5 = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
            _style_dark_chart(ax5)
            data_per_comm = [
                df_agents_flat[df_agents_flat["comm_radius"] == c]["total_ticks"].drop_duplicates().values
                for c in comm_unique
            ]
            bp5 = ax5.boxplot(
                data_per_comm, patch_artist=True, widths=0.4,
                boxprops=dict(facecolor="#FF6B35", color="#FF6B35"),
                medianprops=dict(color="#FFD700", linewidth=2),
                whiskerprops=dict(color="#aaa"),
                capprops=dict(color="#aaa"),
                flierprops=dict(markerfacecolor="#C44E52", marker="o", markersize=5),
            )
            ax5.set_xticklabels([str(c) for c in comm_unique], color="#ccc", fontsize=8)
            ax5.set_xlabel("Raggio Comunicazione", color="#ccc", fontsize=9)
            ax5.set_ylabel("Tick", color="#ccc", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)

        # --- 6. Scatter: tick vs consegne ---
        st.markdown("#### Scatter tick vs consegne (colore = strategia dominante)")
        fig6, ax6 = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
        _style_dark_chart(ax6)
        ax6.scatter(
            df["total_ticks"], df["objects_delivered"],
            c=dominant_colors, s=60, edgecolors="#555", linewidths=0.5, zorder=5,
        )
        ax6.axhline(y=total_obj, color="#FFD700", linestyle=":", linewidth=1, alpha=0.6)
        ax6.set_xlabel("Tick", color="#ccc", fontsize=9)
        ax6.set_ylabel("Consegnati", color="#ccc", fontsize=9)
        # Legenda manuale per strategie presenti
        legend_handles = [
            mpatches.Patch(color=STRATEGY_COLORS.get(sn, "#888"), label=sn)
            for sn in strat_names_present
        ]
        ax6.legend(handles=legend_handles, fontsize=8,
                   facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

        # --- 7. Heatmap: strategia dominante × vis media (tick medi) ---
        # Costruisci colonna "dominant strategy" e "vis bucket" a livello preset
        df["dominant_strategy"] = [
            max(
                {STRATEGIES[sid][0]: 0 for sid in bench_strategy_ids} |
                {STRATEGIES[sid][0]: sum(1 for s, _, _ in r["preset_raw"] if s == sid)
                 for sid in bench_strategy_ids},
                key=lambda sn: sum(1 for s, _, _ in r["preset_raw"] if STRATEGIES[s][0] == sn)
            )
            for r in all_results
        ]
        pivot = df.pivot_table(
            values="total_ticks", index="dominant_strategy",
            columns=pd.cut(df["avg_vis"], bins=max(1, len(vis_values)),
                          include_lowest=True) if len(vis_values) > 1 else "avg_vis",
            aggfunc="mean"
        )
        if not pivot.empty and pivot.shape[0] > 0 and pivot.shape[1] > 1:
            st.markdown("#### Heatmap: strategia dominante x raggio visione medio (tick)")
            fig7, ax7 = plt.subplots(
                figsize=(8, max(3, len(pivot.index) * 0.8)), facecolor="#0e1117"
            )
            _style_dark_chart(ax7)
            im = ax7.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
            ax7.set_xticks(range(len(pivot.columns)))
            ax7.set_xticklabels([str(c) for c in pivot.columns], color="#ccc", fontsize=7)
            ax7.set_yticks(range(len(pivot.index)))
            ax7.set_yticklabels(pivot.index.tolist(), color="#ccc")
            ax7.set_xlabel("Raggio visione medio", color="#ccc", fontsize=9)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax7.text(j, i, f"{val:.0f}", ha="center", va="center",
                                color="white", fontsize=9, fontweight="bold")
            cbar = fig7.colorbar(im, ax=ax7)
            cbar.ax.yaxis.set_tick_params(color="#aaa")
            for label in cbar.ax.yaxis.get_ticklabels():
                label.set_color("#aaa")
            plt.tight_layout()
            st.pyplot(fig7)
            plt.close(fig7)