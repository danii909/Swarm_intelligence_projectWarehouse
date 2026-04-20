from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ui.constants import (
    AGENT_PALETTE,
    AGENT_RGB_PALETTE,
    CELL_RGB,
    STREAMLIT_BG,
    STREAMLIT_CELL_PX,
    STREAMLIT_COMM_BORDER,
    STREAMLIT_COMM_FILL,
    STREAMLIT_COMM_LINE,
    STREAMLIT_GRID,
)
from ui.helpers import agent_label_rgb

_PYGAME = None
_PYGAME_FONT = None
DEFERRED_PICKUP_MSG = "Ora torno a prenderlo"


def get_pygame():
    global _PYGAME, _PYGAME_FONT
    if _PYGAME is None:
        import pygame
        pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        _PYGAME = pygame
        _PYGAME_FONT = pygame.font.SysFont("monospace", 12, bold=True)
    return _PYGAME


def normalize_pygame_surface(surface):
    try:
        return surface.convert_alpha()
    except Exception:
        return surface


def load_pygame_icon(path_str: str):
    path_str = (path_str or "").strip()
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return None
    try:
        pygame = get_pygame()
        surface = pygame.image.load(str(path))
        return normalize_pygame_surface(surface)
    except Exception:
        return None


def load_uploaded_pygame_icon(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        pygame = get_pygame()
        surface = pygame.image.load(io.BytesIO(uploaded_file.getvalue()), uploaded_file.name)
        return normalize_pygame_surface(surface)
    except Exception:
        return None


def draw_package_pygame(screen, cx: int, cy: int, size: int) -> None:
    pygame = get_pygame()
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


def draw_aa_circle_pygame(screen, cx: int, cy: int, radius: int, color, border_color=None, border_width: int = 0) -> None:
    pygame = get_pygame()
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


def draw_agent_message_pygame(screen, cx: int, cy: int, text: str, grid_px: int) -> None:
    """Disegna un fumetto sopra l'agente che si muove insieme alla sua icona."""
    pygame = get_pygame()
    font = _PYGAME_FONT
    if font is None:
        return

    margin = 6
    txt = font.render(text, True, (245, 245, 245))
    bubble_w = txt.get_width() + margin * 2
    bubble_h = txt.get_height() + margin * 2

    x = cx - bubble_w // 2
    y = cy - 26 - bubble_h
    max_x = max(2, grid_px - bubble_w - 2)
    x = max(2, min(x, max_x))
    y = max(2, y)

    bubble = pygame.Surface((bubble_w, bubble_h), pygame.SRCALPHA)
    pygame.draw.rect(bubble, (20, 24, 36, 220), pygame.Rect(0, 0, bubble_w, bubble_h), border_radius=8)
    pygame.draw.rect(bubble, (210, 210, 210, 230), pygame.Rect(0, 0, bubble_w, bubble_h), 1, border_radius=8)
    bubble.blit(txt, (margin, margin))
    screen.blit(bubble, (x, y))


def render_frame(tick: int, agents, env, show_fog: bool = True, agent_icon_img=None, package_icon_img=None) -> np.ndarray:
    from src.agents.sensors import can_communicate
    from src.environment.grid import CellType

    pygame = get_pygame()
    size = env.grid.size
    px = STREAMLIT_CELL_PX
    half = px // 2

    screen = pygame.Surface((size * px, size * px), pygame.SRCALPHA)
    screen.fill(STREAMLIT_BG)

    for r in range(size):
        for c in range(size):
            ct = CellType(env.grid.data[r][c])
            color = tuple(int(channel * 255) for channel in CELL_RGB[ct])
            rect = pygame.Rect(c * px, r * px, px, px)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, STREAMLIT_GRID, rect, 1)

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
                if 0 <= nr < size and 0 <= nc < size and CellType(env.grid.data[nr][nc]) == CellType.WAREHOUSE:
                    direction = (dr, dc) if ct == CellType.ENTRANCE else opposite[(dr, dc)]
                    break
            color = (255, 255, 255) if (r, c) in all_seen else (68, 68, 68)
            points = [(c * px + int(px * x), r * px + int(px * y)) for x, y in dir_to_triangle[direction]]
            pygame.draw.polygon(screen, color, points)

    for r, c in env._objects:
        cx = c * px + half
        cy = r * px + half
        if package_icon_img is not None:
            icon = pygame.transform.smoothscale(package_icon_img, (int(px * 0.72), int(px * 0.72)))
            screen.blit(icon, (cx - icon.get_width() // 2, cy - icon.get_height() // 2))
        else:
            draw_package_pygame(screen, cx, cy, int(px * 0.62))

    for i, agent in enumerate(agents):
        if not agent.is_active:
            continue
        color = AGENT_RGB_PALETTE[i % len(AGENT_RGB_PALETTE)]
        cx = agent.col * px + half
        cy = agent.row * px + half
        radius = int((agent.visibility_radius + 0.5) * px)
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, 30), (radius, radius), radius)
        pygame.draw.circle(surf, (*color, 80), (radius, radius), radius, 1)
        screen.blit(surf, (cx - radius, cy - radius))

    comm_surf = pygame.Surface((size * px, size * px), pygame.SRCALPHA)
    for i in range(len(agents)):
        a = agents[i]
        if not a.is_active:
            continue
        for j in range(i + 1, len(agents)):
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
                pygame.draw.rect(comm_surf, STREAMLIT_COMM_FILL, rect)
                pygame.draw.rect(comm_surf, STREAMLIT_COMM_BORDER, rect, 2)
                pygame.draw.line(comm_surf, STREAMLIT_COMM_LINE, (x1, y1), (x2, y2), 1)
    screen.blit(comm_surf, (0, 0))

    agent_icon_scaled = None
    if agent_icon_img is not None:
        icon_size = int(px * 0.8)
        agent_icon_scaled = pygame.transform.smoothscale(agent_icon_img, (icon_size, icon_size))

    for i, agent in enumerate(agents):
        color = AGENT_RGB_PALETTE[i % len(AGENT_RGB_PALETTE)] if agent.is_active else (85, 85, 85)
        label_color = agent_label_rgb(color)
        cx = agent.col * px + half
        cy = agent.row * px + half
        radius = max(4, half - 4)

        if agent_icon_scaled is not None:
            pygame.draw.circle(screen, (0, 0, 0), (cx, cy), radius + 2, 2)
            icon = agent_icon_scaled.copy()
            if not agent.is_active:
                icon.fill((255, 255, 255, 110), special_flags=pygame.BLEND_RGBA_MULT)
            screen.blit(icon, (cx - icon.get_width() // 2, cy - icon.get_height() // 2))
        else:
            base_color = color if agent.is_active else (95, 95, 95)
            edge_color = (10, 10, 10) if agent.is_active else (55, 55, 55)
            draw_aa_circle_pygame(screen, cx, cy, radius, base_color, edge_color, border_width=2)
            if agent.is_active:
                hi = tuple(min(255, int(ch * 1.2) + 18) for ch in base_color)
                draw_aa_circle_pygame(screen, cx - max(1, radius // 3), cy - max(1, radius // 3), max(2, radius // 3), hi)
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
            pygame.draw.line(screen, (130, 92, 0), (badge_rect.centerx, badge_rect.top + 1), (badge_rect.centerx, badge_rect.bottom - 2), 1)
            pygame.draw.line(screen, (130, 92, 0), (badge_rect.left + 1, badge_rect.centery), (badge_rect.right - 2, badge_rect.centery), 1)

        if agent_icon_scaled is None:
            label = _PYGAME_FONT.render(str(i + 1), True, label_color)
            screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

        if agent.has_deferred_pickup_message:
            draw_agent_message_pygame(screen, cx, cy, DEFERRED_PICKUP_MSG, size * px)

    return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))


def render_matplotlib_frame(tick: int, agents, env, show_fog: bool = True) -> "plt.Figure":
    """
    Renderizza un frame della simulazione usando matplotlib.
    Ritorna una figura matplotlib pronta per st.pyplot().
    
    Parameters
    ----------
    tick : int
        Numero del tick corrente
    agents : list
        Lista degli agenti
    env : Environment
        Oggetto ambiente
    show_fog : bool
        Se True, oscura le celle non esplorate
    
    Returns
    -------
    plt.Figure
        Figura matplotlib pronta per Streamlit
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from src.environment.grid import CellType
    from src.agents.sensors import can_communicate
    
    # Usa palette canonico da constants
    agent_colors_hex = AGENT_PALETTE
    
    # Palette colori celle
    CELL_COLORS = {
        CellType.EMPTY:     np.array([1.00, 1.00, 1.00]),   # bianco
        CellType.WALL:      np.array([0.22, 0.22, 0.22]),   # grigio scuro
        CellType.WAREHOUSE: np.array([0.29, 0.56, 0.85]),   # blu
        CellType.ENTRANCE:  np.array([0.18, 0.80, 0.44]),   # verde
        CellType.EXIT:      np.array([0.91, 0.30, 0.24]),   # rosso
    }
    
    size = env.grid.size
    
    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    
    # Disegna griglia
    grid_img = np.zeros((size, size, 3), dtype=float)
    for r in range(size):
        for c in range(size):
            ct = CellType(env.grid.data[r][c])
            grid_img[r, c] = CELL_COLORS[ct]
    
    ax.imshow(grid_img, interpolation="nearest", origin="upper", extent=[-0.5, size - 0.5, size - 0.5, -0.5])
    
    # Disegna fog of war
    if show_fog:
        all_seen = set()
        for agent in agents:
            all_seen.update(agent.local_map.keys())
        
        fog_img = np.zeros((size, size, 4), dtype=float)
        fog_img[:, :, :3] = 0.05
        fog_img[:, :, 3] = 0.75
        
        for (r, c) in all_seen:
            fog_img[r, c, 3] = 0.0
        
        ax.imshow(fog_img, interpolation="nearest", origin="upper", extent=[-0.5, size - 0.5, size - 0.5, -0.5], zorder=2)
    
    # Griglia di linee
    for i in range(size + 1):
        ax.axhline(i - 0.5, color="#555", lw=0.3, zorder=1)
        ax.axvline(i - 0.5, color="#555", lw=0.3, zorder=1)
    
    # Disegna pacchi come piccoli rettangoli
    obj_positions = list(env._objects)
    for r, c in obj_positions:
        # Disegna pacco come rettangolo con dimensione 0.5x0.3
        rect = mpatches.FancyBboxPatch(
            (c - 0.25, r - 0.15), 0.5, 0.3,
            boxstyle="round,pad=0.02",
            facecolor="#E8A633",
            edgecolor="#8B5A00",
            linewidth=1.2,
            zorder=6,
        )
        ax.add_patch(rect)
        # Aggiungi linea al centro del pacco
        ax.plot([c - 0.15, c + 0.15], [r, r], color="#8B5A00", linewidth=0.8, zorder=7)
    
    # Disegna linee di comunicazione
    for i in range(len(agents)):
        a = agents[i]
        if not a.is_active:
            continue
        for j in range(i + 1, len(agents)):
            b = agents[j]
            if not b.is_active:
                continue
            if can_communicate(a.pos, b.pos, min(a.comm_radius, b.comm_radius)):
                rect = mpatches.Rectangle(
                    (min(a.col, b.col) - 0.5, min(a.row, b.row) - 0.5),
                    abs(a.col - b.col) + 1,
                    abs(a.row - b.row) + 1,
                    facecolor="#0400FF",
                    edgecolor="#0800A3",
                    linewidth=1.2,
                    alpha=0.18,
                    zorder=6.5,
                )
                ax.add_patch(rect)
                ax.plot([a.col, b.col], [a.row, b.row], color="cyan", alpha=0.4, linewidth=0.8, linestyle=":", zorder=7)
    
    # Disegna raggi di visione degli agenti
    for i, agent in enumerate(agents):
        if not agent.is_active:
            continue
        color = agent_colors_hex[i % len(agent_colors_hex)]
        
        # Converti colore hex a RGB per il fill con trasparenza
        try:
            color_hex = color.lstrip("#")
            color_rgb = tuple(int(color_hex[i:i+2], 16) / 255 for i in (0, 2, 4))
        except Exception:
            color_rgb = (0.5, 0.5, 0.5)
        
        # Raggio di visione (fill + bordo)
        visibility_circle = mpatches.Circle(
            (agent.col, agent.row), 
            agent.visibility_radius,
            facecolor=color_rgb,
            edgecolor=color_rgb,
            linewidth=1.2,
            alpha=0.12,
            zorder=5.5
        )
        ax.add_patch(visibility_circle)
        
        # Bordo più marcato del raggio
        visibility_circle_edge = mpatches.Circle(
            (agent.col, agent.row), 
            agent.visibility_radius,
            facecolor="none",
            edgecolor=color_rgb,
            linewidth=1.0,
            alpha=0.35,
            zorder=5.6
        )
        ax.add_patch(visibility_circle_edge)
    
    # Disegna agenti
    for i, agent in enumerate(agents):
        color = agent_colors_hex[i % len(agent_colors_hex)]
        
        # Cerchio agente
        circle = mpatches.Circle(
            (agent.col, agent.row), 0.38,
            facecolor=color, edgecolor="black", linewidth=0.5, zorder=8
        )
        ax.add_patch(circle)
        
        # Numero agente
        ax.text(agent.col, agent.row, str(i + 1), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=9)
        
        # Indicatore oggetto
        if agent.carrying_object:
            pulse = 1 + 0.3 * abs(np.sin(tick * 0.35))
            ax.scatter(agent.col, agent.row, c="gold", marker="o", s=50 * pulse, 
                      facecolors="none", edgecolors="gold", linewidths=1, zorder=10)

        if agent.has_deferred_pickup_message:
            ax.text(
                agent.col,
                agent.row - 0.72,
                DEFERRED_PICKUP_MSG,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#f5f5f5",
                zorder=11,
                bbox=dict(
                    boxstyle="round,pad=0.36",
                    facecolor="#1c2230",
                    edgecolor="#cfd8dc",
                    linewidth=0.8,
                    alpha=0.92,
                ),
            )
    
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    
    # Spina transparent
    for spine in ax.spines.values():
        spine.set_color("#555")
        spine.set_linewidth(0.5)
    
    fig.tight_layout()
    
    return fig
