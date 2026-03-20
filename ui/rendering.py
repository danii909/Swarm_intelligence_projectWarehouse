from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ui.constants import (
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

    return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
