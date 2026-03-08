"""
Visualizzatore live basato su Pygame.

Controlli tastiera
------------------
  SPACE       — pausa / riprendi
  RIGHT / →   — avanza di 1 tick (in modalità pausa)
  + / =       — aumenta velocità (riduce delay)
  - / _       — diminuisce velocità (aumenta delay)
  D           — toggle debug (mostra oggetti non scoperti)
  F           — toggle fog of war (zone inesplorate oscurate)
  C           — toggle linee di comunicazione
  V           — toggle raggi di visione
  Q / ESC     — ferma la simulazione e chiudi

Requisiti
---------
  pip install pygame
"""

from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

from src.visualization.base import BaseVisualizer
from src.environment.grid import CellType
from src.agents.sensors import can_communicate

if TYPE_CHECKING:
    from src.agents.agent import Agent
    from src.environment.environment import Environment


# ---------------------------------------------------------------------------
# Costanti grafiche
# ---------------------------------------------------------------------------

CELL_PX = 28          # pixel per cella
HUD_WIDTH = 260       # larghezza pannello laterale
FPS_CAP = 120         # FPS massimi

# Colori (R, G, B)
C_BG          = (15,  15,  40)
C_EMPTY       = (240, 240, 240)
C_WALL        = (50,  50,  55)
C_WAREHOUSE   = (74,  144, 217)
C_ENTRANCE    = (46,  204, 112)
C_EXIT        = (231, 76,  60)
C_OBJ_UNKNOWN = (255, 215,  0)   # stella gialla (debug)
C_OBJ_ENV     = (255, 215,  0)   # oggetti nel mondo
C_COMM_LINE   = (0,   255, 255)  # ciano
C_FOG         = (15,  15,  40)   # colore nebbia
C_HUD_BG      = (20,  20,  50)
C_WHITE       = (255, 255, 255)
C_GRAY        = (150, 150, 150)
C_YELLOW      = (255, 215,   0)
C_RED         = (255,  80,  80)
C_GREEN       = (80,  200, 120)
C_ORANGE      = (255, 160,  50)

CELL_COLORS = {
    CellType.EMPTY:     C_EMPTY,
    CellType.WALL:      C_WALL,
    CellType.WAREHOUSE: C_WAREHOUSE,
    CellType.ENTRANCE:  C_ENTRANCE,
    CellType.EXIT:      C_EXIT,
}

AGENT_COLORS = [
    (255, 107,  53),  # arancione
    ( 78, 205, 196),  # teal
    (255, 209, 102),  # giallo
    (168, 230, 207),  # verde menta
    (200, 121, 255),  # viola
    (255, 139, 148),  # rosa
    (  6, 214, 160),  # verde acqua
    ( 17, 138, 178),  # blu scuro
    (239,  71, 111),  # rosso rosa
    (255, 180,  50),  # ambra
]


# ---------------------------------------------------------------------------
# Visualizzatore Pygame
# ---------------------------------------------------------------------------

class PygameVisualizer(BaseVisualizer):
    """
    Visualizzatore interattivo basato su Pygame con controlli tastiera.

    Parameters
    ----------
    cell_px      : int   — pixel per cella (default 28)
    tick_delay   : float — secondi tra un tick e l'altro (default 0.05)
    show_fog     : bool  — fog of war iniziale (toggle con F)
    show_vision  : bool  — raggi di visione iniziali (toggle con V)
    show_comm    : bool  — link comunicazione iniziali (toggle con C)
    show_debug   : bool  — mostra oggetti non ancora scoperti (toggle con D)
    """

    def __init__(
        self,
        cell_px: int = CELL_PX,
        tick_delay: float = 0.05,
        show_fog: bool = True,
        show_vision: bool = True,
        show_comm: bool = True,
        show_debug: bool = False,
    ) -> None:
        self.cell_px = cell_px
        self.tick_delay = tick_delay
        self.show_fog = show_fog
        self.show_vision = show_vision
        self.show_comm = show_comm
        self.show_debug = show_debug

        self._paused = False
        self._step_once = False
        self._running = True

        self._pygame = None
        self._screen = None
        self._font_sm = None
        self._font_md = None
        self._font_lg = None
        self._clock = None

        self._env: Optional["Environment"] = None
        self._agents: Optional[List["Agent"]] = None
        self._size: int = 0
        # Superfici
        self._grid_surface = None   # sfondo statico della griglia
        self._fog_surface = None    # overlay nebbia (aggiornato ogni frame)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, env: "Environment", agents: List["Agent"]) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "pygame non installato. Esegui: pip install pygame"
            ) from exc

        self._pygame = pygame
        self._env = env
        self._agents = agents
        self._size = env.grid.size

        pygame.init()
        pygame.display.set_caption("Swarm Intelligence — Simulazione Live")

        win_w = self._size * self.cell_px + HUD_WIDTH
        win_h = self._size * self.cell_px
        self._screen = pygame.display.set_mode((win_w, win_h))
        self._clock = pygame.time.Clock()

        self._font_sm = pygame.font.SysFont("monospace", 11)
        self._font_md = pygame.font.SysFont("monospace", 13, bold=True)
        self._font_lg = pygame.font.SysFont("monospace", 16, bold=True)

        # Pre-renderizza sfondo griglia (statico)
        self._grid_surface = pygame.Surface((self._size * self.cell_px, self._size * self.cell_px))
        self._bake_grid()

        # Superficie fog of war (alpha)
        self._fog_surface = pygame.Surface(
            (self._size * self.cell_px, self._size * self.cell_px),
            pygame.SRCALPHA,
        )

    def _bake_grid(self) -> None:
        """Disegna la griglia statica (celle) sulla superficie di sfondo."""
        pygame = self._pygame
        surf = self._grid_surface
        px = self.cell_px

        for r in range(self._size):
            for c in range(self._size):
                ct = CellType(self._env.grid.data[r][c])
                color = CELL_COLORS[ct]
                rect = pygame.Rect(c * px, r * px, px, px)
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, (100, 100, 100), rect, 1)

    # ------------------------------------------------------------------
    # Update frame
    # ------------------------------------------------------------------

    def update(self, tick: int, agents: List["Agent"], env: "Environment") -> bool:
        pygame = self._pygame

        # --- Gestione eventi ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False
            if event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

        # Se in pausa, aspetta step manuale
        if self._paused and not self._step_once:
            self._render_frame(tick, agents, env)
            self._clock.tick(FPS_CAP)
            return self._running

        self._step_once = False

        # --- Rendering ---
        self._render_frame(tick, agents, env)
        self._clock.tick(FPS_CAP)

        # Delay
        if self.tick_delay > 0:
            pygame.time.delay(int(self.tick_delay * 1000))

        return self._running

    def _handle_key(self, key: int) -> None:
        pygame = self._pygame
        if key == pygame.K_SPACE:
            self._paused = not self._paused
        elif key in (pygame.K_RIGHT,) and self._paused:
            self._step_once = True
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.tick_delay = max(0.0, self.tick_delay - 0.02)
        elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.tick_delay = min(2.0, self.tick_delay + 0.02)
        elif key == pygame.K_d:
            self.show_debug = not self.show_debug
        elif key == pygame.K_f:
            self.show_fog = not self.show_fog
        elif key == pygame.K_c:
            self.show_comm = not self.show_comm
        elif key == pygame.K_v:
            self.show_vision = not self.show_vision
        elif key in (pygame.K_q, pygame.K_ESCAPE):
            self._running = False

    # ------------------------------------------------------------------
    # Rendering interno
    # ------------------------------------------------------------------

    def _render_frame(
        self, tick: int, agents: List["Agent"], env: "Environment"
    ) -> None:
        pygame = self._pygame
        screen = self._screen
        px = self.cell_px

        screen.fill(C_BG)

        # 1. Sfondo griglia
        screen.blit(self._grid_surface, (0, 0))

        # 2. Fog of war
        if self.show_fog:
            self._draw_fog(agents)
            screen.blit(self._fog_surface, (0, 0))

        # 3. Oggetti nell'ambiente (ground truth in debug, altrimenti solo visibili)
        self._draw_objects(screen, env, agents)

        # 4. Raggi di visione
        if self.show_vision:
            self._draw_vision_radii(screen, agents)

        # 5. Linee comunicazione
        if self.show_comm:
            self._draw_comm_lines(screen, agents)

        # 6. Agenti
        self._draw_agents(screen, agents)

        # 7. HUD laterale
        self._draw_hud(screen, tick, agents, env)

        pygame.display.flip()

    def _draw_fog(self, agents: List["Agent"]) -> None:
        """Aggiorna la superficie fog of war."""
        pygame = self._pygame
        px = self.cell_px
        fog = self._fog_surface
        fog.fill((C_FOG[0], C_FOG[1], C_FOG[2], 200))  # nebbia fitta

        # Svela le celle viste da almeno un agente
        all_seen: Set[Tuple[int, int]] = set()
        for agent in agents:
            all_seen.update(agent.local_map.keys())

        for (r, c) in all_seen:
            pygame.draw.rect(fog, (0, 0, 0, 0), pygame.Rect(c * px, r * px, px, px))

    def _draw_objects(
        self, screen, env: "Environment", agents: List["Agent"]
    ) -> None:
        pygame = self._pygame
        px = self.cell_px
        half = px // 2

        # Posizioni viste dagli agenti (per non mostrare oggetti in fog)
        all_seen: Set[Tuple[int, int]] = set()
        for agent in agents:
            all_seen.update(agent.local_map.keys())

        for (r, c) in env._objects:
            cx = c * px + half
            cy = r * px + half
            if (r, c) in all_seen or self.show_debug:
                # Stella dorata
                self._draw_star(screen, cx, cy, half // 2 + 2, C_OBJ_ENV)

    def _draw_star(self, screen, cx: int, cy: int, r: int, color) -> None:
        """Disegna una stella a 5 punte."""
        import math as _math
        pygame = self._pygame
        points = []
        r_inner = r * 0.45
        for i in range(10):
            angle = _math.radians(-90 + i * 36)
            radius = r if i % 2 == 0 else r_inner
            points.append((cx + radius * _math.cos(angle),
                            cy + radius * _math.sin(angle)))
        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, (200, 160, 0), points, 1)

    def _draw_vision_radii(self, screen, agents: List["Agent"]) -> None:
        pygame = self._pygame
        px = self.cell_px
        half = px // 2
        for i, agent in enumerate(agents):
            if not agent.is_active:
                continue
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            cx = agent.col * px + half
            cy = agent.row * px + half
            radius = int((agent.visibility_radius + 0.5) * px)
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, 30), (radius, radius), radius)
            pygame.draw.circle(surf, (*color, 80), (radius, radius), radius, 1)
            screen.blit(surf, (cx - radius, cy - radius))

    def _draw_comm_lines(self, screen, agents: List["Agent"]) -> None:
        pygame = self._pygame
        px = self.cell_px
        half = px // 2
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
                    surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                    pygame.draw.line(surf, (*C_COMM_LINE, 120), (x1, y1), (x2, y2), 1)
                    screen.blit(surf, (0, 0))

    def _draw_agents(self, screen, agents: List["Agent"]) -> None:
        pygame = self._pygame
        px = self.cell_px
        half = px // 2
        r_agent = max(4, half - 4)

        for i, agent in enumerate(agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            cx = agent.col * px + half
            cy = agent.row * px + half

            if agent.is_active:
                # Cerchio principale
                pygame.draw.circle(screen, color, (cx, cy), r_agent)
                pygame.draw.circle(screen, (255, 255, 255), (cx, cy), r_agent, 1)
                # Pallino bianco se trasporta oggetto
                if agent.carrying_object:
                    pygame.draw.circle(screen, C_WHITE, (cx, cy), r_agent // 3)
            else:
                # Agente esaurito: grigio con X
                pygame.draw.circle(screen, (80, 80, 80), (cx, cy), r_agent)
                pygame.draw.circle(screen, (120, 120, 120), (cx, cy), r_agent, 1)
                off = r_agent // 2
                pygame.draw.line(screen, (60, 60, 60), (cx - off, cy - off), (cx + off, cy + off), 2)
                pygame.draw.line(screen, (60, 60, 60), (cx + off, cy - off), (cx - off, cy + off), 2)

            # ID agente
            lbl = self._font_sm.render(str(i), True, (0, 0, 0))
            screen.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

    def _draw_hud(
        self, screen, tick: int, agents: List["Agent"], env: "Environment"
    ) -> None:
        pygame = self._pygame
        px = self.cell_px
        hud_x = self._size * px
        hud_w = HUD_WIDTH
        hud_h = self._size * px

        # Sfondo HUD
        pygame.draw.rect(screen, C_HUD_BG, pygame.Rect(hud_x, 0, hud_w, hud_h))
        pygame.draw.line(screen, (60, 60, 100), (hud_x, 0), (hud_x, hud_h), 2)

        y = 10
        # Titolo
        t = self._font_lg.render("SWARM SIM", True, C_WHITE)
        screen.blit(t, (hud_x + 10, y))
        y += 28

        # Tick
        t = self._font_md.render(f"Tick: {tick}", True, C_WHITE)
        screen.blit(t, (hud_x + 10, y))
        y += 22

        # Oggetti
        ratio = env.delivered / max(1, env.total_objects)
        color = C_GREEN if ratio >= 1.0 else C_YELLOW if ratio > 0.5 else C_ORANGE
        t = self._font_md.render(
            f"Consegnati: {env.delivered}/{env.total_objects}", True, color
        )
        screen.blit(t, (hud_x + 10, y))
        y += 22

        t = self._font_sm.render(f"In giro:  {env.remaining_objects}", True, C_GRAY)
        screen.blit(t, (hud_x + 10, y))
        y += 30

        # Barre batteria agenti
        t = self._font_md.render("Batteria agenti:", True, C_GRAY)
        screen.blit(t, (hud_x + 10, y))
        y += 18

        bar_w = hud_w - 50
        for i, agent in enumerate(agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            ratio = max(0.0, agent.battery / agent.INITIAL_BATTERY)

            # Label
            lbl = self._font_sm.render(f"A{i}", True, color)
            screen.blit(lbl, (hud_x + 10, y + 2))

            # Sfondo barra
            bar_rect = pygame.Rect(hud_x + 30, y + 4, bar_w, 10)
            pygame.draw.rect(screen, (50, 50, 70), bar_rect, border_radius=3)

            # Barra riempita
            fill_w = int(bar_w * ratio)
            if fill_w > 0:
                bar_color = color if ratio > 0.5 else (255, 165, 0) if ratio > 0.2 else (255, 60, 60)
                fill_rect = pygame.Rect(hud_x + 30, y + 4, fill_w, 10)
                pygame.draw.rect(screen, bar_color, fill_rect, border_radius=3)

            # Valore numerico
            val = self._font_sm.render(f"{agent.battery}", True, (180, 180, 180))
            screen.blit(val, (hud_x + 30 + bar_w + 4, y + 2))

            # Stato
            state_str = "CARRY" if agent.carrying_object else ("OFF" if not agent.is_active else "")
            if state_str:
                s = self._font_sm.render(state_str, True, C_WHITE if agent.carrying_object else C_RED)
                screen.blit(s, (hud_x + 10 + 20 + bar_w - 10, y + 2))

            y += 22

        y += 10
        # Strategie
        t = self._font_md.render("Strategie:", True, C_GRAY)
        screen.blit(t, (hud_x + 10, y))
        y += 18
        for i, agent in enumerate(agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            txt = f"A{i}: {agent.strategy.name[:16]}"
            t = self._font_sm.render(txt, True, color)
            screen.blit(t, (hud_x + 10, y))
            y += 16

        y += 10
        # Controlli
        controls = [
            ("SPACE", "Pausa/Riprendi"),
            ("→",     "Step (pausa)"),
            ("+/-",   f"Velocità ({1/max(0.001,self.tick_delay):.0f} t/s)"),
            ("D",     f"Debug obj: {'ON' if self.show_debug else 'OFF'}"),
            ("F",     f"Fog: {'ON' if self.show_fog else 'OFF'}"),
            ("C",     f"Comm: {'ON' if self.show_comm else 'OFF'}"),
            ("V",     f"Visione: {'ON' if self.show_vision else 'OFF'}"),
            ("Q/ESC", "Esci"),
        ]
        t = self._font_md.render("Controlli:", True, C_GRAY)
        screen.blit(t, (hud_x + 10, y))
        y += 16
        for key, desc in controls:
            k = self._font_sm.render(f"[{key}]", True, C_YELLOW)
            d = self._font_sm.render(f" {desc}", True, (180, 180, 180))
            screen.blit(k, (hud_x + 10, y))
            screen.blit(d, (hud_x + 10 + k.get_width(), y))
            y += 14
            if y > hud_h - 20:
                break

        # Indicatore pausa
        if self._paused:
            pause_surf = pygame.Surface((140, 36), pygame.SRCALPHA)
            pause_surf.fill((0, 0, 0, 160))
            pt = self._font_lg.render("  ⏸ PAUSA  ", True, C_YELLOW)
            pause_surf.blit(pt, (5, 5))
            screen.blit(pause_surf, (10, 10))

    # ------------------------------------------------------------------
    # Chiusura
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._pygame:
            self._pygame.quit()
