"""
Sensori dell'agente: visibilità e comunicazione.

Visibilità
----------
Raggio configurabile (1-3 celle, distanza Manhattan).
L'occlusione è modellata con ray-casting semplice:
una cella è visibile solo se il segmento diretto con il centro
non attraversa muri.

Comunicazione
-------------
Due agenti possono comunicare se la distanza Manhattan tra
le loro posizioni è ≤ raggio di comunicazione.
"""

from __future__ import annotations

from typing import Set, Tuple

from src.environment.grid import Grid


def _bresenham(r0: int, c0: int, r1: int, c1: int):
    """Generatore di celle lungo la linea Bresenham tra (r0,c0) e (r1,c1)."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        yield r0, c0
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


def compute_visible_cells(
    grid: Grid,
    row: int,
    col: int,
    radius: int,
) -> Set[Tuple[int, int]]:
    """
    Restituisce l'insieme delle celle visibili dall'agente in (row, col)
    con il raggio specificato, considerando l'occlusione dei muri.
    """
    visible: Set[Tuple[int, int]] = {(row, col)}
    size = grid.size

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if abs(dr) + abs(dc) > radius:          # Manhattan
                continue
            tr, tc = row + dr, col + dc
            if not grid.in_bounds(tr, tc):
                continue
            # Ray-casting: verifica che nessuna cella intermedia sia un muro
            blocked = False
            for cr, cc in _bresenham(row, col, tr, tc):
                if (cr, cc) == (row, col):
                    continue
                if (cr, cc) == (tr, tc):
                    break
                if grid.is_wall(cr, cc):
                    blocked = True
                    break
            if not blocked:
                visible.add((tr, tc))

    return visible


def can_communicate(
    pos_a: Tuple[int, int],
    pos_b: Tuple[int, int],
    comm_radius: int,
) -> bool:
    """True se i due agenti sono entro il raggio di comunicazione."""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) <= comm_radius
