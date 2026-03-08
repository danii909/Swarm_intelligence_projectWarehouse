"""
Costanti e utilità relative alla griglia dell'ambiente.
"""

from __future__ import annotations
from enum import IntEnum
from typing import List, Tuple


class CellType(IntEnum):
    EMPTY = 0       # Corridoio libero (percorribile)
    WALL = 1        # Ostacolo / scaffale (non percorribile)
    WAREHOUSE = 2   # Interno magazzino
    ENTRANCE = 3    # Ingresso magazzino (porta verde)
    EXIT = 4        # Uscita magazzino (porta rossa)


# Celle percorribili da un agente
WALKABLE = {CellType.EMPTY, CellType.ENTRANCE, CellType.EXIT}

# Mosse cardinali (riga, colonna)
DIRECTIONS: List[Tuple[int, int]] = [
    (-1,  0),   # su
    ( 1,  0),   # giù
    ( 0, -1),   # sinistra
    ( 0,  1),   # destra
]


class Grid:
    """Wrapper della griglia 2D con utilità di accesso e walkability."""

    def __init__(self, data: List[List[int]], size: int) -> None:
        self.data: List[List[int]] = data
        self.size: int = size

    # ------------------------------------------------------------------
    # Accesso celle
    # ------------------------------------------------------------------

    def cell(self, row: int, col: int) -> CellType:
        return CellType(self.data[row][col])

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_walkable(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return False
        return CellType(self.data[row][col]) in WALKABLE

    def is_wall(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return CellType(self.data[row][col]) == CellType.WALL

    # ------------------------------------------------------------------
    # Iteratori
    # ------------------------------------------------------------------

    def walkable_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Restituisce le coordinate delle celle adiacenti percorribili."""
        neighbors = []
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if self.is_walkable(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def delivery_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Celle adiacenti percorribili incluso l'interno magazzino (per consegna/uscita)."""
        neighbors = []
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc) and not self.is_wall(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def all_walkable_cells(self) -> List[Tuple[int, int]]:
        cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_walkable(r, c):
                    cells.append((r, c))
        return cells

    # ------------------------------------------------------------------
    # Rappresentazione
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"Grid(size={self.size})"
