"""
Stato globale dell'ambiente: oggetti, magazzini, tick, registrazione agenti.

Il simulatore è l'unico componente che può leggere `objects_truth`; gli agenti
ricevono solo le percezioni filtrate dai sensori.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.environment.grid import Grid, CellType


# ---------------------------------------------------------------------------
# Struttura magazzino
# ---------------------------------------------------------------------------

@dataclass
class Warehouse:
    id: int
    side: str                           # 'top' | 'bottom' | 'left' | 'right'
    entrance: Tuple[int, int]           # (row, col) della porta d'ingresso
    exit: Tuple[int, int]               # (row, col) della porta d'uscita
    cells: Set[Tuple[int, int]] = field(default_factory=set)  # celle interne

    def is_entrance(self, row: int, col: int) -> bool:
        return (row, col) == self.entrance

    def is_exit(self, row: int, col: int) -> bool:
        return (row, col) == self.exit


# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------

class Environment:
    """
    Modello dell'ambiente condiviso tra simulatore e agenti.

    Attributi pubblici per il simulatore
    ------------------------------------
    grid          : Grid         — mappa statica
    warehouses    : List[Warehouse]
    objects_truth : Set[Tuple]   — posizioni ground-truth degli oggetti
                                   (NON accessibile agli agenti)
    delivered     : int          — oggetti consegnati finora
    tick          : int          — tick corrente

    Gli agenti interagiscono tramite:
      - sense_objects(positions)  → oggetti percepiti entro il set di celle
      - deliver_object(pos)       → tenta la consegna da posizione pos
    """

    def __init__(
        self,
        grid: Grid,
        warehouses: List[Warehouse],
        objects_truth: Set[Tuple[int, int]],
    ) -> None:
        self.grid = grid
        self.warehouses: List[Warehouse] = warehouses
        self._objects: Set[Tuple[int, int]] = set(objects_truth)  # ground-truth
        self._initial_total: int = len(self._objects)
        self.delivered: int = 0
        self.tick: int = 0

    # ------------------------------------------------------------------
    # Caricamento da file JSON
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str) -> "Environment":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        size: int = data["metadata"]["grid_size"]
        grid = Grid(data["grid"], size)

        warehouses: List[Warehouse] = []
        for wh_data in data["warehouses"]:
            entrance = tuple(wh_data["entrance"])
            exit_ = tuple(wh_data["exit"])
            side = wh_data["side"]
            wh_id = wh_data["id"]

            # Raccoglie le celle WAREHOUSE interne dal campo "area" del JSON
            cells: Set[Tuple[int, int]] = set()
            for cell_coords in wh_data.get("area", []):
                r, c = cell_coords
                if grid.cell(r, c) == CellType.WAREHOUSE:
                    cells.add((r, c))

            warehouses.append(Warehouse(
                id=wh_id,
                side=side,
                entrance=entrance,
                exit=exit_,
                cells=cells,
            ))

        objects_truth: Set[Tuple[int, int]] = {
            tuple(obj) for obj in data["objects"]
        }

        return cls(grid=grid, warehouses=warehouses, objects_truth=objects_truth)

    # ------------------------------------------------------------------
    # API per gli agenti (percezione filtrata)
    # ------------------------------------------------------------------

    def sense_objects(
        self, visible_cells: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """
        Restituisce gli oggetti presenti nelle celle visibili.
        Gli agenti devono chiamare questo metodo tramite il simulatore.
        """
        return self._objects & visible_cells

    def object_at(self, row: int, col: int) -> bool:
        """True se c'è un oggetto nella cella (row, col)."""
        return (row, col) in self._objects

    def pickup_object(self, row: int, col: int) -> bool:
        """
        Rimuove l'oggetto in (row, col) dall'ambiente (l'agente lo raccoglie).
        Restituisce True se l'oggetto era presente.
        """
        pos = (row, col)
        if pos in self._objects:
            self._objects.discard(pos)
            return True
        return False

    def deliver_object(self, warehouse_entrance: Tuple[int, int]) -> bool:
        """
        Registra la consegna di un oggetto al magazzino.
        Restituisce True se la consegna è valida.
        """
        self.delivered += 1
        return True

    # ------------------------------------------------------------------
    # Stato
    # ------------------------------------------------------------------

    @property
    def total_objects(self) -> int:
        return self._initial_total

    @property
    def remaining_objects(self) -> int:
        return len(self._objects)

    @property
    def all_delivered(self) -> bool:
        return self.delivered == self._initial_total

    def nearest_warehouse_entrance(
        self, row: int, col: int
    ) -> Optional[Tuple[int, int]]:
        """
        Restituisce la posizione dell'ingresso del magazzino più vicino
        (distanza Manhattan) alla cella (row, col).
        """
        best: Optional[Tuple[int, int]] = None
        best_dist = float("inf")
        for wh in self.warehouses:
            er, ec = wh.entrance
            d = abs(er - row) + abs(ec - col)
            if d < best_dist:
                best_dist = d
                best = wh.entrance
        return best

    def warehouse_for_entrance(
        self, row: int, col: int
    ) -> Optional["Warehouse"]:
        """Restituisce il magazzino la cui porta d'ingresso è in (row, col)."""
        for wh in self.warehouses:
            if wh.entrance == (row, col):
                return wh
        return None

    def nearest_warehouse_interior(
        self, row: int, col: int
    ) -> Optional[Tuple[int, int]]:
        """
        Restituisce la cella WAREHOUSE interna più vicina alla posizione corrente.
        Usata dall'agente quando è sull'ingresso per sapere dove depositare.
        """
        # Trova il magazzino a cui appartiene la posizione corrente
        # (l'agente è sull'ENTRANCE oppure già dentro)
        for wh in self.warehouses:
            if wh.entrance == (row, col) or (row, col) in wh.cells:
                if not wh.cells:
                    return None
                return min(
                    wh.cells,
                    key=lambda p: abs(p[0] - row) + abs(p[1] - col),
                )
        return None

    def advance_tick(self) -> None:
        self.tick += 1

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Environment(tick={self.tick}, "
            f"delivered={self.delivered}/{self.total_objects}, "
            f"remaining={self.remaining_objects})"
        )
