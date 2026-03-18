"""
Classe principale dell'agente autonomo.

Nel setup corrente gli agenti conoscono a priori la mappa statica
(muri, corridoi, porte, magazzini), ma non conoscono la posizione
degli oggetti finche' non li osservano.

Ogni agente ha:
    - Posizione corrente
    - Batteria (500 unita', -1 per mossa)
    - Mappa locale dei tipi cella (nota via bootstrap e condivisa)
    - Copertura percettiva delle celle gia' scansionate per oggetti
    - Insieme di oggetti noti (rilevati ma non ancora raccolti)
    - Stato interno (esplora / trasporta / consegna / fermo)
    - Strategia di esplorazione (iniettata alla creazione)
    - Raggio di visibilita' e comunicazione
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from src.environment.grid import CellType
from src.agents.sensors import compute_visible_cells, can_communicate

if TYPE_CHECKING:
    from src.environment.environment import Environment
    from src.agents.strategies.base import ExplorationStrategy
    from src.pathfinding.pathfinder import Pathfinder


class AgentState(Enum):
    EXPLORING = auto()              # nessun oggetto noto / in esplorazione
    MOVING_TO_OBJECT = auto()       # si sposta verso un oggetto noto
    CARRYING = auto()               # ha un oggetto, si dirige all'ingresso magazzino
    DELIVERING = auto()             # dentro il magazzino, cerca cella interna
    EXITING_WAREHOUSE = auto()      # ha consegnato, si dirige all'uscita
    DEPLETED = auto()               # batteria esaurita


class Agent:
    """Agente autonomo per il recupero oggetti."""

    INITIAL_BATTERY: int = 500
    VISIBILITY_RADIUS: int = 2   # default, sovrascrivibile
    COMM_RADIUS: int = 2

    def __init__(
        self,
        agent_id: int,
        strategy: "ExplorationStrategy",
        visibility_radius: int = VISIBILITY_RADIUS,
        comm_radius: int = COMM_RADIUS,
    ) -> None:
        self.id: int = agent_id
        self.strategy: "ExplorationStrategy" = strategy
        self.visibility_radius: int = visibility_radius
        self.comm_radius: int = comm_radius

        # Stato dinamico
        self.row: int = 0
        self.col: int = 0
        self.battery: int = self.INITIAL_BATTERY
        self.state: AgentState = AgentState.EXPLORING
        self.carrying_object: bool = False

        # Mappa locale: (row, col) → CellType
        self.local_map: Dict[Tuple[int, int], CellType] = {}

        # Celle gia' osservate almeno una volta per ricerca oggetti
        self.seen_cells: Set[Tuple[int, int]] = set()
        # Ultimo tick in cui ciascuna cella e' stata osservata
        self.cell_last_seen: Dict[Tuple[int, int], int] = {}

        # Oggetti rilevati ma non ancora raccolti
        self.known_objects: Set[Tuple[int, int]] = set()

        # Posizioni note di altri agenti: id → (posizione, tick_osservazione)
        # Aggiornato per visibilità diretta e per comunicazione transitiva.
        self.known_agents: Dict[int, Tuple[Tuple[int, int], int]] = {}

        # Statistiche
        self.steps_taken: int = 0
        self.objects_delivered: int = 0

        # Percorso pianificato verso destinazione corrente
        self._current_path: list = []
        self._delivery_target: Optional[Tuple[int, int]] = None
        # Porta di uscita del magazzino corrente (impostata quando si entra)
        self._exit_target: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Posizione
    # ------------------------------------------------------------------

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.row, self.col)

    @property
    def is_active(self) -> bool:
        return self.state != AgentState.DEPLETED

    # ------------------------------------------------------------------
    # Percezione
    # ------------------------------------------------------------------

    def bootstrap_known_map(self, env: "Environment") -> None:
        """
        Inizializza la conoscenza statica della mappa globale.

        Questa operazione non rivela oggetti: popola solo i tipi cella.
        """
        size = env.grid.size
        self.local_map = {
            (r, c): env.grid.cell(r, c)
            for r in range(size)
            for c in range(size)
        }

    def perceive(self, env: "Environment") -> Set[Tuple[int, int]]:
        """
        Calcola le celle visibili, aggiorna la mappa locale e
        rileva eventuali oggetti nel campo visivo.
        Restituisce l'insieme delle celle visibili.
        """
        visible = compute_visible_cells(
            env.grid, self.row, self.col, self.visibility_radius
        )

        # Aggiorna mappa locale
        for (r, c) in visible:
            self.local_map[(r, c)] = env.grid.cell(r, c)
            self.seen_cells.add((r, c))
            self.cell_last_seen[(r, c)] = env.tick

        # Rileva oggetti nel campo visivo
        detected = env.sense_objects(visible)
        self.known_objects.update(detected)

        # Rimuovi voci stale: celle visibili senza più un oggetto
        # (oggetti già raccolti da altri agenti)
        self.known_objects -= (visible - detected)

        return visible

    # ------------------------------------------------------------------
    # Movimento
    # ------------------------------------------------------------------

    def move_to(self, row: int, col: int) -> None:
        """Esegue un singolo passo verso (row, col). Consuma 1 unità di batteria."""
        if not self.is_active:
            return
        self.row = row
        self.col = col
        self.battery -= 1
        self.steps_taken += 1
        if self.battery <= 0:
            self.state = AgentState.DEPLETED

    # ------------------------------------------------------------------
    # Inventario
    # ------------------------------------------------------------------

    def pick_up(self, env: "Environment") -> bool:
        """
        Tenta di raccogliere l'oggetto nella cella corrente.
        Restituisce True se la raccolta è avvenuta.
        """
        if self.carrying_object:
            return False
        if env.pickup_object(self.row, self.col):
            self.carrying_object = True
            self.known_objects.discard(self.pos)
            self.state = AgentState.CARRYING
            return True
        return False

    def deliver(self, env: "Environment") -> bool:
        """
        Gestisce la sequenza di consegna in tre fasi:
          1. ENTRANCE  → entra nel magazzino (stato DELIVERING)
          2. WAREHOUSE → deposita l'oggetto (stato EXITING_WAREHOUSE)
        Restituisce True quando l'oggetto viene effettivamente depositato.
        """
        if not self.carrying_object:
            return False

        cell = env.grid.cell(self.row, self.col)

        if cell == CellType.ENTRANCE:
            # Fase 1: agente sull'ingresso — transizione verso interno
            wh = env.warehouse_for_entrance(self.row, self.col)
            if wh is not None:
                self._exit_target = wh.exit
            self.state = AgentState.DELIVERING
            return False

        if cell == CellType.WAREHOUSE:
            # Fase 2: agente sull'interno — deposita
            env.deliver_object(self.pos)
            self.carrying_object = False
            self.objects_delivered += 1
            self.state = AgentState.EXITING_WAREHOUSE
            return True

        return False

    # ------------------------------------------------------------------
    # Comunicazione (merge mappa e oggetti noti)
    # ------------------------------------------------------------------

    def communicate_with(self, other: "Agent") -> None:
        """Scambia informazioni con un altro agente entro il raggio di comunicazione."""
        if not can_communicate(self.pos, other.pos, self.comm_radius):
            return
        # Merge bidirezionale
        self.local_map.update(other.local_map)
        other.local_map.update(self.local_map)
        self.known_objects.update(other.known_objects)
        other.known_objects.update(self.known_objects)

        self.seen_cells.update(other.seen_cells)
        other.seen_cells.update(self.seen_cells)

        for pos, tick in other.cell_last_seen.items():
            if self.cell_last_seen.get(pos, -1) < tick:
                self.cell_last_seen[pos] = tick
        for pos, tick in self.cell_last_seen.items():
            if other.cell_last_seen.get(pos, -1) < tick:
                other.cell_last_seen[pos] = tick

    # ------------------------------------------------------------------
    # Decisione (delegata alla strategia)
    # ------------------------------------------------------------------

    def decide_next_move(
        self,
        env: "Environment",
        pathfinder: "Pathfinder",
        occupied: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """
        Restituisce la prossima cella (row, col) verso cui muoversi,
        oppure None se l'agente è fermo.
        """
        if not self.is_active:
            return None

        # Fase interna al magazzino: gestita direttamente, non dalla strategia
        if self.state == AgentState.DELIVERING:
            target = env.nearest_warehouse_interior(self.row, self.col)
            if target:
                step = pathfinder.next_step(
                    self.pos, target, occupied - {self.pos}, allow_warehouse=True
                )
                if step:
                    return step
            return None

        if self.state == AgentState.EXITING_WAREHOUSE:
            if self._exit_target:
                # Blocca le celle ENTRANCE: l'agente deve uscire dalla porta rossa,
                # non riattraversare quella verde.
                blocked = (occupied | env.entrance_cells) - {self.pos}
                step = pathfinder.next_step(
                    self.pos, self._exit_target, blocked, allow_warehouse=True
                )
                if step:
                    return step
            return None

        # Esplorazione: aggiunge le celle ENTRANCE/EXIT al set delle posizioni
        # occupate così nessuna strategia ci camminerà sopra per errore
        # (es. random_walk filtra già `occupied` dai vicini).
        return self.strategy.next_move(self, env, pathfinder, occupied | env.door_cells)

    # ------------------------------------------------------------------
    # Rappresentazione
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Agent(id={self.id}, pos={self.pos}, "
            f"battery={self.battery}, state={self.state.name}, "
            f"delivered={self.objects_delivered})"
        )
