from __future__ import annotations

from src.environment.grid import CellType

STRATEGIES = {
    0: ("Frontier", "Esplorazione frontier-based sistematica"),
    1: ("Greedy", "Esplorazione warehouse-centric"),
    2: ("Sector", "Suddivide la griglia in settori assegnati"),
    3: ("Repulsion", "Dispersione emergente dagli altri agenti"),
    4: ("Smart Random", "Random walk guidato da info gain, stale e separazione"),
    5: ("Ant-Colony", "Coverage con feromone evaporativo condiviso"),
}

DEFAULT_RADIUS = {0: 2, 1: 3, 2: 2, 3: 2, 4: 1, 5: 2}

STRATEGY_COLORS = {
    "Frontier": "#4C72B0",
    "Greedy": "#DD8452",
    "Sector": "#55A868",
    "Repulsion": "#C44E52",
    "Smart Random": "#8172B2",
    "Ant-Colony": "#1F9D8A",
}

CELL_RGB = {
    CellType.EMPTY: (1.00, 1.00, 1.00),
    CellType.WALL: (0.22, 0.22, 0.22),
    CellType.WAREHOUSE: (0.29, 0.56, 0.85),
    CellType.ENTRANCE: (0.18, 0.80, 0.44),
    CellType.EXIT: (0.91, 0.30, 0.24),
}

AGENT_PALETTE = [
    "#FF1F1F", "#0057FF", "#00B050", "#000000", "#FFD400",
    "#FF8B94", "#06D6A0", "#118AB2", "#EF476F", "#7A1CAC",
]

AGENT_RGB_PALETTE = [
    (255, 31, 31), (0, 87, 255), (0, 176, 80), (0, 0, 0), (255, 212, 0),
    (255, 139, 148), (6, 214, 160), (17, 138, 178), (239, 71, 111), (122, 28, 172),
]

INITIAL_BATTERY = 500
AGENT_ICON_DEFAULT_PATH = "assets/agent.png"
PACKAGE_ICON_DEFAULT_PATH = "assets/package.png"
DEFAULT_FRAME_DELAY = 0.15

STREAMLIT_CELL_PX = 28
STREAMLIT_BG = (15, 15, 40)
STREAMLIT_GRID = (100, 100, 100)
STREAMLIT_COMM_FILL = (30, 144, 255, 50)
STREAMLIT_COMM_BORDER = (0, 0, 139, 200)
STREAMLIT_COMM_LINE = (0, 0, 139, 200)

strategy_options = [f"{sid} — {name}  ({desc})" for sid, (name, desc) in STRATEGIES.items()]
strategy_name_options = [name for sid, (name, _) in STRATEGIES.items()]
strategy_ids = list(STRATEGIES.keys())
