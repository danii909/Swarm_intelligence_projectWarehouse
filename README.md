# E.L.B.E.R.R. — Swarm Intelligence Warehouse Exploration

**Efficient Logistics by Exploration with Robotic Retrieval**

A real-time multi-agent simulation framework where autonomous agents explore warehouse environments, discover objects, and optimize retrieval through swarm intelligence principles. Built with Python (Streamlit) backend for interactive visualization and benchmarking.

Repository: [SwarmLab](https://github.com/danii909/SwarmLab)

Live demo: [swarm-lab.streamlit.app](https://swarm-lab.streamlit.app)

---

## 🚀 Quick Start — Run with Streamlit

### Prerequisites
- **Python 3.9+**
- Virtual environment (optional but recommended)

### Setup & Run (3 steps)

```bash
# 1. Activate your virtual environment (if you have one)
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open your browser at **`http://localhost:8501`** — you'll see the interactive dashboard with:
- Live grid visualization
- Agent configuration panel
- Real-time metrics & battery monitoring
- Preset save/load
- Benchmark mode for batch testing

---

## Features

### Exploration Strategies (5 Usable + 1 Prototype)

| Strategy      | Colour    | Approach                                                   |
| ------------- | --------- | ---------------------------------------------------------- |
| Frontier      | Blue      | Frontier-based systematic exploration with target lock     |
| Greedy        | Orange    | Warehouse-centric greedy search                            |
| Sector        | Green     | Grid-partitioning with assigned sectors per agent          |
| Repulsion     | Red       | Emergent dispersion based on inter-agent repulsion         |
| Smart Random  | Purple    | Information-gain guided random walk with stale avoidance   |
| Ant-Colony    | Teal      | Experimental prototype (currently not usable)              |

### Real-time interactive UI

- **Live grid visualization** with Pygame offscreen rendering — fog-of-war, agent vision radius, communication range
- **Three-column layout**: agent configuration | simulation grid | live metrics & battery bars
- **Preset system** — save/load agent configurations as JSON for reproducible runs
- **Battery monitoring** — real-time per-agent energy consumption tracking
- **Run history** — compare multiple simulation runs side-by-side
- **Full-screen benchmarking** — automated batch testing of strategy combinations

### Comprehensive benchmarking

- **Random preset generation** — vary usable strategies, vision radius, communication radius independently
- **Exhaustive or sampled search** — generate unique presets up to the full combinatorial space
- **Multi-run execution** with configurable random seeds
- **Delivery curves** — cumulative object retrieval over time (per preset)
- **CSV export** — download full results for external analysis
- **Top-10 rankings** — filters by tick count, completion rate, energy consumption
- **Ant-Colony excluded** — currently treated as a prototype and not used in benchmark runs

### Configurable environments

- Two provided warehouse instances (`A.json`, `B.json`) with identical structure:
  - Grid size: 25×25 cells
  - 4 warehouses with entrance/exit per warehouse
  - 10 objects to retrieve per instance
- Easy JSON format for custom scenarios

---

## Architecture

```text
├── 📄 README.md                    (this file)
├── 📄 requirements.txt             (Python dependencies)
├── 📄 app.py                       (Streamlit entry point)
├── 🐍 benchmark_strategies.py      (CLI benchmarking script)
│
├── 📁 assets/                      (Images: logo, icons)
│   ├── 2.png
│   ├── agent.png
│   └── package.png
│
├── 📁 ui/                          (Streamlit UI modules)
│   ├── 📄 __init__.py
│   └── (future: component split)
│
├── 📁 src/
│   ├── 📁 agents/
│   │   ├── 🐍 agent.py             (Core Agent class)
│   │   ├── 🐍 sensors.py           (Perception: visibility, communication)
│   │   └── 📁 strategies/
│   │       ├── 🐍 base.py          (Strategy interface)
│   │       ├── 🐍 frontier.py      (Frontier exploration)
│   │       ├── 🐍 greedy.py        (Warehouse-centric greedy)
│   │       ├── 🐍 sector.py        (Sector partitioning)
│   │       ├── 🐍 Repulsion.py     (Emergent repulsion)
│   │       ├── 🐍 random_walk.py   (Info-gain random walk)
│   │       └── 🐍 ant_colony_lite.py (Pheromone-based)
│   │
│   ├── 📁 environment/
│   │   ├── 🐍 environment.py       (Simulator state & logic)
│   │   ├── 🐍 grid.py              (Cell types, grid data)
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 simulation/
│   │   ├── 🐍 simulator.py         (Tick loop & step generator)
│   │   ├── 🐍 metrics.py           (Statistics collection)
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 pathfinding/
│   │   ├── 🐍 pathfinder.py        (A* navigation)
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 communication/
│   │   ├── 🐍 protocol.py          (Inter-agent messaging)
│   │   └── 🐍 __init__.py
│   │
│   ├── 📁 visualization/
│   │   ├── 🐍 base.py              (Abstract visualizer)
│   │   ├── 🐍 matplotlib_viz.py    (Matplotlib rendering)
│   │   ├── 🐍 pygame_viz.py        (Pygame rendering)
│   │   └── 🐍 __init__.py
│   │
│   └── 📄 __init__.py
│
└── 📁 __pycache__/                 (Python cache)
```

---

## Configuration & Parameters

### Agent Capabilities

| Parameter              | Min | Max | Default | Notes                                                |
| ---------------------- | --- | --- | ------- | ---------------------------------------------------- |
| **Vision radius**      | 1   | 3   | 2       | Cells visible in Manhattan distance (orthogonal)     |
| **Comm radius**        | 1   | 2   | 2       | Max distance for inter-agent message exchange        |
| **Initial battery**    | —   | —   | 500     | Energy units; −1 per step; agent stops at 0         |
| **Grid size**          | —   | —   | 25×25   | Environment dimensions (fixed per instance)          |
| **Num agents**         | 1   | 10  | 5       | Team size; configurable per run                      |
| **Max ticks**          | 100 | 750 | 500     | Simulation duration limit                            |

### Warehouse Geometry

Both instances (`A.json`, `B.json`) contain:

- **4 warehouses** — rectangular regions marked `WAREHOUSE` (value 2)
- **4 entrances** — one per warehouse, marked `ENTRANCE` (value 3)
- **4 exits** — one per warehouse, marked `EXIT` (value 4)
- **Corridors** — `EMPTY` cells (value 0) connecting warehouses
- **Obstacles** — `WALL` cells (value 1) representing shelves
- **10 objects** — coordinates in separate `objects` array (not grid-embedded)

**Cell type values:**

| Value | Type      | Walkable | Role                         |
| ----- | --------- | -------- | ---------------------------- |
| 0     | EMPTY     | ✓        | Corridor, general space      |
| 1     | WALL      | ✗        | Obstacle, shelf              |
| 2     | WAREHOUSE | ✓        | Interior; delivery target    |
| 3     | ENTRANCE  | ✓        | Gateway into warehouse       |
| 4     | EXIT      | ✓        | Gateway out of warehouse     |

---

## Simulation Flow

1. **Initialization**
   - Load environment from JSON
   - Create N agents with assigned strategies
   - Reset batteries, local maps, object tracking

2. **Per-tick loop** (up to `max_ticks`)
   - Each agent perceives surroundings (vision + communication)
   - Each agent executes strategy logic (navigation, exploration)
   - Local maps (agent knowledge) are updated
   - Metrics are collected (ticks, objects delivered, energy)
   - Stop if all objects are delivered

3. **Metrics & reporting**
   - **Total ticks**: simulation duration
   - **Objects delivered**: count of objects successfully carried to warehouse
   - **Completion rate**: delivered / total objects (0–1)
   - **Average energy consumed**: sum of energy spent / num agents
   - **First pickup tick**: when first object was picked up
   - **First delivery tick**: when first object was delivered

---

## Benchmarking Modes

### Interactive Benchmarking (UI)

Run from the **🔬 Benchmark** tab in Streamlit:

1. Select parametrization mode:
   - **Random**: vary each parameter independently across a range
   - **Fixed**: lock each parameter to a single value

2. Configure ranges:
  - Strategy pool (subset of usable strategies; Ant-Colony excluded)
   - Vision radius range (1–5, or fixed)
   - Communication radius range (1–2, or fixed)

3. Generate presets:
   - Displayed: max unique combinations
   - Input: number of presets to test (sampled or exhaustive)

4. Execute & analyze:
   - Run all presets with progress bar
   - Download CSV with all results
   - View delivery curves (cumulative objects/tick)
   - Rank by efficiency (ticks, energy, completion)

---

## Exploration Strategies

### Frontier

Explores based on "frontier" cells — boundaries between known and unknown areas. Prioritizes distant frontiers and locks on selected targets to avoid oscillation. Weights frontier distance sub-linearly to prefer nearby exploration while still reaching far areas.

### Greedy

Performs warehouse-centric search: prioritizes cells closest to known warehouse locations, exploring warehouse interiors first. Simpler but may miss distributed objects.

### Sector

Divides the grid into equal sectors and assigns each agent a sector. Agents explore only their assigned area to minimize overlap and ensure uniform coverage.

### Repulsion

Agents repel each other based on proximity, creating emergent separation. No explicit coordination; behavior emerges from local repulsion forces.

### Smart Random

Enhances random walk with:
- **Information gain** — prefers cells that reveal more unknown area
- **Stale avoidance** — deprioritizes recently visited cells
- **Separation** — avoids crowding with nearby agents

### Ant-Colony

**Prototype**: Pheromone-inspired strategy where agents should lay virtual pheromones and bias motion toward low-pheromone cells. This strategy is not fully implemented yet, so it is documented for completeness only. Do not use in production runs or benchmark comparisons.

---

## Author

**Daniele Barabagallo** — [GitHub](https://github.com/danii909)

---

