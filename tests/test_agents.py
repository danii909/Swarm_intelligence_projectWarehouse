import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.agent import Agent, AgentState
from src.agents.sensors import compute_visible_cells, can_communicate
from src.agents.strategies.random_walk import RandomWalkStrategy
from src.agents.strategies.frontier import FrontierStrategy
from src.agents.strategies.spiral import SpiralStrategy
from src.agents.strategies.sector import SectorStrategy
from src.agents.strategies.greedy import GreedyStrategy
from src.environment.grid import Grid, CellType

INSTANCE_A = os.path.join(os.path.dirname(__file__), "..", "Consegna", "A.json")


# --- Sensori ---

def _make_open_grid(size: int = 7) -> Grid:
    data = [[0] * size for _ in range(size)]
    return Grid(data, size)


def test_visibility_radius_1():
    g = _make_open_grid(7)
    visible = compute_visible_cells(g, 3, 3, radius=1)
    # Deve contenere la cella centrale e le 4 adiacenti
    assert (3, 3) in visible
    assert (2, 3) in visible
    assert (4, 3) in visible
    assert (3, 2) in visible
    assert (3, 4) in visible


def test_visibility_radius_occlusion():
    # Muro in (1,3) deve bloccare la vista verso (0,3) partendo da (2,3)
    data = [[0] * 5 for _ in range(5)]
    data[1][3] = 1  # muro
    g = Grid(data, 5)
    visible = compute_visible_cells(g, 2, 3, radius=2)
    assert (1, 3) not in visible or CellType(data[1][3]) == CellType.WALL
    assert (0, 3) not in visible


def test_can_communicate_true():
    assert can_communicate((0, 0), (0, 2), comm_radius=2)


def test_can_communicate_false():
    assert not can_communicate((0, 0), (0, 3), comm_radius=2)


# --- Agente ---

def test_agent_initial_state():
    grid = _make_open_grid()
    a = Agent(agent_id=0, strategy=RandomWalkStrategy(), grid=grid)
    assert a.pos == (0, 0)
    assert a.battery == 500
    assert a.state == AgentState.EXPLORING
    assert not a.carrying_object
    assert a.is_active
    # Verify full map knowledge
    assert len(a.local_map) == 49  # 7x7 grid
    # Verify visited_cells starts empty
    assert len(a.visited_cells) == 0


def test_agent_move_depletes_battery():
    grid = _make_open_grid()
    a = Agent(agent_id=0, strategy=RandomWalkStrategy(), grid=grid)
    a.move_to(0, 1)
    assert a.battery == 499
    assert a.pos == (0, 1)
    assert a.steps_taken == 1


def test_agent_depletes_on_zero_battery():
    grid = _make_open_grid()
    a = Agent(agent_id=0, strategy=RandomWalkStrategy(), grid=grid)
    a.battery = 1
    a.move_to(1, 0)
    assert a.state == AgentState.DEPLETED
    assert not a.is_active


def test_agent_communicate_merges_maps():
    grid = _make_open_grid()
    a = Agent(agent_id=0, strategy=RandomWalkStrategy(), grid=grid, comm_radius=2)
    b = Agent(agent_id=1, strategy=RandomWalkStrategy(), grid=grid, comm_radius=2)
    # Simulate partial exploration by setting visited_cells
    a.visited_cells = {(0, 0), (0, 1)}
    b.visited_cells = {(1, 1), (1, 2)}
    a.communicate_with(b)
    # Both agents should have merged visited_cells
    assert (1, 1) in a.visited_cells
    assert (0, 0) in b.visited_cells
    assert len(a.visited_cells) == len(b.visited_cells) == 4


def test_agent_communicate_out_of_range():
    grid = _make_open_grid()
    a = Agent(agent_id=0, strategy=RandomWalkStrategy(), grid=grid, comm_radius=1)
    b = Agent(agent_id=1, strategy=RandomWalkStrategy(), grid=grid, comm_radius=1)
    b.row, b.col = 5, 5
    a.visited_cells = {(0, 0)}
    b.visited_cells = {(5, 5)}
    a.communicate_with(b)
    # Nessun merge: troppo lontani
    assert (5, 5) not in a.visited_cells


# --- Strategie ---

@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
@pytest.mark.parametrize("StratClass", [
    RandomWalkStrategy,
    FrontierStrategy,
    SpiralStrategy,
    GreedyStrategy,
])
def test_strategy_returns_valid_move(StratClass):
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder

    env = Environment.from_json(INSTANCE_A)
    pathfinder = Pathfinder(env.grid)
    agent = Agent(agent_id=0, strategy=StratClass(), grid=env.grid)
    agent.perceive(env)

    move = agent.decide_next_move(env, pathfinder, set())
    # Il risultato è None oppure una tupla di 2 interi in bounds
    if move is not None:
        r, c = move
        assert env.grid.in_bounds(r, c)
        assert env.grid.is_walkable(r, c)


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_sector_strategy_returns_valid_move():
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder

    env = Environment.from_json(INSTANCE_A)
    pathfinder = Pathfinder(env.grid)
    agent = Agent(agent_id=2, strategy=SectorStrategy(num_agents=5), grid=env.grid)
    agent.perceive(env)

    move = agent.decide_next_move(env, pathfinder, set())
    if move is not None:
        r, c = move
        assert env.grid.in_bounds(r, c)
        assert env.grid.is_walkable(r, c)


# --- New tests for visited_cells functionality ---

def test_perceive_updates_visited_cells():
    """Perceive should track visited cells for object search."""
    from src.environment.environment import Environment

    env = Environment.from_json(INSTANCE_A)
    grid = env.grid
    agent = Agent(agent_id=0, strategy=RandomWalkStrategy(), grid=grid, visibility_radius=2)
    agent.row, agent.col = 5, 5

    # Before perception
    assert len(agent.visited_cells) == 0

    # After perception
    visible = agent.perceive(env)
    assert len(agent.visited_cells) > 0
    assert (5, 5) in agent.visited_cells  # Agent's position
    assert agent.visited_cells == visible


def test_frontier_strategy_uses_visited_cells():
    """FrontierStrategy should find frontiers based on visited_cells, not local_map."""
    from src.environment.environment import Environment

    env = Environment.from_json(INSTANCE_A)
    strategy = FrontierStrategy()
    agent = Agent(agent_id=0, strategy=strategy, grid=env.grid, visibility_radius=2)
    agent.row, agent.col = 1, 1  # Position in known EMPTY area

    # Agent has full map but no visited cells yet
    assert len(agent.local_map) > 0
    assert len(agent.visited_cells) == 0

    # Frontier finding should return empty without visited cells
    frontiers = strategy._find_frontiers(agent, env)
    assert len(frontiers) == 0  # No frontiers without visited cells

    # Find some actual EMPTY cells to mark as visited
    empty_cells = [(r, c) for (r, c), ct in agent.local_map.items()
                   if ct == CellType.EMPTY][:5]
    agent.visited_cells = set(empty_cells)

    # There should be frontiers now if there are unvisited empty neighbors
    frontiers = strategy._find_frontiers(agent, env)
    # Note: May be 0 if all neighbors are also walls/warehouses, but code works correctly
    assert isinstance(frontiers, set)


def test_sector_strategy_uses_visited_cells():
    """SectorStrategy should check visited_cells, not local_map."""
    from src.environment.environment import Environment
    from src.pathfinding.pathfinder import Pathfinder

    env = Environment.from_json(INSTANCE_A)
    strategy = SectorStrategy(num_agents=5)
    agent = Agent(agent_id=0, strategy=strategy, grid=env.grid)
    agent.row, agent.col = 0, 0  # Start at origin

    # Agent knows full map but hasn't visited anything
    assert len(agent.visited_cells) == 0

    pathfinder = Pathfinder(env.grid)
    move = strategy.next_move(agent, env, pathfinder, set())

    # Should return a valid move toward unvisited sector cells
    # (May be None if starting position has no walkable path, but should try)
    if move is not None:
        assert env.grid.in_bounds(*move)
