import pytest
import json
import os
import sys

# Aggiunge la root del progetto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.grid import Grid, CellType, WALKABLE
from src.environment.environment import Environment, Warehouse

INSTANCE_A = os.path.join(os.path.dirname(__file__), "..", "Consegna", "A.json")


# --- Grid tests ---

def test_cell_type_values():
    assert CellType.EMPTY == 0
    assert CellType.WALL == 1
    assert CellType.WAREHOUSE == 2
    assert CellType.ENTRANCE == 3
    assert CellType.EXIT == 4


def test_grid_in_bounds():
    data = [[0] * 5 for _ in range(5)]
    g = Grid(data, 5)
    assert g.in_bounds(0, 0)
    assert g.in_bounds(4, 4)
    assert not g.in_bounds(-1, 0)
    assert not g.in_bounds(5, 0)
    assert not g.in_bounds(0, 5)


def test_grid_is_walkable_empty():
    data = [[0] * 3 for _ in range(3)]
    g = Grid(data, 3)
    assert g.is_walkable(0, 0)


def test_grid_is_walkable_wall():
    data = [[1, 0], [0, 0]]
    g = Grid(data, 2)
    assert not g.is_walkable(0, 0)
    assert g.is_walkable(0, 1)


def test_grid_walkable_neighbors():
    # Griglia 3x3 tutta vuota
    data = [[0] * 3 for _ in range(3)]
    g = Grid(data, 3)
    neighbors = g.walkable_neighbors(1, 1)
    assert set(neighbors) == {(0, 1), (2, 1), (1, 0), (1, 2)}


def test_grid_walkable_neighbors_corner():
    data = [[0] * 3 for _ in range(3)]
    g = Grid(data, 3)
    neighbors = g.walkable_neighbors(0, 0)
    assert set(neighbors) == {(1, 0), (0, 1)}


# --- Environment tests ---

@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_environment_load_from_json():
    env = Environment.from_json(INSTANCE_A)
    assert env.grid.size == 25
    assert len(env.warehouses) == 4
    assert env.total_objects == 10
    assert env.delivered == 0
    assert env.remaining_objects == 10


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_environment_pickup_object():
    env = Environment.from_json(INSTANCE_A)
    first_obj = next(iter(env._objects))
    assert env.object_at(*first_obj)
    result = env.pickup_object(*first_obj)
    assert result
    assert not env.object_at(*first_obj)
    assert env.remaining_objects == 9


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_environment_deliver_object():
    env = Environment.from_json(INSTANCE_A)
    env.deliver_object((3, 11))
    assert env.delivered == 1


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_environment_sense_objects():
    env = Environment.from_json(INSTANCE_A)
    first_obj = next(iter(env._objects))
    detected = env.sense_objects({first_obj, (0, 0), (1, 1)})
    assert first_obj in detected


@pytest.mark.skipif(not os.path.exists(INSTANCE_A), reason="Consegna/A.json not found")
def test_environment_nearest_warehouse_entrance():
    env = Environment.from_json(INSTANCE_A)
    # Agente vicino all'ingresso del magazzino 0 in [3,11]
    entrance = env.nearest_warehouse_entrance(3, 10)
    assert entrance is not None
