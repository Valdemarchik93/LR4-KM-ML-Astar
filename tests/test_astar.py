"""Тесты для src.astar."""

import numpy as np

from src.astar import calc_shortest_path
from src.grid import create_empty_grid


def test_path_on_empty_grid():
    grid = create_empty_grid((10, 10))
    path = calc_shortest_path(grid, (0, 0), (9, 9))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (9, 9)
    assert len(path) - 1 == 18  # манхэттенское расстояние


def test_path_around_wall():
    grid = create_empty_grid((5, 5))
    # Стена посередине, кроме верхней клетки
    grid[1:4, 2] = 1

    path = calc_shortest_path(grid, (2, 0), (2, 4))
    assert path is not None
    assert path[0] == (2, 0)
    assert path[-1] == (2, 4)
    # Путь должен обойти стену сверху или снизу
    assert any(p[0] == 0 or p[0] == 4 for p in path)


def test_no_path():
    grid = create_empty_grid((5, 5))
    grid[:, 2] = 1  # полная стена
    path = calc_shortest_path(grid, (0, 0), (0, 4))
    assert path is None


def test_start_equals_goal():
    grid = create_empty_grid((5, 5))
    path = calc_shortest_path(grid, (2, 2), (2, 2))
    assert path == [(2, 2)]