"""Тесты для src.bfs."""

import numpy as np

from src.bfs import INF, bfs_distances, is_reachable
from src.grid import create_empty_grid


def test_bfs_on_empty_grid():
    grid = create_empty_grid((5, 5))
    dist = bfs_distances(grid, (0, 0))

    assert dist[0, 0] == 0
    assert dist[4, 4] == 8
    assert dist[0, 4] == 4


def test_bfs_with_wall():
    grid = create_empty_grid((5, 5))
    grid[:, 2] = 1  # полная стена

    dist = bfs_distances(grid, (0, 0))
    assert dist[0, 0] == 0
    assert dist[0, 4] == INF
    assert not is_reachable(dist, (0, 4))


def test_bfs_goal_on_obstacle():
    grid = create_empty_grid((5, 5))
    grid[0, 0] = 1
    dist = bfs_distances(grid, (0, 0))
    assert np.all(dist == INF)