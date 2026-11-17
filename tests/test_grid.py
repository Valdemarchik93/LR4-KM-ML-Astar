"""Тесты для src.grid."""

import numpy as np
import pytest

from src.grid import (
    create_empty_grid,
    fill_obstacles,
    gen_point_in_left_half,
    gen_point_in_right_half,
    gen_start_and_goal,
    make_rng,
)


def test_create_empty_grid():
    grid = create_empty_grid((10, 20))
    assert grid.shape == (10, 20)
    assert np.all(grid == 0)


def test_fill_obstacles():
    rng = make_rng(42)
    grid = create_empty_grid((20, 20))
    created = fill_obstacles(grid, 10, 1, 3, rng)

    # Создано хотя бы одно препятствие
    assert created > 0
    # Число занятых клеток >= числа препятствий (каждое длиной >= 1)
    assert grid.sum() >= created
    # И не больше, чем created * max_len
    assert grid.sum() <= created * 3


def test_points_in_correct_halves():
    rng = make_rng(42)
    grid = create_empty_grid((50, 60))
    fill_obstacles(grid, 100, 1, 3, rng)

    start = gen_point_in_left_half(grid, rng)
    goal = gen_point_in_right_half(grid, rng)

    assert start[1] < 30
    assert goal[1] >= 30


def test_start_and_goal_min_distance():
    rng = make_rng(42)
    grid = create_empty_grid((50, 60))
    fill_obstacles(grid, 100, 1, 3, rng)

    start, goal = gen_start_and_goal(grid, rng, min_distance=10)
    dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    assert dist >= 10


def test_start_and_goal_on_free_cells():
    rng = make_rng(42)
    grid = create_empty_grid((50, 60))
    fill_obstacles(grid, 100, 1, 3, rng)

    start, goal = gen_start_and_goal(grid, rng, min_distance=10)
    assert grid[start] == 0
    assert grid[goal] == 0