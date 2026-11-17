"""Генерация клеточного поля с препятствиями и точек A/B."""

from typing import Tuple

import numpy as np
from numpy.random import Generator, PCG64


def make_rng(seed: int | None = None) -> Generator:
    """Создать генератор случайных чисел.

    Если seed=None, используется случайный seed на основе текущего времени.
    """
    if seed is None:
        import time
        seed = int(time.time() * 1000) % 2**32
    return Generator(PCG64(seed=seed))


def create_empty_grid(size: Tuple[int, int]) -> np.ndarray:
    """Создать пустую сетку размера size = (m, n)."""
    return np.zeros(size, dtype=int)


def fill_obstacles(
    grid: np.ndarray,
    n_obstacles: int,
    min_len: int,
    max_len: int,
    rng: Generator,
) -> int:
    """Заполнить поле grid препятствиями.

    Препятствие имеет случайную длину от min_len до max_len
    и единичную ширину (горизонтальное или вертикальное).

    Возвращает фактическое число созданных препятствий.
    """
    m, n = grid.shape
    obstacles_created = 0
    max_attempts = n_obstacles * 100

    for _ in range(max_attempts):
        if obstacles_created >= n_obstacles:
            break

        orientation = rng.integers(0, 2)  # 0 — горизонтальное, 1 — вертикальное
        length = rng.integers(min_len, max_len + 1)

        if orientation == 0:  # горизонтальное
            x = rng.integers(0, n - length + 1)
            y = rng.integers(0, m)
            if np.all(grid[y, x:x + length] == 0):
                grid[y, x:x + length] = 1
                obstacles_created += 1
        else:  # вертикальное
            x = rng.integers(0, n)
            y = rng.integers(0, m - length + 1)
            if np.all(grid[y:y + length, x] == 0):
                grid[y:y + length, x] = 1
                obstacles_created += 1

    return obstacles_created


def gen_point_in_left_half(grid: np.ndarray, rng: Generator) -> Tuple[int, int]:
    """Сгенерировать случайную свободную точку в левой половине поля."""
    m, n = grid.shape
    left_boundary = n // 2

    free_cells = [
        (y, x)
        for y in range(m)
        for x in range(left_boundary)
        if grid[y, x] == 0
    ]

    if not free_cells:
        free_cells = [tuple(c) for c in np.argwhere(grid == 0)]

    if not free_cells:
        raise ValueError("Нет свободных клеток для генерации точки старта!")

    idx = rng.integers(0, len(free_cells))
    return free_cells[idx]


def gen_point_in_right_half(grid: np.ndarray, rng: Generator) -> Tuple[int, int]:
    """Сгенерировать случайную свободную точку в правой половине поля."""
    m, n = grid.shape
    right_boundary = n // 2

    free_cells = [
        (y, x)
        for y in range(m)
        for x in range(right_boundary, n)
        if grid[y, x] == 0
    ]

    if not free_cells:
        free_cells = [tuple(c) for c in np.argwhere(grid == 0)]

    if not free_cells:
        raise ValueError("Нет свободных клеток для генерации точки цели!")

    idx = rng.integers(0, len(free_cells))
    return free_cells[idx]


def gen_start_and_goal(
    grid: np.ndarray,
    rng: Generator,
    min_distance: int = 10,
    max_attempts: int = 1000,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Сгенерировать стартовую точку A в левой половине и цель B в правой.

    Гарантирует, что манхэттенское расстояние между A и B не меньше min_distance.
    """
    m, n = grid.shape

    for _ in range(max_attempts):
        start = gen_point_in_left_half(grid, rng)
        goal = gen_point_in_right_half(grid, rng)

        if start == goal:
            continue

        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if distance >= min_distance:
            return start, goal

    # Если не удалось — возвращаем любые корректные точки
    print(
        f"Предупреждение: не удалось найти точки на расстоянии {min_distance}. "
        "Использую ближайшие доступные."
    )
    start = gen_point_in_left_half(grid, rng)
    goal = gen_point_in_right_half(grid, rng)

    if start == goal:
        free_cells = np.argwhere(grid == 0)
        for cell in free_cells:
            candidate = tuple(cell)
            if candidate != start and candidate[1] >= n // 2:
                return start, candidate

    return start, goal