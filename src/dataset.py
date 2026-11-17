"""Сбор датасета для обучения нейросетевой эвристики."""

from typing import List, Tuple

import numpy as np

from src.bfs import INF, bfs_distances
from src.grid import (
    create_empty_grid,
    fill_obstacles,
    gen_start_and_goal,
    make_rng,
)

Point = Tuple[int, int]


def make_patch(
    grid: np.ndarray,
    center: Point,
    patch_size: int,
) -> np.ndarray:
    """Вырезать патч patch_size × patch_size вокруг клетки center.

    Клетки за границей поля заполняются 1 (как препятствия),
    чтобы нейросеть видела край поля как стену.
    """
    m, n = grid.shape
    half = patch_size // 2
    cy, cx = center

    patch = np.ones((patch_size, patch_size), dtype=np.float32)

    for i in range(patch_size):
        for j in range(patch_size):
            y = cy - half + i
            x = cx - half + j
            if 0 <= y < m and 0 <= x < n:
                patch[i, j] = float(grid[y, x])

    return patch


def build_dataset(
    n_maps: int = 300,
    m: int = 80,
    n: int = 70,
    n_obstacles: int = 700,
    obs_len: Tuple[int, int] = (1, 3),
    patch_size: int = 11,
    min_distance: int = 20,
    samples_per_map: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сгенерировать датасет: (патч, смещение до цели) → расстояние.

    Возвращает:
        X — патчи (N, patch_size, patch_size);
        offsets — смещения (N, 2): (dy, dx) от клетки до цели;
        y — расстояния до цели (N,).
    """
    rng = make_rng(seed)

    X_list = []
    offsets_list = []
    y_list = []

    for _ in range(n_maps):
        grid = create_empty_grid((m, n))
        fill_obstacles(grid, n_obstacles, obs_len[0], obs_len[1], rng)

        try:
            _, goal = gen_start_and_goal(grid, rng, min_distance=min_distance)
        except ValueError:
            continue

        distances = bfs_distances(grid, goal)
        free_cells = np.argwhere((grid == 0) & (distances != INF))
        if len(free_cells) == 0:
            continue

        k = min(samples_per_map, len(free_cells))
        idxs = rng.choice(len(free_cells), size=k, replace=False)

        for idx in idxs:
            y, x = free_cells[idx]
            patch = make_patch(grid, (int(y), int(x)), patch_size)
            X_list.append(patch)
            # Смещение от клетки до цели
            dy = float(goal[0] - y) / m
            dx = float(goal[1] - x) / n
            offsets_list.append([dy, dx])
            y_list.append(float(distances[y, x]))

    X = np.stack(X_list).astype(np.float32)
    offsets = np.array(offsets_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, offsets, y


def save_dataset(path: str, X: np.ndarray, offsets: np.ndarray, y: np.ndarray) -> None:
    np.savez_compressed(path, X=X, offsets=offsets, y=y)


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["offsets"], data["y"]