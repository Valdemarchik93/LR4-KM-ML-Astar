"""BFS-разметка расстояний до цели на клеточном поле."""

from collections import deque
from typing import Tuple

import numpy as np

Point = Tuple[int, int]

INF = np.iinfo(np.int32).max


def bfs_distances(grid: np.ndarray, goal: Point) -> np.ndarray:
    """Вычислить кратчайшие расстояния от каждой свободной клетки до goal.

    Возвращает матрицу distances того же размера, что и grid:
    - для свободных клеток — число шагов до цели;
    - для препятствий и недостижимых клеток — INF.
    """
    m, n = grid.shape
    distances = np.full((m, n), INF, dtype=np.int32)

    gy, gx = goal
    if grid[gy, gx] != 0:
        return distances

    distances[gy, gx] = 0
    queue = deque([goal])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()
        d = distances[y, x]

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < m and 0 <= nx < n and grid[ny, nx] == 0:
                if distances[ny, nx] == INF:
                    distances[ny, nx] = d + 1
                    queue.append((ny, nx))

    return distances


def is_reachable(distances: np.ndarray, point: Point) -> bool:
    """Проверить, достижима ли клетка до цели по BFS-разметке."""
    y, x = point
    return distances[y, x] != INF