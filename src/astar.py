"""Алгоритм A* для поиска пути на клеточном поле."""

import heapq
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.heuristics import heuristic

Point = Tuple[int, int]
Heuristic = Callable[[Point, Point], float]


class Node:
    """Узел в алгоритме A*."""

    def __init__(self, pos: Point, g: float, h: float, parent: "Node | None" = None):
        self.pos = pos
        self.g = g          # стоимость пути от старта
        self.h = h          # эвристическая оценка до цели
        self.f = g + h      # общая оценка
        self.parent = parent

    def __lt__(self, other: "Node") -> bool:
        return self.f < other.f


def _neighbors(node: Node, grid: np.ndarray) -> List[Point]:
    """Свободные соседние клетки (вверх, вниз, влево, вправо)."""
    m, n = grid.shape
    y, x = node.pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    result = []
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < m and 0 <= nx < n and grid[ny, nx] == 0:
            result.append((ny, nx))
    return result


def calc_shortest_path(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    w: float = 1.0,
    heuristic_fn: Heuristic = heuristic,
) -> Optional[List[Point]]:
    """Найти кратчайший путь от start до goal алгоритмом A*.

    Параметры:
        grid — клеточное поле (0 — свободно, 1 — препятствие);
        start — начальная клетка;
        goal — целевая клетка;
        w — вес эвристики;
        heuristic_fn — функция эвристики (по умолчанию манхэттенская).

    Возвращает список клеток пути от start до goal или None, если пути нет.
    """
    open_list: List[Node] = []
    closed_set: set[Point] = set()

    start_node = Node(start, 0, w * heuristic_fn(start, goal))
    heapq.heappush(open_list, start_node)
    node_dict: dict[Point, Node] = {start: start_node}

    while open_list:
        current = heapq.heappop(open_list)

        if current.pos == goal:
            path = []
            node: Optional[Node] = current
            while node is not None:
                path.append(node.pos)
                node = node.parent
            return path[::-1]

        closed_set.add(current.pos)

        for neighbor_pos in _neighbors(current, grid):
            if neighbor_pos in closed_set:
                continue

            g_new = current.g + 1
            h_new = w * heuristic_fn(neighbor_pos, goal)

            if neighbor_pos in node_dict:
                existing = node_dict[neighbor_pos]
                if g_new < existing.g:
                    existing.g = g_new
                    existing.h = h_new
                    existing.f = g_new + h_new
                    existing.parent = current
                    heapq.heapify(open_list)
            else:
                new_node = Node(neighbor_pos, g_new, h_new, current)
                node_dict[neighbor_pos] = new_node
                heapq.heappush(open_list, new_node)

    return None