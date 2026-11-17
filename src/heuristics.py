"""Эвристические функции для алгоритма A*."""

from typing import Tuple

Point = Tuple[int, int]


def manhattan(s: Point, d: Point) -> float:
    """Манхэттенское расстояние между двумя клетками."""
    return abs(s[0] - d[0]) + abs(s[1] - d[1])


def heuristic(s: Point, d: Point) -> float:
    """Базовая эвристика: манхэттенское расстояние."""
    return manhattan(s, d)


def calc_cost(s: Point, d: Point, c: float, w: float = 1.0) -> float:
    """Рассчитать стоимость: g + w * h.

    s — текущая клетка, d — цель, c — стоимость до текущей клетки,
    w — вес эвристики.
    """
    return c + 1 + w * heuristic(s, d)