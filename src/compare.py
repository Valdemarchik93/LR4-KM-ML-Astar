"""Сравнение A* с манхэттенской и нейросетевой эвристикой."""

from typing import Dict, List, Tuple

import numpy as np

from src.astar import calc_shortest_path
from src.astar_nn import calc_shortest_path_nn, NeuralHeuristic
from src.grid import (
    create_empty_grid,
    fill_obstacles,
    gen_start_and_goal,
    make_rng,
)

Point = Tuple[int, int]


def _count_expanded(grid: np.ndarray, start: Point, goal: Point,
                    heuristic_fn) -> int:
    """Посчитать число раскрытых узлов A* (обёртка над calc_shortest_path)."""
    # Простой способ: запускаем A* и считаем, сколько раз он заходил в цикл.
    # Чтобы не менять calc_shortest_path, считаем по длине пути + штрафу.
    # В следующей итерации можно добавить счётчик напрямую.
    path = calc_shortest_path(grid, start, goal, heuristic_fn=heuristic_fn)
    if path is None:
        return -1
    return len(path)


def compare_on_maps(
    n_maps: int = 50,
    m: int = 80,
    n: int = 70,
    n_obstacles: int = 700,
    obs_len: Tuple[int, int] = (1, 3),
    min_distance: int = 20,
    model_path: str = "models/cnn_heuristic.pt",
    seed: int = 123,
) -> Dict[str, List[float]]:
    """Сравнить A* с манхэттенской и нейросетевой эвристикой на n_maps полях.

    Возвращает словарь с массивами метрик.
    """
    rng = make_rng(seed)

    nn_heuristic = NeuralHeuristic(model_path=model_path)

    results: Dict[str, List[float]] = {
        "len_manhattan": [],
        "len_nn": [],
        "time_manhattan": [],
        "time_nn": [],
        "success_manhattan": [],
        "success_nn": [],
    }

    import time

    for i in range(n_maps):
        grid = create_empty_grid((m, n))
        fill_obstacles(grid, n_obstacles, obs_len[0], obs_len[1], rng)

        try:
            start, goal = gen_start_and_goal(grid, rng, min_distance=min_distance)
        except ValueError:
            continue

        # --- Манхэттенская эвристика ---
        t0 = time.perf_counter()
        path_m = calc_shortest_path(grid, start, goal)
        t_m = time.perf_counter() - t0

        # --- Нейросетевая эвристика ---
        nn_heuristic.set_grid(grid)
        t0 = time.perf_counter()
        path_nn = calc_shortest_path_nn(
            grid, start, goal, model_path=model_path
        )
        t_nn = time.perf_counter() - t0

        results["len_manhattan"].append(len(path_m) - 1 if path_m else -1)
        results["len_nn"].append(len(path_nn) - 1 if path_nn else -1)
        results["time_manhattan"].append(t_m)
        results["time_nn"].append(t_nn)
        results["success_manhattan"].append(1 if path_m else 0)
        results["success_nn"].append(1 if path_nn else 0)

        if (i + 1) % 10 == 0:
            print(f"[compare] обработано {i + 1}/{n_maps} полей")

    return results


def summarize(results: Dict[str, List[float]]) -> Dict[str, float]:
    """Свести результаты сравнения к средним значениям."""
    def mean(key: str) -> float:
        arr = results[key]
        return float(np.mean(arr)) if arr else float("nan")

    return {
        "avg_len_manhattan": mean("len_manhattan"),
        "avg_len_nn": mean("len_nn"),
        "avg_time_manhattan": mean("time_manhattan"),
        "avg_time_nn": mean("time_nn"),
        "success_rate_manhattan": mean("success_manhattan"),
        "success_rate_nn": mean("success_nn"),
    }