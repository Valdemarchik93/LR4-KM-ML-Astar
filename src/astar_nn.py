"""A* с нейросетевой эвристикой."""

from typing import List, Optional, Tuple

import numpy as np
import torch

from src.astar import calc_shortest_path
from src.dataset import make_patch
from src.model import HeuristicCNN

Point = Tuple[int, int]


class NeuralHeuristic:
    def __init__(self, model_path="models/cnn_heuristic.pt", device=None):
        self.device = torch.device(device if device else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(model_path, map_location=self.device)
        self.patch_size = ckpt["patch_size"]
        self.y_mean = ckpt["y_mean"]
        self.y_std = ckpt["y_std"]
        self.m = ckpt.get("m", 80)
        self.n = ckpt.get("n", 70)

        self.model = HeuristicCNN(patch_size=self.patch_size).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self._cache = {}
        self._grid = None

    def set_grid(self, grid):
        self._grid = grid
        self._cache.clear()

    def __call__(self, s, d):
        if self._grid is None:
            raise RuntimeError("NeuralHeuristic: сначала вызовите set_grid(grid)")

        key = (s, d)
        if key in self._cache:
            return self._cache[key]

        patch = make_patch(self._grid, s, self.patch_size)
        x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)

        dy = float(d[0] - s[0]) / self.m
        dx = float(d[1] - s[1]) / self.n
        off = torch.tensor([[dy, dx]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(x, off).item()

        pred_steps = max(0.0, pred_norm * self.y_std + self.y_mean)
        self._cache[key] = pred_steps
        return pred_steps


def calc_shortest_path_nn(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    model_path: str = "models/cnn_heuristic.pt",
    w: float = 1.0,
) -> Optional[List[Point]]:
    """Найти путь A* с нейросетевой эвристикой."""
    heuristic = NeuralHeuristic(model_path=model_path)
    heuristic.set_grid(grid)
    return calc_shortest_path(grid, start, goal, w=w, heuristic_fn=heuristic)