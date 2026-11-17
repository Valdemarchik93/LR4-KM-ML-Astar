"""Тесты для src.dataset."""

import numpy as np

from src.dataset import build_dataset, make_patch, save_dataset, load_dataset
from src.grid import create_empty_grid


def test_make_patch_shape():
    grid = create_empty_grid((20, 20))
    patch = make_patch(grid, (10, 10), patch_size=11)
    assert patch.shape == (11, 11)


def test_make_patch_border():
    grid = create_empty_grid((20, 20))
    patch = make_patch(grid, (0, 0), patch_size=11)
    assert patch.shape == (11, 11)
    # За границей поля должны быть единицы
    assert patch[0, 0] == 1.0


def test_build_dataset_small():
    X, y = build_dataset(
        n_maps=2, m=20, n=20, n_obstacles=20,
        obs_len=(1, 2), patch_size=11,
        samples_per_map=5, seed=42,
    )
    assert X.ndim == 3
    assert X.shape[1:] == (11, 11)
    assert X.shape[0] == len(y)
    assert y.min() >= 0


def test_save_load_dataset(tmp_path):
    X = np.random.rand(10, 11, 11).astype(np.float32)
    y = np.random.rand(10).astype(np.float32)

    path = str(tmp_path / "ds.npz")
    save_dataset(path, X, y)
    X2, y2 = load_dataset(path)

    assert np.allclose(X, X2)
    assert np.allclose(y, y2)