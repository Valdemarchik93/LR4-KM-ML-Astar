"""Обучение нейросетевой эвристики."""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import load_dataset
from src.model import HeuristicCNN


def train_model(
    dataset_path: str = "data/dataset.npz",
    model_path: str = "models/cnn_heuristic.pt",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, offsets, y = load_dataset(dataset_path)
    patch_size = X.shape[1]

    y_mean = float(y.mean())
    y_std = float(y.std() + 1e-8)
    y_norm = (y - y_mean) / y_std

    n_samples = len(X)
    n_val = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.from_numpy(X[train_idx]).unsqueeze(1)
    off_train = torch.from_numpy(offsets[train_idx])
    y_train = torch.from_numpy(y_norm[train_idx])

    X_val = torch.from_numpy(X[val_idx]).unsqueeze(1)
    off_val = torch.from_numpy(offsets[val_idx])
    y_val = torch.from_numpy(y_norm[val_idx])

    train_loader = DataLoader(
        TensorDataset(X_train, off_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, off_val, y_val),
        batch_size=batch_size, shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeuristicCNN(patch_size=patch_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, ob, yb in train_loader:
            xb, ob, yb = xb.to(device), ob.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, ob)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for xb, ob, yb in val_loader:
                xb, ob, yb = xb.to(device), ob.to(device), yb.to(device)
                pred = model(xb, ob)
                val_loss += criterion(pred, yb).item() * xb.size(0)

                pred_steps = pred * y_std + y_mean
                true_steps = yb * y_std + y_mean
                val_mae += torch.abs(pred_steps - true_steps).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_MAE={val_mae:.2f} шагов")

        # Сохраняем чекпоинт каждые 5 эпох
        if epoch % 5 == 0:
            ckpt_dir = os.path.join(os.path.dirname(model_path), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "patch_size": patch_size,
                "y_mean": y_mean,
                "y_std": y_std,
                "m": 80,
                "n": 70,
                "epoch": epoch,
            }, ckpt_path)
            print(f"[train] Чекпоинт сохранён: {ckpt_path}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "patch_size": patch_size,
        "y_mean": y_mean,
        "y_std": y_std,
        "m": 80,  # нужно для нормализации offsets в astar_nn
        "n": 70,
    }, model_path)
    print(f"[train] Модель сохранена: {model_path}")
    return history