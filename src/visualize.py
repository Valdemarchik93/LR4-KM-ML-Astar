"""Визуализация: карты, пути, графики обучения, сравнение."""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.grid import create_empty_grid, fill_obstacles, gen_start_and_goal, make_rng

Point = Tuple[int, int]


# ---------------------------------------------------------------------------
# Карта поля + путь
# ---------------------------------------------------------------------------

def plot_path_on_grid(
    grid: np.ndarray,
    start: Optional[Point] = None,
    goal: Optional[Point] = None,
    path: Optional[List[Point]] = None,
    title: str = "A* path",
    path_out: str = "results/maps/path.png",
) -> None:
    """Нарисовать поле, точки A/B и найденный путь."""
    m, n = grid.shape
    fig, ax = plt.subplots(figsize=(12, 10))

    # Препятствия
    obstacle_positions = np.argwhere(grid == 1)
    for y, x in obstacle_positions:
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            linewidth=0.3, edgecolor="black", facecolor="black", alpha=0.7,
        )
        ax.add_patch(rect)

    # Граница левой/правой половины
    mid_x = n // 2 - 0.5
    ax.axvline(x=mid_x, color="red", linestyle="--", linewidth=1.5, alpha=0.5)

    # Путь
    if path:
        path_y, path_x = zip(*path)
        for (y, x) in path:
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                linewidth=0.5, edgecolor="blue", facecolor="lightblue", alpha=0.6,
            )
            ax.add_patch(rect)
        ax.plot(
            [x + 0.5 for x in path_x],
            [y + 0.5 for y in path_y],
            c="darkblue", linewidth=1.5, alpha=0.8,
        )

    # Точки A и B
    if start is not None:
        sy, sx = start
        ax.add_patch(patches.Rectangle(
            (sx - 0.5, sy - 0.5), 1, 1,
            linewidth=2, edgecolor="darkgreen", facecolor="lime", alpha=0.9,
        ))
        ax.text(sx, sy, "A", fontsize=10, weight="bold",
                ha="center", va="center", color="black")

    if goal is not None:
        gy, gx = goal
        ax.add_patch(patches.Rectangle(
            (gx - 0.5, gy - 0.5), 1, 1,
            linewidth=2, edgecolor="darkred", facecolor="red", alpha=0.9,
        ))
        ax.text(gx, gy, "B", fontsize=10, weight="bold",
                ha="center", va="center", color="white")

    # Оформление
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, weight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Сохранено: {path_out}")


# ---------------------------------------------------------------------------
# Графики обучения
# ---------------------------------------------------------------------------

def plot_training_history(history: Dict[str, List[float]],
                          path_out: str = "results/training/history.png") -> None:
    """Нарисовать графики loss и MAE по эпохам."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("MSE (нормированный)")
    axes[0].set_title("Loss по эпохам")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_mae"], color="orange", label="val_MAE")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("MAE, шагов")
    axes[1].set_title("Ошибка предсказания (шаги)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Сохранено: {path_out}")


# ---------------------------------------------------------------------------
# Сравнение A* vs A*+NN
# ---------------------------------------------------------------------------

def plot_comparison(results: Dict[str, List[float]],
                    path_out: str = "results/comparison/compare.png") -> None:
    """Построить сравнительные графики: длина пути, время, успешность."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Длина пути
    axes[0].boxplot(
    [results["len_manhattan"], results["len_nn"]],
    tick_labels=["Манхэттен", "NN"],
    )
    axes[0].set_title("Длина пути")
    axes[0].set_ylabel("шагов")
    axes[0].grid(True, alpha=0.3)

    # Время
    axes[1].boxplot(
    [results["time_manhattan"], results["time_nn"]],
    tick_labels=["Манхэттен", "NN"],
    )
    axes[1].set_title("Время работы A*")
    axes[1].set_ylabel("секунд")
    axes[1].grid(True, alpha=0.3)

    # Успешность
    axes[2].bar(
        ["Манхэттен", "NN"],
        [
            np.mean(results["success_manhattan"]) * 100,
            np.mean(results["success_nn"]) * 100,
        ],
        color=["steelblue", "orange"],
    )
    axes[2].set_title("Успешность")
    axes[2].set_ylabel("% найденных путей")
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Сохранено: {path_out}")


def plot_two_paths_side_by_side(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    path_manhattan: Optional[List[Point]],
    path_nn: Optional[List[Point]],
    path_out: str = "results/comparison/two_paths.png",
) -> None:
    """Нарисовать два пути на одном поле: манхэттен и NN."""
    m, n = grid.shape
    fig, ax = plt.subplots(figsize=(12, 10))

    obstacle_positions = np.argwhere(grid == 1)
    for y, x in obstacle_positions:
        ax.add_patch(patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            linewidth=0.3, edgecolor="black", facecolor="black", alpha=0.6,
        ))

    if path_manhattan:
        ys, xs = zip(*path_manhattan)
        ax.plot([x + 0.5 for x in xs], [y + 0.5 for y in ys],
                c="blue", linewidth=2, alpha=0.9, label="Манхэттен")

    if path_nn:
        ys, xs = zip(*path_nn)
        ax.plot([x + 0.5 for x in xs], [y + 0.5 for y in ys],
                c="orange", linewidth=2, alpha=0.9, linestyle="--", label="NN")

    sy, sx = start
    ax.add_patch(patches.Rectangle(
        (sx - 0.5, sy - 0.5), 1, 1,
        linewidth=2, edgecolor="darkgreen", facecolor="lime", alpha=0.9,
    ))
    gy, gx = goal
    ax.add_patch(patches.Rectangle(
        (gx - 0.5, gy - 0.5), 1, 1,
        linewidth=2, edgecolor="darkred", facecolor="red", alpha=0.9,
    ))

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Сравнение путей: манхэттен vs NN", fontsize=14, weight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Сохранено: {path_out}")


# ---------------------------------------------------------------------------
# Пример поля для README
# ---------------------------------------------------------------------------

def plot_example_map(path_out: str = "results/maps/example_map.png",
                     seed: int = 42) -> None:
    """Сгенерировать пример поля с препятствиями и точками A/B."""
    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)
    plot_path_on_grid(grid, start, goal, None,
                      title="Пример поля с препятствиями",
                      path_out=path_out)

# ---------------------------------------------------------------------------
# GIF-анимации
# ---------------------------------------------------------------------------

def create_astar_animation(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    path: Optional[List[Point]],
    title: str = "A* поиск пути",
    path_out: str = "results/maps/astar_animation.gif",
    fps: int = 10,
    frame_step: int = 3,
) -> None:
    """GIF-анимация: как A* шаг за шагом идёт по пути."""
    import io
    from PIL import Image

    if path is None:
        print("[visualize] Путь не найден, анимация не создана")
        return

    m, n = grid.shape
    frames = []

    for i in range(0, len(path), frame_step):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Препятствия
        for y, x in np.argwhere(grid == 1):
            ax.add_patch(patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                linewidth=0.3, edgecolor="black", facecolor="black", alpha=0.6,
            ))

        # Уже пройденный путь
        partial = path[:i + 1]
        if partial:
            ys, xs = zip(*partial)
            ax.plot([x + 0.5 for x in xs], [y + 0.5 for y in ys],
                    c="darkblue", linewidth=2, alpha=0.9)

        # Текущая клетка
        cy, cx = path[i]
        ax.add_patch(patches.Rectangle(
            (cx - 0.5, cy - 0.5), 1, 1,
            linewidth=2, edgecolor="yellow", facecolor="gold", alpha=0.9,
        ))

        # Старт и цель
        sy, sx = start
        ax.add_patch(patches.Rectangle(
            (sx - 0.5, sy - 0.5), 1, 1,
            linewidth=2, edgecolor="darkgreen", facecolor="lime", alpha=0.9,
        ))
        gy, gx = goal
        ax.add_patch(patches.Rectangle(
            (gx - 0.5, gy - 0.5), 1, 1,
            linewidth=2, edgecolor="darkred", facecolor="red", alpha=0.9,
        ))

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(m - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title} | шаг {i}/{len(path)-1}", fontsize=12, weight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()
        plt.close(fig)

    if frames:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        frames[0].save(path_out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
        print(f"[visualize] GIF сохранён: {path_out}")


def create_comparison_animation(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    path_manhattan: Optional[List[Point]],
    path_nn: Optional[List[Point]],
    path_out: str = "results/comparison/comparison_animation.gif",
    fps: int = 8,
    frame_step: int = 4,
) -> None:
    """GIF: два пути идут одновременно (манхэттен и NN)."""
    import io
    from PIL import Image

    if path_manhattan is None or path_nn is None:
        print("[visualize] Один из путей не найден, анимация не создана")
        return

    m, n = grid.shape
    max_len = max(len(path_manhattan), len(path_nn))
    frames = []

    for i in range(0, max_len, frame_step):
        fig, ax = plt.subplots(figsize=(10, 8))

        for y, x in np.argwhere(grid == 1):
            ax.add_patch(patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                linewidth=0.3, edgecolor="black", facecolor="black", alpha=0.5,
            ))

        # Манхэттен
        if i < len(path_manhattan):
            ys, xs = zip(*path_manhattan[:i + 1])
            ax.plot([x + 0.5 for x in xs], [y + 0.5 for y in ys],
                    c="blue", linewidth=2, label="Манхэттен")

        # NN
        if i < len(path_nn):
            ys, xs = zip(*path_nn[:i + 1])
            ax.plot([x + 0.5 for x in xs], [y + 0.5 for y in ys],
                    c="orange", linewidth=2, linestyle="--", label="NN")

        sy, sx = start
        ax.add_patch(patches.Rectangle(
            (sx - 0.5, sy - 0.5), 1, 1,
            linewidth=2, edgecolor="darkgreen", facecolor="lime", alpha=0.9,
        ))
        gy, gx = goal
        ax.add_patch(patches.Rectangle(
            (gx - 0.5, gy - 0.5), 1, 1,
            linewidth=2, edgecolor="darkred", facecolor="red", alpha=0.9,
        ))

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(m - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Сравнение путей | шаг {min(i, max_len)}/{max_len}",
                     fontsize=12, weight="bold")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()
        plt.close(fig)

    if frames:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        frames[0].save(path_out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
        print(f"[visualize] GIF сохранён: {path_out}")

# ---------------------------------------------------------------------------
# Дополнительные визуализации для отчёта и README
# ---------------------------------------------------------------------------

def plot_prediction_errors(
    dataset_path: str = "data/dataset.npz",
    model_path: str = "models/cnn_heuristic.pt",
    path_out: str = "results/training/error_histogram.png",
) -> None:
    """Гистограмма ошибок предсказания CNN (predicted - true)."""
    import torch
    from src.dataset import load_dataset
    from src.model import HeuristicCNN

    X, offsets, y = load_dataset(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    patch_size = ckpt["patch_size"]
    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]

    model = HeuristicCNN(patch_size=patch_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Берём подвыборку, чтобы не гонять всё
    n = min(2000, len(X))
    idx = np.random.choice(len(X), size=n, replace=False)

    X_t = torch.from_numpy(X[idx]).unsqueeze(1).to(device)
    off_t = torch.from_numpy(offsets[idx]).to(device)

    with torch.no_grad():
        pred_norm = model(X_t, off_t).cpu().numpy()

    pred = pred_norm * y_std + y_mean
    true = y[idx]
    errors = pred - true

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Идеал")
    ax.set_xlabel("Ошибка предсказания (шаги)")
    ax.set_ylabel("Количество примеров")
    ax.set_title(
        f"Распределение ошибок CNN\n"
        f"MAE={np.mean(np.abs(errors)):.2f} шагов, "
        f"std={np.std(errors):.2f}",
        fontsize=12, weight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Сохранено: {path_out}")


def plot_patch_examples(
    dataset_path: str = "data/dataset.npz",
    model_path: str = "models/cnn_heuristic.pt",
    path_out: str = "results/training/patch_examples.png",
    n_examples: int = 9,
    seed: int = 42,
) -> None:
    """Сетка 3×3: патч поля + истинное расстояние + предсказание."""
    import torch
    from src.dataset import load_dataset
    from src.model import HeuristicCNN

    X, offsets, y = load_dataset(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    patch_size = ckpt["patch_size"]
    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]

    model = HeuristicCNN(patch_size=patch_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_examples, replace=False)

    X_t = torch.from_numpy(X[idx]).unsqueeze(1).to(device)
    off_t = torch.from_numpy(offsets[idx]).to(device)

    with torch.no_grad():
        pred_norm = model(X_t, off_t).cpu().numpy()

    pred = pred_norm * y_std + y_mean
    true = y[idx]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for k, ax in enumerate(axes.flat):
        ax.imshow(X[idx[k]], cmap="gray_r", interpolation="nearest")
        ax.set_title(
            f"Истина: {true[k]:.0f}\nПредсказание: {pred[k]:.1f}",
            fontsize=10, weight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Примеры патчей поля и предсказания CNN",
                 fontsize=14, weight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Сохранено: {path_out}")


def plot_nn_heatmap(
    model_path: str = "models/cnn_heuristic.pt",
    seed: int = 42,
    path_out: str = "results/maps/nn_heatmap.png",
) -> None:
    """Heatmap: как CNN оценивает расстояние до цели по всему полю."""
    import torch
    from src.dataset import make_patch
    from src.grid import (
        create_empty_grid, fill_obstacles, gen_start_and_goal, make_rng,
    )
    from src.model import HeuristicCNN

    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    patch_size = ckpt["patch_size"]
    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]

    model = HeuristicCNN(patch_size=patch_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    heatmap = np.full((m, n), np.nan, dtype=np.float32)

    # Считаем предсказание для каждой свободной клетки
    free_cells = np.argwhere(grid == 0)
    batch_size = 256

    for i in range(0, len(free_cells), batch_size):
        batch = free_cells[i:i + batch_size]
        patches = np.stack([
            make_patch(grid, (int(y), int(x)), patch_size) for y, x in batch
        ])
        X_t = torch.from_numpy(patches).unsqueeze(1).to(device)

        dy = (goal[0] - batch[:, 0]).astype(np.float32) / m
        dx = (goal[1] - batch[:, 1]).astype(np.float32) / n
        off_t = torch.from_numpy(np.stack([dy, dx], axis=1)).to(device)

        with torch.no_grad():
            pred_norm = model(X_t, off_t).cpu().numpy()

        pred = pred_norm * y_std + y_mean
        for k, (y_, x_) in enumerate(batch):
            heatmap[int(y_), int(x_)] = max(0.0, pred[k])

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap, cmap="viridis", interpolation="nearest")

    # Отмечаем A и B
    ax.scatter(start[1], start[0], c="lime", s=200, edgecolors="darkgreen",
               marker="s", label="A")
    ax.scatter(goal[1], goal[0], c="red", s=200, edgecolors="darkred",
               marker="s", label="B")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Предсказание CNN: расстояние до цели по всему полю",
                 fontsize=13, weight="bold")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Предсказанное расстояние, шагов")
    plt.tight_layout()

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    fig.savefig(path_out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Сохранено: {path_out}")

# ---------------------------------------------------------------------------
# Обучающие GIF-анимации
# ---------------------------------------------------------------------------

def create_learning_progress_animation(
    model_dir: str = "models/checkpoints",
    seed: int = 42,
    path_out: str = "results/training/learning_progress.gif",
) -> None:
    """GIF: как меняется предсказание CNN по эпохам (по чекпоинтам)."""
    import io
    import glob
    from PIL import Image
    import torch
    from src.dataset import make_patch
    from src.grid import (
        create_empty_grid, fill_obstacles, gen_start_and_goal, make_rng,
    )
    from src.bfs import bfs_distances
    from src.model import HeuristicCNN

    # Находим все чекпоинты
    ckpt_paths = sorted(glob.glob(os.path.join(model_dir, "epoch_*.pt")))
    if not ckpt_paths:
        print(f"[visualize] Чекпоинты не найдены в {model_dir}")
        return

    print(f"[visualize] Найдено чекпоинтов: {len(ckpt_paths)}")

    # Генерируем поле для визуализации
    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)
    true_dist = bfs_distances(grid, goal)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    free_cells = np.argwhere(grid == 0)
    batch_size = 256

    frames = []

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        patch_size = ckpt["patch_size"]
        y_mean = ckpt["y_mean"]
        y_std = ckpt["y_std"]
        epoch = ckpt.get("epoch", "?")

        model = HeuristicCNN(patch_size=patch_size).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        pred_map = np.full((m, n), np.nan, dtype=np.float32)

        for i in range(0, len(free_cells), batch_size):
            batch = free_cells[i:i + batch_size]
            patches = np.stack([
                make_patch(grid, (int(y_), int(x_)), patch_size)
                for y_, x_ in batch
            ])
            X_t = torch.from_numpy(patches).unsqueeze(1).to(device)

            dy = (goal[0] - batch[:, 0]).astype(np.float32) / m
            dx = (goal[1] - batch[:, 1]).astype(np.float32) / n
            off_t = torch.from_numpy(np.stack([dy, dx], axis=1)).to(device)

            with torch.no_grad():
                pred_norm = model(X_t, off_t).cpu().numpy()

            pred = pred_norm * y_std + y_mean
            for k, (y_, x_) in enumerate(batch):
                pred_map[int(y_), int(x_)] = max(0.0, pred[k])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(true_dist, cmap="viridis", interpolation="nearest")
        axes[0].scatter(goal[1], goal[0], c="red", s=200,
                        edgecolors="darkred", marker="s")
        axes[0].scatter(start[1], start[0], c="lime", s=200,
                        edgecolors="darkgreen", marker="s")
        axes[0].set_title("Истинное расстояние (BFS)", fontsize=12, weight="bold")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(pred_map, cmap="viridis", interpolation="nearest")
        axes[1].scatter(goal[1], goal[0], c="red", s=200,
                        edgecolors="darkred", marker="s")
        axes[1].scatter(start[1], start[0], c="lime", s=200,
                        edgecolors="darkgreen", marker="s")
        axes[1].set_title(f"Предсказание CNN (эпоха {epoch})",
                          fontsize=12, weight="bold")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        plt.suptitle(
            f"Как сеть учится предсказывать расстояние до цели (эпоха {epoch})",
            fontsize=14, weight="bold",
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()
        plt.close(fig)

        print(f"[visualize] Кадр для эпохи {epoch} готов")

    if frames:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        frames[0].save(path_out, save_all=True, append_images=frames[1:],
                       duration=800, loop=0)
        print(f"[visualize] GIF сохранён: {path_out}")


def create_astar_nn_animation(
    model_path: str = "models/cnn_heuristic.pt",
    seed: int = 42,
    path_out: str = "results/comparison/astar_nn_animation.gif",
    fps: int = 8,
    frame_step: int = 3,
) -> None:
    """GIF: два A* (манхэттен и NN) шаг за шагом на одном поле."""
    import io
    from PIL import Image
    from src.astar import calc_shortest_path
    from src.astar_nn import calc_shortest_path_nn
    from src.grid import (
        create_empty_grid, fill_obstacles, gen_start_and_goal, make_rng,
    )

    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)

    path_m = calc_shortest_path(grid, start, goal)
    path_nn = calc_shortest_path_nn(grid, start, goal, model_path=model_path)

    if path_m is None or path_nn is None:
        print("[visualize] Один из путей не найден, анимация не создана")
        return

    max_len = max(len(path_m), len(path_nn))
    frames = []

    for i in range(0, max_len, frame_step):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, path, title, color in [
            (axes[0], path_m, "A* (манхэттен)", "blue"),
            (axes[1], path_nn, "A* (NN)", "orange"),
        ]:
            for y_, x_ in np.argwhere(grid == 1):
                ax.add_patch(patches.Rectangle(
                    (x_ - 0.5, y_ - 0.5), 1, 1,
                    linewidth=0.3, edgecolor="black",
                    facecolor="black", alpha=0.5,
                ))

            if i < len(path):
                ys, xs = zip(*path[:i + 1])
                ax.plot([x + 0.5 for x in xs], [y + 0.5 for y in ys],
                        c=color, linewidth=2)

            sy, sx = start
            ax.add_patch(patches.Rectangle(
                (sx - 0.5, sy - 0.5), 1, 1,
                linewidth=2, edgecolor="darkgreen",
                facecolor="lime", alpha=0.9,
            ))
            gy, gx = goal
            ax.add_patch(patches.Rectangle(
                (gx - 0.5, gy - 0.5), 1, 1,
                linewidth=2, edgecolor="darkred",
                facecolor="red", alpha=0.9,
            ))

            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(m - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title} | шаг {min(i, len(path))}/{len(path) - 1}",
                         fontsize=11, weight="bold")

        plt.suptitle("Сравнение A* шаг за шагом", fontsize=14, weight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=70, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()
        plt.close(fig)

    if frames:
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        frames[0].save(path_out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
        print(f"[visualize] GIF сохранён: {path_out}")