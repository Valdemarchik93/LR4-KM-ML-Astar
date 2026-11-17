"""CLI для запуска ЛР4."""

import os

import click

from src.compare import compare_on_maps, summarize
from src.dataset import build_dataset, save_dataset
from src.grid import (
    create_empty_grid,
    fill_obstacles,
    gen_start_and_goal,
    make_rng,
)
from src.astar import calc_shortest_path
from src.astar_nn import calc_shortest_path_nn
from src.train import train_model
from src.visualize import (
    plot_comparison,
    plot_example_map,
    plot_path_on_grid,
    plot_training_history,
    plot_two_paths_side_by_side,
)


@click.group()
def cli():
    """CLI для ЛР4: нейросетевая эвристика для A*."""
    pass


@cli.command()
@click.option("--n-maps", default=300, help="Число полей для датасета")
@click.option("--samples-per-map", default=100, help="Клеток с одного поля")
@click.option("--patch-size", default=11, help="Размер патча")
@click.option("--output", default="data/dataset.npz", help="Куда сохранить датасет")
def generate_data(n_maps, samples_per_map, patch_size, output):
    """Сгенерировать датасет для обучения."""
    click.echo(f"Генерация датасета: {n_maps} полей × {samples_per_map} клеток")
    X, offsets, y = build_dataset(
        n_maps=n_maps,
        samples_per_map=samples_per_map,
        patch_size=patch_size,
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_dataset(output, X, offsets, y)
    click.echo(f"Сохранено: {output} | X={X.shape}, offsets={offsets.shape}, y={y.shape}")


@cli.command()
@click.option("--model-dir", default="models/checkpoints")
@click.option("--seed", default=42)
@click.option("--output", default="results/training/learning_progress.gif")
def animate_learning(model_dir, seed, output):
    """GIF: как меняется предсказание CNN по эпохам (по чекпоинтам)."""
    from src.visualize import create_learning_progress_animation
    create_learning_progress_animation(
        model_dir=model_dir,
        seed=seed,
        path_out=output,
    )


@cli.command()
@click.option("--seed", default=42, help="Seed для генерации поля")
@click.option("--output", default="results/maps/astar_manhattan.png")
def run_astar(seed, output):
    """Запустить A* с манхэттенской эвристикой на одном поле."""
    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)

    path = calc_shortest_path(grid, start, goal)
    if path:
        click.echo(f"Путь найден: {len(path) - 1} шагов")
    else:
        click.echo("Путь не найден")

    plot_path_on_grid(
        grid, start, goal, path,
        title=f"A* (манхэттен), длина={len(path) - 1 if path else '—'}",
        path_out=output,
    )


@cli.command()
@click.option("--seed", default=42, help="Seed для генерации поля")
@click.option("--model", default="models/cnn_heuristic.pt")
@click.option("--output", default="results/maps/astar_nn.png")
def run_astar_nn(seed, model, output):
    """Запустить A* с нейросетевой эвристикой на одном поле."""
    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)

    path = calc_shortest_path_nn(grid, start, goal, model_path=model)
    if path:
        click.echo(f"Путь найден: {len(path) - 1} шагов")
    else:
        click.echo("Путь не найден")

    plot_path_on_grid(
        grid, start, goal, path,
        title=f"A* (NN), длина={len(path) - 1 if path else '—'}",
        path_out=output,
    )


@cli.command()
@click.option("--n-maps", default=50, help="Сколько полей сравнивать")
@click.option("--model", default="models/cnn_heuristic.pt")
@click.option("--output", default="results/comparison/compare.png")
def compare(n_maps, model, output):
    """Сравнить A* с манхэттенской и нейросетевой эвристикой."""
    click.echo(f"Сравнение на {n_maps} полях...")
    results = compare_on_maps(n_maps=n_maps, model_path=model)
    stats = summarize(results)

    click.echo("Средние значения:")
    for k, v in stats.items():
        click.echo(f"  {k}: {v:.4f}")

    plot_comparison(results, path_out=output)


@cli.command()
@click.option("--seed", default=42)
@click.option("--model", default="models/cnn_heuristic.pt")
@click.option("--output", default="results/comparison/two_paths.png")
def compare_two_paths(seed, model, output):
    """Нарисовать два пути (манхэттен и NN) на одном поле."""
    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)

    path_m = calc_shortest_path(grid, start, goal)
    path_nn = calc_shortest_path_nn(grid, start, goal, model_path=model)

    plot_two_paths_side_by_side(grid, start, goal, path_m, path_nn,
                                path_out=output)


@cli.command()
@click.option("--output", default="results/maps/example_map.png")
def example_map(output):
    """Сохранить пример поля с препятствиями и точками A/B."""
    plot_example_map(path_out=output)

@cli.command()
@click.option("--seed", default=42)
@click.option("--output", default="results/maps/astar_animation.gif")
def animate_astar(seed, output):
    """Создать GIF-анимацию поиска пути A* (манхэттен)."""
    from src.visualize import create_astar_animation

    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)

    path = calc_shortest_path(grid, start, goal)
    create_astar_animation(grid, start, goal, path,
                           title="A* (манхэттен)",
                           path_out=output)


@cli.command()
@click.option("--seed", default=42)
@click.option("--model", default="models/cnn_heuristic.pt")
@click.option("--output", default="results/comparison/comparison_animation.gif")
def animate_comparison(seed, model, output):
    """Создать GIF-анимацию сравнения двух путей (манхэттен vs NN)."""
    from src.visualize import create_comparison_animation

    rng = make_rng(seed)
    m, n = 80, 70
    grid = create_empty_grid((m, n))
    fill_obstacles(grid, 700, 1, 3, rng)
    start, goal = gen_start_and_goal(grid, rng, min_distance=20)

    path_m = calc_shortest_path(grid, start, goal)
    path_nn = calc_shortest_path_nn(grid, start, goal, model_path=model)

    create_comparison_animation(grid, start, goal, path_m, path_nn,
                                path_out=output)

@cli.command()
@click.option("--dataset", default="data/dataset.npz")
@click.option("--model", default="models/cnn_heuristic.pt")
@click.option("--output", default="results/training/error_histogram.png")
def plot_errors(dataset, model, output):
    """Построить гистограмму ошибок предсказания CNN."""
    from src.visualize import plot_prediction_errors
    plot_prediction_errors(dataset_path=dataset, model_path=model, path_out=output)


@cli.command()
@click.option("--dataset", default="data/dataset.npz")
@click.option("--model", default="models/cnn_heuristic.pt")
@click.option("--output", default="results/training/patch_examples.png")
def plot_patches(dataset, model, output):
    """Показать примеры патчей и предсказаний CNN."""
    from src.visualize import plot_patch_examples
    plot_patch_examples(dataset_path=dataset, model_path=model, path_out=output)


@cli.command()
@click.option("--model", default="models/cnn_heuristic.pt")
@click.option("--seed", default=42)
@click.option("--output", default="results/maps/nn_heatmap.png")
def plot_heatmap(model, seed, output):
    """Построить heatmap предсказаний CNN по всему полю."""
    from src.visualize import plot_nn_heatmap
    plot_nn_heatmap(model_path=model, seed=seed, path_out=output)


@cli.command()
@click.option("--model", default="models/cnn_heuristic.pt")
@click.option("--seed", default=42)
@click.option("--output", default="results/comparison/astar_nn_animation.gif")
def animate_astar_nn(model, seed, output):
    """GIF: два A* (манхэттен и NN) шаг за шагом."""
    from src.visualize import create_astar_nn_animation
    create_astar_nn_animation(
        model_path=model, seed=seed, path_out=output,
    )