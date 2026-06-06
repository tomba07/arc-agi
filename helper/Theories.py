from typing import Callable, List

import numpy as np

Grid = np.ndarray


def _make_hollow(grid: Grid) -> Grid:
    interior = np.zeros(grid.shape, dtype=bool)
    interior[1:-1, 1:-1] = (
        (grid[1:-1, 1:-1] != 0)
        & (grid[:-2, 1:-1] != 0)
        & (grid[2:, 1:-1] != 0)
        & (grid[1:-1, :-2] != 0)
        & (grid[1:-1, 2:] != 0)
    )
    return np.where(interior, 0, grid)


def _swap_colors(grid: Grid) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    color1, color2 = colors
    return np.select([grid == color1, grid == color2], [color2, color1], grid)


def _color_replace_and_erase(grid: Grid, replace_color: int) -> Grid:
    other_colors = sorted(set(np.unique(grid)) - {0, replace_color})
    if len(other_colors) != 1:
        return grid
    other_color = other_colors[0]
    return np.select(
        [grid == replace_color, grid == other_color], [other_color, 0], grid
    )


def _crop_to_content(grid: Grid) -> Grid:
    rows, cols = np.where(grid != 0)
    if len(rows) == 0:
        return grid
    return grid[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _rotate_90(grid: Grid) -> Grid:
    return np.rot90(grid, k=1)


def _rotate_180(grid: Grid) -> Grid:
    return np.rot90(grid, k=2)


def _rotate_270(grid: Grid) -> Grid:
    return np.rot90(grid, k=3)


def _flip_lr(grid: Grid) -> Grid:
    return np.fliplr(grid)


def _flip_ud(grid: Grid) -> Grid:
    return np.flipud(grid)


def _transpose(grid: Grid) -> Grid:
    return grid.T


def _anti_transpose(grid: Grid) -> Grid:
    return np.rot90(grid.T)


def _overlay_flip_ud(grid: Grid) -> Grid:
    return np.maximum(grid, np.flipud(grid))


def _recolor(from_color: int, to_color: int) -> Callable[[Grid], Grid]:
    def fn(grid: Grid) -> Grid:
        return np.where(grid == from_color, to_color, grid)

    return fn


def generate_theories() -> List[Callable[[Grid], Grid]]:
    transforms: List[Callable[[Grid], Grid]] = [
        _rotate_90,
        _rotate_180,
        _rotate_270,
        _flip_lr,
        _flip_ud,
        _transpose,
        _anti_transpose,
        _crop_to_content,
        _make_hollow,
        _overlay_flip_ud,
        _swap_colors,
    ]
    for from_color in range(1, 10):
        for to_color in range(0, 10):
            if from_color != to_color:
                transforms.append(_recolor(from_color, to_color))
        transforms.append(lambda grid, a=from_color: _color_replace_and_erase(grid, a))
    return transforms
