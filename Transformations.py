from typing import Callable, TYPE_CHECKING

import numpy as np

from Observations import Grid, Shape, Observations

Theory = list[Callable[[Grid, Observations], Grid]]


def rotate_90(grid: Grid, obs: Observations) -> Grid:
    return np.rot90(grid, k=1)


def rotate_180(grid: Grid, obs: Observations) -> Grid:
    return np.rot90(grid, k=2)


def rotate_270(grid: Grid, obs: Observations) -> Grid:
    return np.rot90(grid, k=3)


def mirror_horizontally(grid: Grid, obs: Observations) -> Grid:
    return np.maximum(grid, np.flipud(grid))


def recolor(from_color: int, to_color: int) -> Callable[[Grid, Observations], Grid]:
    def fn(grid: Grid, obs: Observations) -> Grid:
        return np.where(grid == from_color, to_color, grid)

    return fn


def make_color_map_fn(mapping: dict) -> Callable[[Grid, Observations], Grid]:
    def fn(grid: Grid, obs: Observations) -> Grid:
        result = np.zeros_like(grid)
        for k, v in mapping.items():
            result[grid == k] = v
        return result

    return fn


def swap_colors(grid: Grid, obs: Observations) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    color1, color2 = colors
    return np.select([grid == color1, grid == color2], [color2, color1], grid)


def crop_to_content(grid: Grid, obs: Observations) -> Grid:
    if not obs.shapes:
        return grid
    min_row = min(s.row for s in obs.shapes)
    max_row = max(s.row + s.height - 1 for s in obs.shapes)
    min_col = min(s.col for s in obs.shapes)
    max_col = max(s.col + s.width - 1 for s in obs.shapes)
    return grid[min_row : max_row + 1, min_col : max_col + 1]


def make_hollow(grid: Grid, obs: Observations) -> Grid:
    all_cells = set(cell for s in obs.shapes for cell in s.cells)
    result = grid.copy()
    for row, col in all_cells:
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if all(n in all_cells for n in neighbors):
            result[row, col] = 0
    return result
