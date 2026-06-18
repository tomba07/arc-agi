from typing import Callable

import numpy as np

from Observations import Shape

Grid = np.ndarray
Theory = list[Callable[[Grid, list[Shape]], Grid]]


def apply_theory(theory: Theory, grid: Grid, shapes: list[Shape]) -> Grid:
    for fn in theory:
        grid = fn(grid, shapes)
    return grid


def rotate_90(grid: Grid, shapes: list[Shape]) -> Grid:
    return np.rot90(grid, k=1)


def rotate_180(grid: Grid, shapes: list[Shape]) -> Grid:
    return np.rot90(grid, k=2)


def rotate_270(grid: Grid, shapes: list[Shape]) -> Grid:
    return np.rot90(grid, k=3)


def mirror_horizontally(grid: Grid, shapes: list[Shape]) -> Grid:
    return np.maximum(grid, np.flipud(grid))


def recolor(from_color: int, to_color: int) -> Callable[[Grid, list[Shape]], Grid]:
    def fn(grid: Grid, shapes: list[Shape]) -> Grid:
        return np.where(grid == from_color, to_color, grid)
    return fn


def swap_colors(grid: Grid, shapes: list[Shape]) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    color1, color2 = colors
    return np.select([grid == color1, grid == color2], [color2, color1], grid)


def crop_to_content(grid: Grid, shapes: list[Shape]) -> Grid:
    if not shapes:
        return grid
    min_row = min(s.row for s in shapes)
    max_row = max(s.row + s.height - 1 for s in shapes)
    min_col = min(s.col for s in shapes)
    max_col = max(s.col + s.width - 1 for s in shapes)
    return grid[min_row : max_row + 1, min_col : max_col + 1]


def make_hollow(grid: Grid, shapes: list[Shape]) -> Grid:
    all_cells = set(cell for s in shapes for cell in s.cells)
    result = grid.copy()
    for row, col in all_cells:
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if all(n in all_cells for n in neighbors):
            result[row, col] = 0
    return result
