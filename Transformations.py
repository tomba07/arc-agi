from typing import Callable

import numpy as np

from Observations import Shape, Observations

Grid = np.ndarray
Transform = Callable[[Grid, Observations, int | None], Grid]
Theory = list[Transform]


def apply_theory(
    theory: Theory,
    observations: Observations,
    example_index: int | None,
) -> Grid:
    if example_index is None:
        grid = observations.test_observations.input_grid.copy()
    else:
        grid = observations.example_observations[example_index].input_grid.copy()
    for fn in theory:
        grid = fn(grid, observations, example_index)
    return grid


def _get_shapes(observations: Observations, example_index: int | None) -> list[Shape]:
    if example_index is None:
        return observations.test_observations.input_shapes
    return observations.example_observations[example_index].input_shapes


def rotate_90(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    return np.rot90(grid, k=1)


def rotate_180(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    return np.rot90(grid, k=2)


def rotate_270(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    return np.rot90(grid, k=3)


def mirror_horizontally(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    return np.maximum(grid, np.flipud(grid))


def recolor(from_color: int, to_color: int) -> Transform:
    def fn(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
        return np.where(grid == from_color, to_color, grid)

    return fn


def swap_colors(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    color1, color2 = colors
    return np.select([grid == color1, grid == color2], [color2, color1], grid)


def crop_to_content(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    shapes = _get_shapes(observations, example_index)
    if not shapes:
        return grid
    min_row = min(s.row for s in shapes)
    max_row = max(s.row + s.height - 1 for s in shapes)
    min_col = min(s.col for s in shapes)
    max_col = max(s.col + s.width - 1 for s in shapes)
    return grid[min_row : max_row + 1, min_col : max_col + 1]


def make_hollow(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    shapes = _get_shapes(observations, example_index)
    all_cells = set(cell for s in shapes for cell in s.cells)
    result = grid.copy()
    for row, col in all_cells:
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if all(n in all_cells for n in neighbors):
            result[row, col] = 0
    return result


def generate_spiral(color: int) -> Transform:
    def fn(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
        grid = grid.copy()
        grid.fill(color)
        max_row, max_col = grid.shape
        top, bottom, left, right = 1, max_row - 2, 0, max_col - 2

        while top <= bottom and left <= right:
            # Draw black path from left to right
            for c in range(left, right + 1):
                grid[top][c] = 0

            # Draw black path downward
            for r in range(top, bottom + 1):
                grid[r][right] = 0

            # Draw black path from right to left
            if top < bottom - 1:
                for c in range(right, left, -1):
                    grid[bottom][c] = 0

            # Draw black path upward
            if left < right:
                for r in range(bottom, top + 1, -1):
                    grid[r][left + 1] = 0

            top += 2
            bottom -= 2
            left += 2
            right -= 2

        return grid

    return fn

def crop_to_square_abstraction(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    example = observations.test_observations if example_index is None else observations.example_observations[example_index]
    input_square_abstraction = example.input_square_abstraction
    
    if input_square_abstraction is None:
        return grid  # No abstraction to crop to
    else:
        row, col, width, height = input_square_abstraction.row, input_square_abstraction.col, input_square_abstraction.width, input_square_abstraction.height
        
        return grid[row + 1:row + height - 1, col + 1:col + width - 1]
    
def recolor_to_square_abstraction(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
    example = observations.test_observations if example_index is None else observations.example_observations[example_index]
    input_square_abstraction = example.input_square_abstraction
    
    if input_square_abstraction is None:
        return grid  # No abstraction to recolor to
    else:
        color = input_square_abstraction.color
        return np.where(grid != 0, color, grid)