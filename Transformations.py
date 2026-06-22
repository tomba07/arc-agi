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


def rotate_90(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    return np.rot90(grid, k=1)


def rotate_180(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    return np.rot90(grid, k=2)


def rotate_270(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    return np.rot90(grid, k=3)


def mirror_horizontally(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    return np.maximum(grid, np.flipud(grid))


def make_recolor_transformation(from_color: int, to_color: int) -> Transform:
    def fn(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
        return np.where(grid == from_color, to_color, grid)

    return fn


def swap_colors(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    color1, color2 = colors
    return np.select([grid == color1, grid == color2], [color2, color1], grid)


def crop_to_content(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    shapes = _get_shapes(observations, example_index)
    if not shapes:
        return grid
    min_row = min(s.row for s in shapes)
    max_row = max(s.row + s.height - 1 for s in shapes)
    min_col = min(s.col for s in shapes)
    max_col = max(s.col + s.width - 1 for s in shapes)
    return grid[min_row : max_row + 1, min_col : max_col + 1]


def make_hollow(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    shapes = _get_shapes(observations, example_index)
    all_cells = set(cell for s in shapes for cell in s.cells)
    result = grid.copy()
    for row, col in all_cells:
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if all(n in all_cells for n in neighbors):
            result[row, col] = 0
    return result


def make_spiral_transformation(color: int) -> Transform:
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


def crop_to_square_abstraction(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    example = (
        observations.test_observations
        if example_index is None
        else observations.example_observations[example_index]
    )
    input_square_abstraction = example.input_square_abstraction

    if input_square_abstraction is None:
        return grid  # No abstraction to crop to
    else:
        row, col, width, height = (
            input_square_abstraction.row,
            input_square_abstraction.col,
            input_square_abstraction.width,
            input_square_abstraction.height,
        )

        return grid[row + 1 : row + height - 1, col + 1 : col + width - 1]


def recolor_to_square_abstraction(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    example = (
        observations.test_observations
        if example_index is None
        else observations.example_observations[example_index]
    )
    input_square_abstraction = example.input_square_abstraction

    if input_square_abstraction is None:
        return grid  # No abstraction to recolor to
    else:
        color = input_square_abstraction.color
        return np.where(grid != 0, color, grid)


def cast_uni_ray_from_two_by_twos(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    consistent_direction = observations.consistent_two_by_two_uni_ray_direction_by_color
    all_inputs_only_two_by_twos = observations.all_inputs_only_two_by_twos

    if all_inputs_only_two_by_twos is not True or consistent_direction is None:
        return grid  # No consistent direction to cast rays from
    else:
        shapes = _get_shapes(observations, example_index)
        result = grid.copy()

        # We already validated that all shapes are 2x2
        for shape in shapes:
            direction = consistent_direction.get(shape.color)

            if direction == "tl":
                start_row = shape.row - 1
                start_col = shape.col - 1
            elif direction == "tr":
                start_row = shape.row - 1
                start_col = shape.col + 2
            elif direction == "bl":
                start_row = shape.row + 2
                start_col = shape.col - 1
            elif direction == "br":
                start_row = shape.row + 2
                start_col = shape.col + 2

            while 0 <= start_row < grid.shape[0] and 0 <= start_col < grid.shape[1]:
                result[start_row, start_col] = shape.color

                if direction == "tl":
                    start_row -= 1
                    start_col -= 1
                elif direction == "tr":
                    start_row -= 1
                    start_col += 1
                elif direction == "bl":
                    start_row += 1
                    start_col -= 1
                elif direction == "br":
                    start_row += 1
                    start_col += 1
        return result


def make_recolor_by_enclosure_transformation(flip_colors: bool = False) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example_index: int | None
    ) -> Grid:
        colors = observations.consistent_new_output_colors
        if colors is None or len(colors) != 2:
            return grid
        color_a, color_b = sorted(colors)
        enclosed_color, non_enclosed_color = (color_b, color_a) if flip_colors else (color_a, color_b)
        example = (
            observations.test_observations
            if example_index is None
            else observations.example_observations[example_index]
        )
        result = grid.copy()
        for shape in example.enclosed_zero_shapes:
            for cell in shape.cells:
                result[cell] = enclosed_color
        for shape in example.non_enclosed_zero_shapes:
            for cell in shape.cells:
                result[cell] = non_enclosed_color
        return result

    return fn
