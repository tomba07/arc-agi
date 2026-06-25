from typing import Callable

import numpy as np

from Observations import Shape, Observations
from Enums import Direction, DiagonalDirection, AxisDirection, LogicalOperation

Grid = np.ndarray
Transform = Callable[[Grid, Observations, int | None], Grid]
Theory = list[Transform]


def _get_example(observations: Observations, example_index: int | None):
    if example_index is None:
        return observations.test_observations
    return observations.example_observations[example_index]


def _get_shapes(observations: Observations, example_index: int | None) -> list[Shape]:
    return _get_example(observations, example_index).input_shapes or []


def apply_theory(
    theory: Theory,
    observations: Observations,
    example_index: int | None,
) -> Grid:
    grid = _get_example(observations, example_index).input_grid.copy()
    for fn in theory:
        grid = fn(grid, observations, example_index)
    return grid


_DIAGONAL_DELTA: dict[DiagonalDirection, tuple[int, int]] = {
    DiagonalDirection.TL: (-1, -1),
    DiagonalDirection.TR: (-1, +1),
    DiagonalDirection.BL: (+1, -1),
    DiagonalDirection.BR: (+1, +1),
}


def _cast_ray(
    result: Grid, start_row: int, start_col: int, dr: int, dc: int, color: int
) -> None:
    r, c = start_row, start_col
    while 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
        result[r, c] = color
        r += dr
        c += dc


def make_rotation(k: int) -> Transform:
    def fn(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
        return np.rot90(grid, k=k)
    return fn


rotate_90 = make_rotation(1)
rotate_180 = make_rotation(2)
rotate_270 = make_rotation(3)


def transpose(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    return grid.T


def mirror_across_horizontal_axis(
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
    example = _get_example(observations, example_index)
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
    example = _get_example(observations, example_index)
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
    if observations.all_inputs_only_two_by_twos is not True or consistent_direction is None:
        return grid

    shapes = _get_shapes(observations, example_index)
    result = grid.copy()
    for shape in shapes:
        direction = consistent_direction.get(shape.color)
        if direction is None:
            continue
        dr, dc = _DIAGONAL_DELTA[direction]
        start_row = shape.row + (shape.height if dr > 0 else -1)
        start_col = shape.col + (shape.width if dc > 0 else -1)
        _cast_ray(result, start_row, start_col, dr, dc, shape.color)
    return result


def make_recolor_by_enclosure_transformation(flip_colors: bool = False) -> Transform:
    def fn(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
        colors = observations.consistent_new_output_colors
        if colors is None or len(colors) != 2:
            return grid
        color_a, color_b = sorted(colors)
        enclosed_color, non_enclosed_color = (
            (color_b, color_a) if flip_colors else (color_a, color_b)
        )
        example = _get_example(observations, example_index)
        result = grid.copy()
        for shape in example.enclosed_zero_shapes:
            for cell in shape.cells:
                result[cell] = enclosed_color
        for shape in example.non_enclosed_zero_shapes:
            for cell in shape.cells:
                result[cell] = non_enclosed_color
        return result

    return fn


def make_arrange_colored_cells_transformations(
    direction: AxisDirection, increasing: bool = True
) -> Transform:
    def fn(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
        example = _get_example(observations, example_index)
        cell_count_by_color = example.input_cell_count_by_color

        if cell_count_by_color is None:
            return grid

        rows = None
        cols = None

        if direction == AxisDirection.HORIZONTAL:
            rows = len(cell_count_by_color)
            cols = max(cell_count_by_color.values())
        elif direction == AxisDirection.VERTICAL:
            cols = len(cell_count_by_color)
            rows = max(cell_count_by_color.values())

        result = np.zeros((rows, cols), dtype=int)

        sorted_colors = sorted(
            cell_count_by_color.items(), key=lambda x: x[1], reverse=not increasing
        )
        for i, (color, count) in enumerate(sorted_colors):
            if direction == AxisDirection.HORIZONTAL:
                result[i, :count] = color
            elif direction == AxisDirection.VERTICAL:
                result[:count, i] = color

        return result

    return fn


def connect_same_color_opposing_cells(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    example = _get_example(observations, example_index)
    opposing_cells = example.opposing_same_color_single_cells

    if not opposing_cells:
        return grid  # No opposing cells to connect

    result = grid.copy()
    for cell1, cell2 in opposing_cells:
        row1, col1 = cell1.row, cell1.col
        row2, col2 = cell2.row, cell2.col
        color = result[row1, col1]

        if row1 == row2:
            for c in range(min(col1, col2), max(col1, col2) + 1):
                result[row1, c] = color
        elif col1 == col2:
            for r in range(min(row1, row2), max(row1, row2) + 1):
                result[r, col1] = color

    return result


def create_beam_from_spaceship_tip(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    spaceship_shape = _get_example(observations, example_index).spaceship_shape
    if spaceship_shape is None:
        return grid

    result = grid.copy()
    tip_row, tip_col = spaceship_shape.tip_row, spaceship_shape.tip_col
    color = spaceship_shape.beam_color
    _CARDINAL_DELTA = {
        Direction.UP: (-1, 0),
        Direction.DOWN: (+1, 0),
        Direction.LEFT: (0, -1),
        Direction.RIGHT: (0, +1),
    }
    dr, dc = _CARDINAL_DELTA[spaceship_shape.direction]
    _cast_ray(result, tip_row + dr, tip_col + dc, dr, dc, color)
    return result


def mirror_horizontally_and_vertically(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    top = np.hstack([grid, np.fliplr(grid)])
    bottom = np.hstack([np.flipud(grid), np.flipud(np.fliplr(grid))])
    return np.vstack([top, bottom])


def make_divider_operation(operation: LogicalOperation) -> Transform:
    def fn(grid: Grid, observations: Observations, example_index: int | None) -> Grid:
        example = _get_example(observations, example_index)
        if not example.single_horizontal_divider and not example.single_vertical_divider:
            return grid
        output_color = observations.single_output_color or 3
        if example.single_horizontal_divider:
            mid = grid.shape[0] // 2
            return _perform_logical_operation(grid[:mid, :], grid[mid + 1:, :], operation, output_color)
        else:
            mid = grid.shape[1] // 2
            return _perform_logical_operation(grid[:, :mid], grid[:, mid + 1:], operation, output_color)
    return fn


def _perform_logical_operation(
    grid1: Grid, grid2: Grid, operation: LogicalOperation, output_color: int
) -> Grid:
    a = grid1 != 0
    b = grid2 != 0
    if operation == LogicalOperation.AND:
        mask = a & b
    elif operation == LogicalOperation.OR:
        mask = a | b
    elif operation == LogicalOperation.XOR:
        mask = a ^ b
    elif operation == LogicalOperation.NAND:
        mask = ~(a & b)
    elif operation == LogicalOperation.NOR:
        mask = ~(a | b)
    elif operation == LogicalOperation.XNOR:
        mask = ~(a ^ b)
    else:
        raise ValueError(f"Unsupported logical operation: {operation}")
    return np.where(mask, output_color, 0)


def change_enclosing_shapes_color(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    example = _get_example(observations, example_index)
    color = next(iter(observations.consistent_new_output_colors), None)
    if not example.enclosing_shapes or color is None:
        return grid

    result = grid.copy()
    for shape in example.enclosing_shapes:
        for cell in shape.cells:
            result[cell] = color
    return result


def fill_with_increasing_rows(
    grid: Grid, observations: Observations, example_index: int | None
) -> Grid:
    cols = grid.shape[1]
    filled_cols = int(np.count_nonzero(grid[0]))
    if filled_cols == 0:
        return grid
    color = int(grid[0, 0])
    rows = cols // 2
    result = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        result[i, : filled_cols + i] = color
    return result
