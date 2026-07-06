from typing import Callable

import numpy as np

from Shapes import Shape, Grid
from Observations import Observations, ExampleObservations
from Enums import Direction, DiagonalDirection, AxisDirection, LogicalOperation

Transform = Callable[[Grid, Observations, ExampleObservations], Grid]
Theory = list[Transform]


def _get_shapes(example: ExampleObservations) -> list[Shape]:
    return example.input_shapes or []


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
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        return np.rot90(grid, k=k)

    return fn


rotate_90 = make_rotation(1)
rotate_180 = make_rotation(2)
rotate_270 = make_rotation(3)


def transpose(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    return grid.T


def mirror_across_horizontal_axis(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    return np.maximum(grid, np.flipud(grid))


def make_recolor_transformation(from_color: int, to_color: int) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        return np.where(grid == from_color, to_color, grid)

    return fn


def swap_colors(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    color1, color2 = colors
    return np.select([grid == color1, grid == color2], [color2, color1], grid)


def crop_to_content(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    shapes = _get_shapes(example)
    if not shapes:
        return grid
    min_row = min(s.row for s in shapes)
    max_row = max(s.row + s.height - 1 for s in shapes)
    min_col = min(s.col for s in shapes)
    max_col = max(s.col + s.width - 1 for s in shapes)
    return grid[min_row : max_row + 1, min_col : max_col + 1]


def make_hollow(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    shapes = _get_shapes(example)
    all_cells = set(cell for s in shapes for cell in s.cells)
    result = grid.copy()
    for row, col in all_cells:
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if all(n in all_cells for n in neighbors):
            result[row, col] = 0
    return result


def make_spiral_transformation(color: int) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
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
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
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
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    input_square_abstraction = example.input_square_abstraction

    if input_square_abstraction is None:
        return grid  # No abstraction to recolor to
    else:
        color = input_square_abstraction.color
        return np.where(grid != 0, color, grid)


def cast_uni_ray_from_two_by_twos(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    consistent_direction = observations.consistent_two_by_two_uni_ray_direction_by_color
    if (
        observations.all_inputs_only_two_by_twos is not True
        or consistent_direction is None
    ):
        return grid

    shapes = _get_shapes(example)
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
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        colors = observations.consistent_new_output_colors
        if colors is None or len(colors) != 2:
            return grid
        color_a, color_b = sorted(colors)
        enclosed_color, non_enclosed_color = (
            (color_b, color_a) if flip_colors else (color_a, color_b)
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


def make_arrange_colored_cells_transformations(
    direction: AxisDirection, increasing: bool = True
) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
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
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
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
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    spaceship_shape = example.spaceship_shape
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
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    top = np.hstack([grid, np.fliplr(grid)])
    bottom = np.hstack([np.flipud(grid), np.flipud(np.fliplr(grid))])
    return np.vstack([top, bottom])


def make_divider_operation(operation: LogicalOperation) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        if (
            not example.single_horizontal_divider
            and not example.single_vertical_divider
        ):
            return grid
        output_color = observations.single_output_color or 3
        if example.single_horizontal_divider:
            mid = grid.shape[0] // 2
            return _perform_logical_operation(
                grid[:mid, :], grid[mid + 1 :, :], operation, output_color
            )
        else:
            mid = grid.shape[1] // 2
            return _perform_logical_operation(
                grid[:, :mid], grid[:, mid + 1 :], operation, output_color
            )

    return fn


def make_implicit_divider_operation(operation: LogicalOperation) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        if (
            not example.single_implicit_horizontal_divider
            and not example.single_implicit_vertical_divider
        ):
            return grid
        output_color = observations.single_output_color or 3
        if example.single_implicit_horizontal_divider:
            mid = grid.shape[0] // 2
            return _perform_logical_operation(
                grid[:mid, :], grid[mid:, :], operation, output_color
            )
        else:
            mid = grid.shape[1] // 2
            return _perform_logical_operation(
                grid[:, :mid], grid[:, mid:], operation, output_color
            )

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
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    color = next(iter(observations.consistent_new_output_colors), None)
    if not example.enclosing_shapes or color is None:
        return grid

    result = grid.copy()
    for shape in example.enclosing_shapes:
        for cell in shape.cells:
            result[cell] = color
    return result


def fill_with_increasing_rows(
    grid: Grid, observations: Observations, example: ExampleObservations
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


def fill_enclosing_shapes_with_dominant_color(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()

    for enclosing_shape in example.enclosing_shapes:
        if not enclosing_shape.enclosed_cells:
            continue

        color_counts: dict[int, int] = {}

        for cell in enclosing_shape.enclosed_cells:
            color = grid[cell]
            if color != 0:
                color_counts[color] = color_counts.get(color, 0) + 1

        if not color_counts:
            continue

        dominant_color = max(color_counts, key=lambda c: color_counts[c])

        for cell in enclosing_shape.enclosed_cells:
            result[cell] = dominant_color

    return result


def remove_non_enclosed_single_cells(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()

    all_enclosed_cells = set()
    single_cells = set()

    for s in example.enclosing_shapes:
        all_enclosed_cells |= s.enclosed_cells

    for s in example.input_color_strict_shapes:
        if len(s.cells) == 1:
            single_cells |= s.cells

    for cell in single_cells:
        if cell not in all_enclosed_cells:
            result[cell] = 0

    return result


def connect_similar_shapes(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    shapes = [
        s
        for s in (example.input_diagonal_shapes or [])
        if s.color != 0 and (s.width > 1 or s.height > 1)
    ]
    new_colors = observations.consistent_new_output_colors
    if (
        not shapes
        or new_colors is None
        or len(new_colors) != 1
        or not observations.only_similar_input_shapes
    ):
        return grid

    new_color = next(iter(new_colors))

    result = grid.copy()
    shape_cells = set()
    for s in shapes:
        for r in range(s.row, s.row + s.height):
            for c in range(s.col, s.col + s.width):
                shape_cells.add((r, c))

    for shape in shapes:
        center_row = shape.row + shape.height // 2
        center_col = shape.col + shape.width // 2
        for other_shape in shapes:
            if shape == other_shape:
                continue

            other_center_row = other_shape.row + other_shape.height // 2
            other_center_col = other_shape.col + other_shape.width // 2

            if center_row == other_center_row:
                left = min(shape.col + shape.width, other_shape.col + other_shape.width)
                right = max(shape.col, other_shape.col)
                for c in range(left, right):
                    if (center_row, c) not in shape_cells:
                        result[center_row, c] = new_color
            elif center_col == other_center_col:
                top = min(
                    shape.row + shape.height, other_shape.row + other_shape.height
                )
                bottom = max(shape.row, other_shape.row)
                for r in range(top, bottom):
                    if (r, center_col) not in shape_cells:
                        result[r, center_col] = new_color

    return result


def put_shapes_into_bottom_gaps(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    max_row = grid.shape[0] - 1
    result = grid.copy()
    gaps = example.bottom_gaps
    count_consistent = len(example.input_shapes) - 1 != len(gaps)

    if not observations.bottom_gaps_everywhere or count_consistent:
        return grid

    for shape in example.input_shapes:
        is_bottom = shape.row + shape.height - 1 == max_row

        if not is_bottom:
            original_cells = shape.cells

            if shape.height != 2:
                if shape.width == 2:
                    shape = Shape(
                        row=shape.row,
                        col=shape.col,
                        width=shape.height,
                        height=shape.width,
                        cells=shape.cells,
                        color=shape.color,
                    )
                else:
                    continue

            for gap_row, gap_col_start, gap_col_end in gaps:
                gap_width = gap_col_end - gap_col_start + 1

                if shape.width == gap_width:
                    # add the shape to the gap in the result grid
                    for r in range(shape.height):
                        for c in range(shape.width):
                            result[
                                gap_row - shape.height + r + 1, gap_col_start + c
                            ] = shape.color

                    # remove the gap from the list of gaps
                    gaps.remove((gap_row, gap_col_start, gap_col_end))

                    # remove the original shape from the result grid
                    for row, col in original_cells:
                        result[row, col] = 0
                    break

    return result


def print_two_by_two_color_count(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    non_two_by_two_shape = next(
        shape for shape in example.input_shapes if not shape.is_two_by_two
    )
    non_two_by_two_color = non_two_by_two_shape.color

    color_count = {}
    for shape in example.input_shapes:
        if shape.is_two_by_two and shape.color != non_two_by_two_color:
            color_count[shape.color] = color_count.get(shape.color, 0) + 1

    max_color_count = max(color_count.values(), default=0)

    # print colored cells matching the color counts in increasing order of color count
    sorted_colors = sorted(color_count.items(), key=lambda x: x[1])
    # output height is number of columns, width is max color count
    result = np.zeros((len(sorted_colors), max_color_count), dtype=int)

    for i, (color, count) in enumerate(sorted_colors):
        result[i, :count] = color

    return result


def mirror_single_enclosed_shape(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()

    for enclosing_shape in example.enclosing_shapes:
        if not enclosing_shape.enclosed_shapes:
            continue

        row_sum = enclosing_shape.row * 2 + enclosing_shape.height - 1
        col_sum = enclosing_shape.col * 2 + enclosing_shape.width - 1

        color = enclosing_shape.enclosed_shapes[0].color
        cells = {cell for s in enclosing_shape.enclosed_shapes for cell in s.cells}
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]

        row_offset = abs((min(rows) + max(rows)) - row_sum)
        col_offset = abs((min(cols) + max(cols)) - col_sum)

        if row_offset >= col_offset and enclosing_shape.height >= enclosing_shape.width:
            for r, c in cells:
                result[row_sum - r, c] = color
        else:
            for r, c in cells:
                result[r, col_sum - c] = color

    return result


def cast_rays_from_single_cells(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()
    for shape in example.input_shapes:
        if shape.width == 1 and shape.height == 1:
            color = shape.color
            row, col = shape.row, shape.col
            for row_diff, col_diff in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                new_row, new_col = row + row_diff, col + col_diff
                while 0 <= new_row < result.shape[0] and 0 <= new_col < result.shape[1]:
                    result[new_row, new_col] = color
                    new_row += row_diff
                    new_col += col_diff
    return result


def grow_one_by_ones(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()
    new_color = next(iter(observations.consistent_new_output_colors), None)

    for shape in example.input_shapes:
        if shape.is_one_by_one:
            row, col = shape.row, shape.col
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    new_row, new_col = row + dr, col + dc
                    
                    result[new_row, new_col] = new_color
    return result
