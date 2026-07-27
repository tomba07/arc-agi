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
    shapes = example.input_diagonal_shapes
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


def _draw_black_lr(top, bottom, left, right, grid):
    for c in range(left, right + 1):
        grid[top][c] = 0


def _draw_black_tb(top, bottom, left, right, grid):
    for r in range(top, bottom + 1):
        grid[r][left] = 0


def _draw_black_rl(top, bottom, left, right, grid):
    for c in range(right, left - 1, -1):
        grid[bottom][c] = 0


def _draw_black_bt(top, bottom, left, right, grid):
    for r in range(bottom, top - 1, -1):
        grid[r][right] = 0


_SPIRAL_DRAW_OPERATIONS = [
    _draw_black_lr,
    _draw_black_bt,
    _draw_black_rl,
    _draw_black_tb,
]
_SPIRAL_START_POSITION = {
    DiagonalDirection.TL: 0,
    DiagonalDirection.TR: 1,
    DiagonalDirection.BR: 2,
    DiagonalDirection.BL: 3,
}


def _draw_spiral(color, grid, start: DiagonalDirection = DiagonalDirection.TL):
    result = grid.copy()
    result.fill(color)
    max_row, max_col = grid.shape
    top, bottom, left, right = 1, max_row - 2, 0, max_col - 2
    operation_offset = _SPIRAL_START_POSITION[start]
    operations = (
        _SPIRAL_DRAW_OPERATIONS[operation_offset:]
        + _SPIRAL_DRAW_OPERATIONS[:operation_offset]
    )

    while top <= bottom and left <= right:
        for op in operations:
            op(top, bottom, left, right, result)
        top += 2
        bottom -= 2
        left += 2
        right -= 2

    return result


def make_spiral_transformation(color: int, rotation: int = 0) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        result = grid.copy()
        result.fill(color)
        work = np.rot90(result, k=rotation)
        path_color = int(grid[0, 0])
        max_row, max_col = work.shape
        top, bottom, left, right = 1, max_row - 2, 0, max_col - 2

        while top <= bottom and left <= right:
            for c in range(left, right + 1):
                work[top][c] = path_color
            for r in range(top, bottom + 1):
                work[r][right] = path_color
            if top < bottom - 1:
                for c in range(right, left, -1):
                    work[bottom][c] = path_color
            if left < right:
                for r in range(bottom, top + 1, -1):
                    work[r][left + 1] = path_color
            top += 2
            bottom -= 2
            left += 2
            right -= 2

        return np.rot90(work, k=-rotation)

    return fn


def make_spiral_transformation_reversed(color: int, rotation: int = 0) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        result = grid.copy()
        result.fill(color)
        work = np.rot90(result, k=rotation)
        path_color = int(grid[0, 0])
        max_row, max_col = work.shape
        top, bottom, left, right = 1, max_row - 2, 0, max_col - 2

        while top <= bottom and left <= right:
            for r in range(top, bottom + 1):
                work[r][left] = path_color
            for c in range(left, right + 1):
                work[bottom][c] = path_color
            if left < right - 1:
                for r in range(bottom, top, -1):
                    work[r][right] = path_color
            if top < bottom:
                for c in range(right, left, -1):
                    work[top][c] = path_color
            top += 2
            bottom -= 2
            left += 2
            right -= 2

        return np.rot90(work, k=-rotation)

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


def mirror_horizontally_vertically_and_diagonally(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    mirrored_horizontally = np.flipud(grid)
    mirrored_vertically = np.fliplr(grid)
    mirrored_diagonally = np.rot90(grid, 2)
    return np.block(
        [
            [mirrored_diagonally, mirrored_horizontally, mirrored_diagonally],
            [mirrored_vertically, grid, mirrored_vertically],
            [mirrored_diagonally, mirrored_horizontally, mirrored_diagonally],
        ]
    )


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
        return np.where(a & b, output_color, 0)
    elif operation == LogicalOperation.OR:
        return np.where(a | b, output_color, 0)
    elif operation == LogicalOperation.XOR:
        return np.where(a ^ b, output_color, 0)
    elif operation == LogicalOperation.NAND:
        return np.where(~(a & b), output_color, 0)
    elif operation == LogicalOperation.NOR:
        return np.where(~(a | b), output_color, 0)
    elif operation == LogicalOperation.XNOR:
        return np.where(~(a ^ b), output_color, 0)


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
    shape_cells = {cell for s in shapes for cell in s.cells}

    for zero_shape in example.enclosed_zero_shapes or []:
        shape_cells |= zero_shape.cells

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

    # prerequisites
    if grid.shape[1] < 3 or grid.shape[0] < 3:
        return result

    for shape in example.input_shapes:
        if shape.is_one_by_one:
            row, col = shape.row, shape.col
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    new_row, new_col = row + dr, col + dc

                    if (
                        new_row >= 0
                        and new_row < result.shape[0]
                        and new_col >= 0
                        and new_col < result.shape[1]
                    ):
                        result[new_row, new_col] = new_color
    return result


def move_one_by_ones_to_same_colored_wall(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = np.zeros_like(grid)
    wall_by_color = {
        shape.color: shape for shape in example.input_shapes if shape.is_wall
    }

    for shape in example.input_shapes:
        if shape.is_wall:
            # copy over walls to result
            result[
                shape.row : shape.row + shape.height,
                shape.col : shape.col + shape.width,
            ] = shape.color
        elif not shape.is_wall:
            row = shape.row
            col = shape.col
            color = shape.color
            matching_wall = wall_by_color.get(color)

            if matching_wall:
                wall_row = matching_wall.row
                wall_col = matching_wall.col

                # wall is left wall — place shape's left edge just inside wall
                if wall_col == 0:
                    result[row : row + shape.height, wall_col + 1 : wall_col + 1 + shape.width] = color

                # wall is top wall — place shape's top edge just inside wall
                elif wall_row == 0:
                    result[wall_row + 1 : wall_row + 1 + shape.height, col : col + shape.width] = color

                # wall is bottom wall — place shape's bottom edge just inside wall
                elif wall_row + matching_wall.height == result.shape[0]:
                    result[wall_row - shape.height : wall_row, col : col + shape.width] = color

                # wall is right wall — place shape's right edge just inside wall
                elif wall_col + matching_wall.width == result.shape[1]:
                    result[row : row + shape.height, wall_col - shape.width : wall_col] = color

    return result


def make_two_divider_overlay_operation(direction: Direction) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        def perform_overlay(subgrid1: Grid, subgrid2: Grid) -> Grid:
            return np.where(subgrid2 != 0, subgrid2, subgrid1)

        # overlay means that the last last grid wins. 0s are ignored
        if direction == Direction.DOWN:
            mid1 = grid.shape[0] // 3
            mid2 = 2 * grid.shape[0] // 3

            first_subgrid = grid[:mid1, :]
            second_subgrid = grid[mid1 + 1 : mid2, :]
            third_subgrid = grid[mid2 + 1 :, :]

            result = perform_overlay(first_subgrid, second_subgrid)
            result = perform_overlay(result, third_subgrid)

        elif direction == Direction.UP:
            mid1 = grid.shape[0] // 3
            mid2 = 2 * grid.shape[0] // 3

            first_subgrid = grid[mid2 + 1 :, :]
            second_subgrid = grid[mid1 + 1 : mid2, :]
            third_subgrid = grid[:mid1, :]

            result = perform_overlay(first_subgrid, second_subgrid)
            result = perform_overlay(result, third_subgrid)

        elif direction == Direction.RIGHT:
            mid1 = grid.shape[1] // 3
            mid2 = 2 * grid.shape[1] // 3

            first_subgrid = grid[:, :mid1]
            second_subgrid = grid[:, mid1 + 1 : mid2]
            third_subgrid = grid[:, mid2 + 1 :]

            result = perform_overlay(first_subgrid, second_subgrid)
            result = perform_overlay(result, third_subgrid)

        elif direction == Direction.LEFT:
            mid1 = grid.shape[1] // 3
            mid2 = 2 * grid.shape[1] // 3

            first_subgrid = grid[:, mid2 + 1 :]
            second_subgrid = grid[:, mid1 + 1 : mid2]
            third_subgrid = grid[:, :mid1]

            result = perform_overlay(first_subgrid, second_subgrid)
            result = perform_overlay(result, third_subgrid)

        return result

    return fn


def make_single_divider_overlay_operation(direction: Direction) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        def perform_overlay(subgrid1: Grid, subgrid2: Grid) -> Grid:
            return np.where(subgrid2 != 0, subgrid2, subgrid1)

        if direction == Direction.DOWN:
            mid = grid.shape[0] // 2
            first_subgrid = grid[:mid, :]
            second_subgrid = grid[mid + 1 :, :]

            result = perform_overlay(first_subgrid, second_subgrid)

        elif direction == Direction.UP:
            mid = grid.shape[0] // 2
            first_subgrid = grid[mid + 1 :, :]
            second_subgrid = grid[:mid, :]

            result = perform_overlay(first_subgrid, second_subgrid)

        elif direction == Direction.RIGHT:
            mid = grid.shape[1] // 2
            first_subgrid = grid[:, :mid]
            second_subgrid = grid[:, mid + 1 :]

            result = perform_overlay(first_subgrid, second_subgrid)

        elif direction == Direction.LEFT:
            mid = grid.shape[1] // 2
            first_subgrid = grid[:, mid + 1 :]
            second_subgrid = grid[:, :mid]

            result = perform_overlay(first_subgrid, second_subgrid)

        return result

    return fn


def single_cell_attraction(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()
    color_moved = None

    for ex in observations.example_observations:
        input_shapes = ex.input_shapes
        output_shapes = ex.output_shapes

        if len(input_shapes) != 2:
            return grid

        for shape in input_shapes:
            output_match = next(
                (s for s in output_shapes if s.color == shape.color), None
            )
            if output_match is None:
                continue
            if (shape.row, shape.col) != (output_match.row, output_match.col):
                if color_moved is not None and color_moved != shape.color:
                    return grid
                color_moved = shape.color

    if color_moved is None:
        return grid

    input_shapes = example.input_shapes
    if len(input_shapes) != 2:
        return grid

    moving_shape = next((s for s in input_shapes if s.color == color_moved), None)
    stationary_shape = next((s for s in input_shapes if s.color != color_moved), None)
    if moving_shape is None or stationary_shape is None:
        return grid

    new_row = moving_shape.row + (
        1
        if moving_shape.row < stationary_shape.row
        else -1
        if moving_shape.row > stationary_shape.row
        else 0
    )
    new_col = moving_shape.col + (
        1
        if moving_shape.col < stationary_shape.col
        else -1
        if moving_shape.col > stationary_shape.col
        else 0
    )

    result[moving_shape.row, moving_shape.col] = 0
    result[new_row, new_col] = moving_shape.color

    return result


def move_inner_shapes_outward_horizontal(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    shapes = example.input_color_strict_diagonal_shapes or []
    if len(shapes) != 4:
        return grid

    result = np.zeros_like(grid)
    s = sorted(shapes, key=lambda x: x.col)
    outer_left, inner_left, inner_right, outer_right = s[0], s[1], s[2], s[3]

    def _place_direct(shape: Shape) -> None:
        for row, col in shape.cells:
            result[row, col] = shape.color

    def _place_h_mirrored(shape: Shape, col_offset: int) -> None:
        min_col = min(col for _, col in shape.cells)
        max_col = max(col for _, col in shape.cells)
        for row, col in shape.cells:
            new_col = min_col + max_col - col + col_offset
            if 0 <= row < result.shape[0] and 0 <= new_col < result.shape[1]:
                result[row, new_col] = shape.color

    inner_left_offset = outer_left.col - inner_left.col - inner_left.width - 1
    inner_right_offset = outer_right.col + outer_right.width + 1 - inner_right.col

    _place_direct(outer_left)
    _place_h_mirrored(inner_left, inner_left_offset)
    _place_h_mirrored(inner_right, inner_right_offset)
    _place_direct(outer_right)
    return result


def move_inner_shapes_outward_vertical(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    shapes = example.input_color_strict_diagonal_shapes or []
    if len(shapes) != 4:
        return grid

    result = np.zeros_like(grid)
    s = sorted(shapes, key=lambda x: x.row)
    outer_top, inner_top, inner_bottom, outer_bottom = s[0], s[1], s[2], s[3]

    def _place_direct(shape: Shape) -> None:
        for row, col in shape.cells:
            if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
                result[row, col] = shape.color

    def _place_v_mirrored(shape: Shape, row_offset: int) -> None:
        min_row = min(row for row, _ in shape.cells)
        max_row = max(row for row, _ in shape.cells)
        for row, col in shape.cells:
            new_row = min_row + max_row - row + row_offset
            if 0 <= new_row < result.shape[0] and 0 <= col < result.shape[1]:
                result[new_row, col] = shape.color

    inner_top_offset = outer_top.row - inner_top.row - inner_top.height - 1
    inner_bottom_offset = outer_bottom.row + outer_bottom.height + 1 - inner_bottom.row

    _place_direct(outer_top)
    _place_v_mirrored(inner_top, inner_top_offset)
    _place_v_mirrored(inner_bottom, inner_bottom_offset)
    _place_direct(outer_bottom)
    return result


def move_inner_shapes_outward(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    if example.has_four_horizontally_aligned_shapes:
        return move_inner_shapes_outward_horizontal(grid, observations, example)
    return move_inner_shapes_outward_vertical(grid, observations, example)


def grow_and_connect_single_cells(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    offsets = [-1, 0, 1]
    result = grid.copy()

    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return result

    input_colors = set(np.unique(grid)) - {0}

    # grow 1x1 to 3x3 with opposite color
    for shape in example.input_shapes:
        if shape.width == 1 and shape.height == 1:
            new_color = (input_colors - {shape.color}).pop()

            for row, col in shape.cells:
                for row_offset in offsets:
                    for col_offset in offsets:
                        if row_offset == 0 and col_offset == 0:
                            continue
                        new_row = row + row_offset
                        new_col = col + col_offset

                        if (
                            0 <= new_row < result.shape[0]
                            and 0 <= new_col < result.shape[1]
                        ):
                            result[new_row, new_col] = new_color

    # build connections between horiz or vertically aligned 3x3 shapes with new output color
    connector_color = next(iter(observations.consistent_new_output_colors))

    for i, shape in enumerate(example.input_shapes):
        for j, other_shape in enumerate(example.input_shapes):
            if i != j and (
                shape.row == other_shape.row or shape.col == other_shape.col
            ):
                is_horizontal = shape.row == other_shape.row

                if is_horizontal:
                    start_col = min(shape.col, other_shape.col) + 2
                    end_col = max(shape.col, other_shape.col) - 2

                    while start_col <= end_col:
                        result[shape.row, start_col] = connector_color
                        if start_col != end_col:
                            result[shape.row, end_col] = connector_color
                        start_col += 2
                        end_col -= 2
                else:
                    start_row = min(shape.row, other_shape.row) + 2
                    end_row = max(shape.row, other_shape.row) - 2

                    while start_row <= end_row:
                        result[start_row, shape.col] = connector_color
                        if start_row != end_row:
                            result[end_row, shape.col] = connector_color
                        start_row += 2
                        end_row -= 2

    return result


def make_two_cell_line_connection(start_color: int) -> Transform:
    def fn(
        grid: Grid, observations: Observations, example: ExampleObservations
    ) -> Grid:
        result = grid.copy()
        colors = (
            example.input_colors
            if example.input_colors is not None
            else (set(np.unique(example.input_grid)) - {0})
        )
        shapes = example.input_shapes

        if start_color not in colors or len(colors) != 2 or len(shapes) != 2:
            return grid

        end_color = next(iter(colors - {start_color}))
        line_color = next(iter(observations.consistent_new_output_colors))
        start_shape = next(shape for shape in shapes if shape.color == start_color)
        end_shape = next(shape for shape in shapes if shape.color == end_color)

        row_diff = end_shape.row - start_shape.row
        col_diff = end_shape.col - start_shape.col
        row_sign = np.sign(row_diff)
        col_sign = np.sign(col_diff)
        # stop one step before end_shape in each axis that moves
        target_row = end_shape.row - row_sign if row_diff != 0 else start_shape.row
        target_col = end_shape.col - col_sign if col_diff != 0 else start_shape.col

        row, col = start_shape.row, start_shape.col
        # go diagonal along the minor axis, then straight along the major axis
        if abs(row_diff) <= abs(col_diff):
            while row != target_row:
                row += row_sign
                col += col_sign
                if row == end_shape.row and col == end_shape.col:
                    break
                result[row, col] = line_color
            while col != target_col:
                col += col_sign
                if row == end_shape.row and col == end_shape.col:
                    break
                result[row, col] = line_color
        else:
            while col != target_col:
                row += row_sign
                col += col_sign
                if row == end_shape.row and col == end_shape.col:
                    break
                result[row, col] = line_color
            while row != target_row:
                row += row_sign
                if row == end_shape.row and col == end_shape.col:
                    break
                result[row, col] = line_color

        return result

    return fn


def overlay_if_no_overlap(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    horizontal_divider = example.single_horizontal_divider
    vertical_divider = example.single_vertical_divider

    def overlay_subgrids_if_no_overlap(grid1, grid2):
        result = grid1.copy()

        for cell in np.argwhere(grid1 != 0):
            if grid2[cell[0], cell[1]] != 0:
                return result  # overlap: return left half unchanged

        for cell in np.argwhere(grid2 != 0):
            result[cell[0], cell[1]] = grid2[cell[0], cell[1]]

        return result

    if horizontal_divider:
        mid = grid.shape[0] // 2
        grid1 = grid[:mid, :]
        grid2 = grid[mid + 1 :, :]

        return overlay_subgrids_if_no_overlap(grid1, grid2)

    elif vertical_divider:
        mid = grid.shape[1] // 2
        grid1 = grid[:, :mid]
        grid2 = grid[:, mid + 1 :]

        return overlay_subgrids_if_no_overlap(grid1, grid2)


def connect_two_single_cells_on_rim(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()
    occupied_color = next(iter(observations.consistent_new_output_colors), None)
    rows, cols = grid.shape

    if occupied_color is None:
        return result

    def is_on_rim(r, c):
        return r == 0 or c == 0 or r == rows - 1 or c == cols - 1

    cells_by_color: dict = {}
    for shape in example.input_color_strict_shapes:
        if (
            shape.width == 1
            and shape.height == 1
            and is_on_rim(shape.row, shape.col)
            and shape.color != 0
        ):
            if shape.color not in cells_by_color:
                cells_by_color[shape.color] = []
            cells_by_color[shape.color].append(shape)

    cell_pair = next(
        (shapes for shapes in cells_by_color.values() if len(shapes) == 2), None
    )

    if cell_pair is not None:
        cell1, cell2 = cell_pair
        row, col = cell1.row, cell1.col
        row_sign = int(np.sign(cell2.row - cell1.row))
        col_sign = int(np.sign(cell2.col - cell1.col))
        row += row_sign
        col += col_sign

        while row != cell2.row or col != cell2.col:
            if result[row, col] == 0:
                result[row, col] = cell1.color
            else:
                result[row, col] = occupied_color
            row += row_sign
            col += col_sign

    return result


def count_enclosed_cells(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    if not observations.consistent_output_grid_size:
        return grid

    if len(example.enclosing_shapes) != 1:
        return grid

    enclosing_shape = example.enclosing_shapes[0]
    if not enclosing_shape.enclosed_cells:
        return grid

    enclosed_color = None
    count = 0
    for row, col in enclosing_shape.enclosed_cells:
        v = grid[row, col]
        if v != 0:
            if enclosed_color is None:
                enclosed_color = v
            count += 1

    if enclosed_color is None:
        return grid

    result = np.zeros(observations.consistent_output_grid_size)
    for i in range(count):
        r = i // result.shape[1]
        c = i % result.shape[1]
        if r < result.shape[0]:
            result[r, c] = enclosed_color

    return result


def expand_enclosing_shapes(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()
    enclosing_shapes = example.enclosing_shapes
    touch_directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    def valid_row_and_col(row, col):
        return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

    if len(enclosing_shapes) == 0:
        return result

    # derive new colors from the first example that has enclosing shapes
    new_colors = None
    for ex in observations.example_observations:
        if ex.has_enclosing_shapes and ex.new_output_colors:
            new_colors = ex.new_output_colors
            first_example_output = ex.output_grid
            break

    if not new_colors or len(new_colors) < 2:
        return result

    color_counts = {c: int(np.sum(first_example_output == c)) for c in new_colors}
    dominant_color = max(color_counts, key=lambda c: color_counts[c])
    other_color = next(c for c in new_colors if c != dominant_color)
    enclosing_shape_color = enclosing_shapes[0].color

    # inner
    for enclosed_shape in example.enclosed_zero_shapes:
        for cell in enclosed_shape.cells:
            for row_diff, col_diff in touch_directions:
                new_row = cell[0] + row_diff
                new_col = cell[1] + col_diff
                if valid_row_and_col(new_row, new_col):
                    if grid[new_row][new_col] == enclosing_shape_color:
                        result[cell[0], cell[1]] = other_color
                        break

    # outer
    for enclosing_shape in enclosing_shapes:
        for border_cell in enclosing_shape.cells:
            for row_diff, col_diff in touch_directions:
                new_row = border_cell[0] + row_diff
                new_col = border_cell[1] + col_diff
                if valid_row_and_col(new_row, new_col):
                    if result[new_row, new_col] == 0:
                        result[new_row, new_col] = dominant_color

    return result


def crop_and_color_change(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    indicator_shapes = example.color_change_indicator_shapes
    if not indicator_shapes:
        return grid

    color_change_data = {}
    for shape in indicator_shapes:
        cells = sorted(shape.cells)
        colors = [grid[r, c] for r, c in cells]
        color1 = colors[0]
        color2 = next(c for c in colors if c != color1)
        color_change_data[color2] = color1

    main_shape = next(
        (
            shape
            for shape in (example.input_shapes or [])
            if shape not in indicator_shapes
        ),
        None,
    )
    if main_shape is None:
        return grid

    row = main_shape.row
    col = main_shape.col
    result = grid[row : row + main_shape.height, col : col + main_shape.width].copy()

    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            if result[row, col] in color_change_data:
                result[row, col] = color_change_data[result[row, col]]

    return result


def crop_and_color_change_reversed(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    indicator_shapes = example.color_change_indicator_shapes
    if not indicator_shapes:
        return grid

    color_change_data = {}
    for shape in indicator_shapes:
        cells = sorted(shape.cells)
        colors = [grid[r, c] for r, c in cells]
        color1 = colors[0]
        color2 = next(c for c in colors if c != color1)
        color_change_data[color1] = color2

    main_shape = next(
        (
            shape
            for shape in (example.input_shapes or [])
            if shape not in indicator_shapes
        ),
        None,
    )
    if main_shape is None:
        return grid

    row = main_shape.row
    col = main_shape.col
    result = grid[row : row + main_shape.height, col : col + main_shape.width].copy()

    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            if result[row, col] in color_change_data:
                result[row, col] = color_change_data[result[row, col]]

    return result


def add_missing_mirrored_shape(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    result = grid.copy()

    for shape in example.input_color_strict_diagonal_shapes or []:
        missing_mirror = getattr(shape, "missing_diagonal_mirror", None)
        if missing_mirror is None:
            continue

        row, col, height, width = shape.row, shape.col, shape.height, shape.width
        subgrid = grid[row : row + height, col : col + width]
        mirrored = np.flipud(np.fliplr(subgrid))

        if missing_mirror == DiagonalDirection.TL:
            new_row, new_col = row - height - 1, col - width - 1
        elif missing_mirror == DiagonalDirection.TR:
            new_row, new_col = row - height - 1, col + width + 1
        elif missing_mirror == DiagonalDirection.BL:
            new_row, new_col = row + height + 1, col - width - 1
        else:  # BR
            new_row, new_col = row + height + 1, col + width + 1

        if (
            0 <= new_row
            and new_row + height <= result.shape[0]
            and 0 <= new_col
            and new_col + width <= result.shape[1]
        ):
            result[new_row : new_row + height, new_col : new_col + width] = mirrored

    return result


def add_roof_and_stripes(
    grid: Grid, observations: Observations, example: ExampleObservations
) -> Grid:
    input_grid_width = grid.shape[1]
    result = np.zeros((input_grid_width, input_grid_width), dtype=int)

    old_color = next(iter(example.input_colors or (set(np.unique(grid)) - {0})), None)
    new_color = next(iter(observations.consistent_new_output_colors or []), None)

    if old_color and new_color:
        # build roof starting from top
        left = right = input_grid_width // 2
        row = 0
        while left >= 0 and right < input_grid_width:
            result[row][left] = old_color
            if right != left:
                result[row][right] = old_color
            left -= 1
            right += 1
            row += 1

        # build stripes starting from (3, pos-1), step (+2, -2)
        starting_row = 3
        starting_col = input_grid_width // 2 - 1
        while starting_row < input_grid_width:
            row = starting_row + max(0, -starting_col)
            col = max(0, starting_col)
            while row < input_grid_width and col < input_grid_width:
                result[row][col] = new_color
                row += 1
                col += 1
            starting_row += 2
            starting_col -= 2

    return result
