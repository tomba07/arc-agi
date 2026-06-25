from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

Grid = np.ndarray


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class DiagonalDirection(str, Enum):
    TL = "tl"
    TR = "tr"
    BL = "bl"
    BR = "br"


class AxisDirection(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class LogicalOperation(str, Enum):
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"
    XNOR = "XNOR"


@dataclass
class CellsInfo:
    cells: set[tuple[int, int]]
    color: int


@dataclass
class Shape:
    row: int
    col: int
    width: int
    height: int
    cells: set[tuple[int, int]]
    is_square_abstraction: bool = False
    color: int | None = None
    is_two_by_two: bool = False
    encloses_cells: bool = False


@dataclass
class Spaceship_Shape(Shape):
    is_spaceship_shape: bool = True
    beam_color: int | None = None
    direction: Direction | None = None
    tip_row: int | None = None
    tip_col: int | None = None


@dataclass
class ExampleObservations:
    input_grid: Grid
    output_grid: Grid | None = None
    input_shape_count: int = 0
    enclosed_zero_shapes: list[Shape] | None = None
    non_enclosed_zero_shapes: list[Shape] | None = None
    output_colors: set[int] | None = None
    output_colors_count: int = 0
    new_output_colors: set[int] | None = None
    new_output_colors_count: int = 0
    input_shapes: list[Shape] | None = None
    output_shapes: list[Shape] | None = None
    input_square_abstraction: Shape | None = None
    input_square_abstraction_color: int | None = None
    input_only_two_by_twos: bool = False
    two_by_two_uni_ray_direction_by_color: dict[int, DiagonalDirection | None] | None = None
    input_cell_count_by_color: dict[int, int] | None = None
    output_cell_count_by_color: dict[int, int] | None = None
    cell_count_by_color_identical: bool | None = None
    opposing_same_color_single_cells: list[tuple[Shape, Shape]] | None = None
    spaceship_shape: Spaceship_Shape | None = None
    output_twice_as_large_as_input: bool = False
    single_horizontal_divider: bool = False
    single_vertical_divider: bool = False
    has_enclosing_shapes: bool = False
    enclosing_shapes: list[Shape] = field(default_factory=list)
    input_color_strict_shapes: list[Shape] | None = None
    output_color_strict_shapes: list[Shape] | None = None
    output_height_half_of_width: bool = False


@dataclass
class Observations:
    example_observations: list[ExampleObservations]
    test_observations: ExampleObservations
    grid_size_stays_identical: bool | None = None
    grid_size_decreases: bool | None = None
    shapes_collected: bool | None = None
    single_shape_everywhere: bool | None = None
    all_inputs_empty: bool | None = None
    single_output_color: int | None = None
    input_square_abstraction_everywhere: bool | None = None
    all_inputs_only_two_by_twos: bool | None = None
    consistent_two_by_two_uni_ray_direction_by_color: dict[int, DiagonalDirection | None] | None = None
    consistent_new_output_colors: list[int] | None = None
    two_new_output_colors_everywhere: bool | None = None
    enclosed_zero_shapes_everywhere: bool | None = None
    non_enclosed_zero_shapes_everywhere: bool | None = None
    cell_count_by_color_identical_everywhere: bool | None = None
    has_opposing_same_color_single_cells_everywhere: bool | None = None
    has_spaceship_shape_everywhere: bool | None = None
    all_outputs_twice_as_large_as_inputs: bool | None = None
    has_single_horizontal_divider_everywhere: bool | None = None
    has_single_vertical_divider_everywhere: bool | None = None
    input_color_always_zeroed: int | None = None
    has_enclosing_shapes_everywhere: bool | None = None
    output_height_half_of_width_everywhere: bool | None = None


def _collect_cells(
    grid: Grid, start_row: int, start_col: int, visited: set
) -> CellsInfo:
    cells: set[tuple[int, int]] = set()
    queue = [(start_row, start_col)]
    color = grid[start_row, start_col]
    while queue:
        row, col = queue.pop(0)
        if (row, col) in visited or not (
            0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        ):
            continue
        if grid[row, col] == 0:
            continue
        visited.add((row, col))
        cells.add((row, col))
        queue.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])

    return CellsInfo(cells=cells, color=color)


def _make_shape(cells: set[tuple[int, int]], color: int) -> Shape:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    return Shape(
        row=min_row,
        col=min_col,
        width=width,
        height=height,
        cells=cells,
        color=color,
        is_two_by_two=(height == 2 and width == 2),
    )


def get_shapes(grid: Grid) -> list[Shape]:
    visited: set[tuple[int, int]] = set()
    shapes = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] == 0 or (start_row, start_col) in visited:
                continue
            cells_info = _collect_cells(grid, start_row, start_col, visited)
            shapes.append(_make_shape(cells_info.cells, cells_info.color))

    return shapes


def get_color_strict_shapes(grid: Grid) -> list[Shape]:
    """Like get_shapes but each connected same-color region is its own shape."""
    visited: set[tuple[int, int]] = set()
    shapes = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if (start_row, start_col) in visited:
                continue
            color = int(grid[start_row, start_col])
            cells: set[tuple[int, int]] = set()
            queue = [(start_row, start_col)]
            while queue:
                r, c = queue.pop()
                if (r, c) in visited or not (
                    0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                ):
                    continue
                if grid[r, c] != color:
                    continue
                visited.add((r, c))
                cells.add((r, c))
                queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
            shapes.append(_make_shape(cells, color))
    return shapes


def _check_spaceship_shape(shapes: list[Shape], grid: Grid) -> Shape | None:
    directions = list(Direction)

    if len(shapes) != 1:
        return None
    else:
        shape = shapes[0]
        height = shape.height
        width = shape.width
        ratio_correct = width == height * 2 - 1 or height == width * 2 - 1

        if ratio_correct:
            for direction in directions:
                beam_color = _check_pyramid_shape_and_beam_color(shape, grid, direction)
                if beam_color is not None:
                    tip_row = (
                        shape.row
                        if direction == Direction.UP
                        else shape.row + shape.height - 1
                        if direction == Direction.DOWN
                        else shape.row + shape.height // 2
                    )
                    tip_col = (
                        shape.col + shape.width // 2
                        if direction in [Direction.UP, Direction.DOWN]
                        else shape.col
                        if direction == Direction.LEFT
                        else shape.col + shape.width - 1
                    )
                    return Spaceship_Shape(
                        row=shape.row,
                        col=shape.col,
                        width=shape.width,
                        height=shape.height,
                        cells=shape.cells,
                        color=shape.color,
                        is_spaceship_shape=True,
                        direction=direction,
                        beam_color=beam_color,
                        tip_row=tip_row,
                        tip_col=tip_col,
                    )
    return None


def _check_pyramid_shape_and_beam_color(
    shape: Shape, grid: Grid, direction: Direction
) -> int | None:
    center_row = shape.row + shape.height // 2
    center_col = shape.col + shape.width // 2

    if direction in (Direction.UP, Direction.DOWN):
        beam_row = shape.row + shape.height - 1 if direction == Direction.UP else shape.row
        beam_col = center_col
        cells = [
            (
                shape.row + i
                if direction == Direction.UP
                else shape.row + shape.height - 1 - i,
                center_col + dc,
            )
            for i in range(shape.height)
            for dc in range(-i, i + 1)
        ]
    else:
        beam_row = center_row
        beam_col = shape.col + shape.width - 1 if direction == Direction.LEFT else shape.col
        cells = [
            (
                center_row + dr,
                shape.col + i
                if direction == Direction.LEFT
                else shape.col + shape.width - 1 - i,
            )
            for i in range(shape.width)
            for dr in range(-i, i + 1)
        ]

    for row, col in cells:
        if (row, col) == (beam_row, beam_col):
            continue
        if grid[row, col] != shape.color:
            return None

    beam_color = grid[beam_row, beam_col]
    return None if beam_color == shape.color else beam_color


def _collect_opposing_same_color_single_cells(
    shapes: list[Shape],
) -> list[tuple[Shape, Shape]]:
    single_cell_shapes = [
        shape for shape in shapes if shape.width == 1 and shape.height == 1
    ]
    opposing_pairs = []
    for i, shape1 in enumerate(single_cell_shapes):
        for j, shape2 in enumerate(single_cell_shapes):
            if i >= j:
                continue
            if shape1.color == shape2.color:
                left_right = (
                    shape1.col == 0 and shape2.col == max(shape1.col, shape2.col)
                ) or (shape1.col == max(shape1.col, shape2.col) and shape2.col == 0)
                top_bottom = (
                    shape1.row == 0 and shape2.row == max(shape1.row, shape2.row)
                ) or (shape1.row == max(shape1.row, shape2.row) and shape2.row == 0)
                if left_right or top_bottom:
                    opposing_pairs.append((shape1, shape2))
    return opposing_pairs


def _collect_zero_cells(
    grid: Grid, start_row: int, start_col: int, visited: set
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    queue = [(start_row, start_col)]
    while queue:
        row, col = queue.pop(0)
        if (row, col) in visited or not (
            0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        ):
            continue
        if grid[row, col] != 0:
            continue
        visited.add((row, col))
        cells.add((row, col))
        queue.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])
    return cells


def _get_zero_shapes(grid: Grid) -> list[Shape]:
    visited: set[tuple[int, int]] = set()
    zero_shapes = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] != 0 or (start_row, start_col) in visited:
                continue
            cells = _collect_zero_cells(grid, start_row, start_col, visited)
            zero_shapes.append(_make_shape(cells, 0))

    return zero_shapes


def _get_enclosed_shapes(
    shapes: list[Shape], grid_shape: tuple[int, int]
) -> list[Shape]:
    max_row, max_col = grid_shape[0] - 1, grid_shape[1] - 1
    enclosed_shapes = []
    for shape in shapes:
        if all(0 < row < max_row and 0 < col < max_col for row, col in shape.cells):
            enclosed_shapes.append(shape)
    return enclosed_shapes


def _get_non_enclosed_shapes(
    shapes: list[Shape], grid_shape: tuple[int, int]
) -> list[Shape]:
    max_row, max_col = grid_shape[0] - 1, grid_shape[1] - 1
    non_enclosed_shapes = []
    for shape in shapes:
        if any(
            row == 0 or col == 0 or row == max_row or col == max_col
            for row, col in shape.cells
        ):
            non_enclosed_shapes.append(shape)
    return non_enclosed_shapes


def get_square_abstraction(shapes: list[Shape]) -> Shape:
    shapes_by_color: dict[int, list[Shape]] = {}

    for shape in shapes:
        if shape.color not in shapes_by_color:
            shapes_by_color[shape.color] = []
        shapes_by_color[shape.color].append(shape)

    for color, shapes in shapes_by_color.items():
        if len(shapes) == 4 and all(
            shape.width == 1 and shape.height == 1 for shape in shapes
        ):
            rows = sorted(shape.row for shape in shapes)
            cols = sorted(shape.col for shape in shapes)

            if (
                rows[0] == rows[1]
                and rows[2] == rows[3]
                and cols[0] == cols[1]
                and cols[2] == cols[3]
            ):
                return Shape(
                    row=rows[0],
                    col=cols[0],
                    width=cols[2] - cols[0] + 1,
                    height=rows[2] - rows[0] + 1,
                    cells=set((row, col) for row in rows for col in cols),
                    is_square_abstraction=True,
                    color=color,
                )


def get_two_by_two_uni_ray_direction_by_color(
    input_shapes: list[Shape], output_grid: Grid
) -> dict[int, DiagonalDirection | None]:
    direction_by_color: dict[int, DiagonalDirection | None] = {}
    for shape in input_shapes:
        if shape.is_two_by_two:
            color = shape.color
            directions_info = [
                (DiagonalDirection.TL, (-1, -1)),
                (DiagonalDirection.TR, (-1, 1)),
                (DiagonalDirection.BL, (1, -1)),
                (DiagonalDirection.BR, (1, 1)),
            ]
            for direction, (offset_row, offset_col) in directions_info:
                start_row = shape.row + offset_row
                start_col = shape.col + offset_col
                if _check_ray_from_location(
                    start_row, start_col, direction, color, output_grid
                ):
                    if color in direction_by_color:
                        direction_by_color[color] = None
                    else:
                        direction_by_color[color] = direction

    return direction_by_color


def _check_ray_from_location(
    start_row: int, start_col: int, direction: DiagonalDirection, color: int, output_grid: Grid
) -> bool:
    if (
        start_row < 0
        or start_row >= output_grid.shape[0]
        or start_col < 0
        or start_col >= output_grid.shape[1]
    ):
        return False

    while (
        0 <= start_row < output_grid.shape[0] and 0 <= start_col < output_grid.shape[1]
    ):
        if output_grid[start_row, start_col] != color:
            return False
        if direction == DiagonalDirection.TL:
            start_row -= 1
            start_col -= 1
        elif direction == DiagonalDirection.TR:
            start_row -= 1
            start_col += 1
        elif direction == DiagonalDirection.BL:
            start_row += 1
            start_col -= 1
        elif direction == DiagonalDirection.BR:
            start_row += 1
            start_col += 1
    return True


ObservationCheck = Callable[["Observations"], None]


def initialize_observations(examples: list, test_input: Grid) -> "Observations":
    example_observations = [
        ExampleObservations(input_grid=inp, output_grid=out) for inp, out in examples
    ]
    test_observations = ExampleObservations(input_grid=test_input)
    return Observations(
        example_observations=example_observations,
        test_observations=test_observations,
    )


def check_grid_sizes(obs: Observations) -> None:
    examples = obs.example_observations
    obs.grid_size_stays_identical = all(
        ex.input_grid.shape == ex.output_grid.shape for ex in examples
    )
    obs.grid_size_decreases = any(
        ex.input_grid.size > ex.output_grid.size for ex in examples
    )


def collect_shapes(obs: Observations) -> None:
    for ex in obs.example_observations:
        ex.input_shapes = get_shapes(ex.input_grid)
        ex.input_shape_count = len(ex.input_shapes)
        ex.output_shapes = get_shapes(ex.output_grid)
        ex.input_color_strict_shapes = get_color_strict_shapes(ex.input_grid)
        ex.output_color_strict_shapes = get_color_strict_shapes(ex.output_grid)

    test = obs.test_observations
    test.input_shapes = get_shapes(test.input_grid)
    test.input_shape_count = len(test.input_shapes)
    test.input_color_strict_shapes = get_color_strict_shapes(test.input_grid)

    obs.single_shape_everywhere = all(
        len(ex.input_shapes) == 1 and len(ex.output_shapes) == 1
        for ex in obs.example_observations
    )
    obs.all_inputs_empty = all(
        ex.input_shape_count == 0 for ex in obs.example_observations
    )
    obs.shapes_collected = True


def check_output_size_ratio(obs: Observations) -> None:
    for ex in obs.example_observations:
        ex.output_twice_as_large_as_input = (
            ex.output_grid.shape[0] == 2 * ex.input_grid.shape[0]
            and ex.output_grid.shape[1] == 2 * ex.input_grid.shape[1]
        )
    obs.all_outputs_twice_as_large_as_inputs = all(
        ex.output_twice_as_large_as_input for ex in obs.example_observations
    )


def check_color_sets(obs: Observations) -> None:
    single_output_color = None
    first = True

    for ex in obs.example_observations:
        input_colors = set(np.unique(ex.input_grid)) - {0}
        output_colors = set(np.unique(ex.output_grid)) - {0}
        ex.output_colors = output_colors
        ex.output_colors_count = len(output_colors)
        ex.new_output_colors = output_colors - input_colors
        ex.new_output_colors_count = len(ex.new_output_colors)

        if ex.output_colors_count == 1:
            color = next(iter(output_colors))
            if first:
                single_output_color = color
                first = False
            elif single_output_color != color:
                single_output_color = None
        else:
            single_output_color = None
            first = False

    obs.single_output_color = single_output_color
    obs.two_new_output_colors_everywhere = all(
        ex.new_output_colors_count == 2 for ex in obs.example_observations
    )
    examples = obs.example_observations
    obs.consistent_new_output_colors = (
        examples[0].new_output_colors
        if all(ex.new_output_colors == examples[0].new_output_colors for ex in examples)
        else None
    )


def check_zero_shapes(obs: Observations) -> None:
    for ex in obs.example_observations:
        zero_shapes = _get_zero_shapes(ex.input_grid)
        ex.enclosed_zero_shapes = _get_enclosed_shapes(zero_shapes, ex.input_grid.shape)
        ex.non_enclosed_zero_shapes = _get_non_enclosed_shapes(
            zero_shapes, ex.input_grid.shape
        )

    test = obs.test_observations
    test_zero_shapes = _get_zero_shapes(test.input_grid)
    test.enclosed_zero_shapes = _get_enclosed_shapes(
        test_zero_shapes, test.input_grid.shape
    )
    test.non_enclosed_zero_shapes = _get_non_enclosed_shapes(
        test_zero_shapes, test.input_grid.shape
    )

    obs.enclosed_zero_shapes_everywhere = all(
        len(ex.enclosed_zero_shapes) > 0 for ex in obs.example_observations
    )
    obs.non_enclosed_zero_shapes_everywhere = all(
        len(ex.non_enclosed_zero_shapes) > 0 for ex in obs.example_observations
    )


def check_cell_counts(obs: Observations) -> None:
    for ex in obs.example_observations:
        input_colors = set(np.unique(ex.input_grid)) - {0}
        output_colors = set(np.unique(ex.output_grid)) - {0}
        ex.input_cell_count_by_color = {
            color: int(np.sum(ex.input_grid == color)) for color in input_colors
        }
        ex.output_cell_count_by_color = {
            color: int(np.sum(ex.output_grid == color)) for color in output_colors
        }
        ex.cell_count_by_color_identical = (
            ex.input_cell_count_by_color == ex.output_cell_count_by_color
        )

    test = obs.test_observations
    test_input_colors = set(np.unique(test.input_grid)) - {0}
    test.input_cell_count_by_color = {
        color: int(np.sum(test.input_grid == color)) for color in test_input_colors
    }

    obs.cell_count_by_color_identical_everywhere = all(
        ex.cell_count_by_color_identical for ex in obs.example_observations
    )


def check_square_abstraction(obs: Observations) -> None:
    for ex in obs.example_observations:
        ex.input_square_abstraction = get_square_abstraction(ex.input_shapes)
        ex.input_square_abstraction_color = (
            ex.input_square_abstraction.color if ex.input_square_abstraction else None
        )

    test = obs.test_observations
    test.input_square_abstraction = get_square_abstraction(test.input_shapes)
    test.input_square_abstraction_color = (
        test.input_square_abstraction.color if test.input_square_abstraction else None
    )

    obs.input_square_abstraction_everywhere = all(
        ex.input_square_abstraction_color is not None for ex in obs.example_observations
    )


def check_opposing_cells(obs: Observations) -> None:
    for ex in obs.example_observations:
        ex.opposing_same_color_single_cells = _collect_opposing_same_color_single_cells(
            ex.input_shapes
        )

    test = obs.test_observations
    test.opposing_same_color_single_cells = _collect_opposing_same_color_single_cells(
        test.input_shapes
    )

    obs.has_opposing_same_color_single_cells_everywhere = all(
        ex.opposing_same_color_single_cells for ex in obs.example_observations
    ) and bool(test.opposing_same_color_single_cells)


def check_spaceship(obs: Observations) -> None:
    for ex in obs.example_observations:
        ex.spaceship_shape = _check_spaceship_shape(ex.input_shapes, ex.input_grid)

    test = obs.test_observations
    test.spaceship_shape = _check_spaceship_shape(test.input_shapes, test.input_grid)

    obs.has_spaceship_shape_everywhere = (
        all(ex.spaceship_shape is not None for ex in obs.example_observations)
        and test.spaceship_shape is not None
    )


def check_two_by_two_rays(obs: Observations) -> None:
    for ex in obs.example_observations:
        ex.input_only_two_by_twos = all(s.is_two_by_two for s in ex.input_shapes)
        if ex.input_only_two_by_twos:
            ex.two_by_two_uni_ray_direction_by_color = (
                get_two_by_two_uni_ray_direction_by_color(
                    ex.input_shapes, ex.output_grid
                )
            )

    obs.all_inputs_only_two_by_twos = all(
        ex.input_only_two_by_twos for ex in obs.example_observations
    )
    examples = obs.example_observations
    obs.consistent_two_by_two_uni_ray_direction_by_color = (
        examples[0].two_by_two_uni_ray_direction_by_color
        if obs.all_inputs_only_two_by_twos
        and all(
            ex.two_by_two_uni_ray_direction_by_color
            == examples[0].two_by_two_uni_ray_direction_by_color
            for ex in examples
        )
        else None
    )


def check_dividers(obs: Observations) -> None:
    for ex in obs.example_observations:
        input_rows, input_cols = ex.input_grid.shape
        output_rows, output_cols = ex.output_grid.shape

        horizontal_ratio_correct = (
            input_rows == 2 * output_rows + 1 and input_cols == output_cols
        )
        vertical_ratio_correct = (
            input_cols == 2 * output_cols + 1 and input_rows == output_rows
        )

        if horizontal_ratio_correct:
            mid_row = input_rows // 2
            ex.single_horizontal_divider = all(
                ex.input_grid[mid_row, col] != 0 for col in range(input_cols)
            )
        if vertical_ratio_correct:
            mid_col = input_cols // 2
            ex.single_vertical_divider = all(
                ex.input_grid[row, mid_col] != 0 for row in range(input_rows)
            )

    obs.has_single_horizontal_divider_everywhere = all(
        ex.single_horizontal_divider for ex in obs.example_observations
    )
    obs.has_single_vertical_divider_everywhere = all(
        ex.single_vertical_divider for ex in obs.example_observations
    )

    test = obs.test_observations
    test_rows, test_cols = test.input_grid.shape
    if obs.has_single_horizontal_divider_everywhere:
        mid_row = test_rows // 2
        test.single_horizontal_divider = all(
            test.input_grid[mid_row, col] != 0 for col in range(test_cols)
        )
    if obs.has_single_vertical_divider_everywhere:
        mid_col = test_cols // 2
        test.single_vertical_divider = all(
            test.input_grid[row, mid_col] != 0 for row in range(test_rows)
        )


def check_zeroed_color(obs: Observations) -> None:
    input_color_sets = [
        set(np.unique(ex.input_grid)) - {0} for ex in obs.example_observations
    ]
    always_in_input = input_color_sets[0].intersection(*input_color_sets[1:])
    output_color_sets = [
        set(np.unique(ex.output_grid)) - {0} for ex in obs.example_observations
    ]
    always_zeroed = always_in_input - set().union(*output_color_sets)
    obs.input_color_always_zeroed = (
        next(iter(always_zeroed)) if len(always_zeroed) == 1 else None
    )


def _shape_encloses_cells(shape: Shape, grid: Grid) -> bool:
    rows, cols = grid.shape
    shape_cell_set = shape.cells
    visited = set()
    queue = []
    for r in range(rows):
        for c in [0, cols - 1]:
            if (r, c) not in shape_cell_set:
                queue.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if (r, c) not in shape_cell_set:
                queue.append((r, c))
    while queue:
        r, c = queue.pop()
        if (r, c) in visited:
            continue
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        if (r, c) in shape_cell_set:
            continue
        visited.add((r, c))
        queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    return len(visited) < rows * cols - len(shape_cell_set)


def check_enclosing_shapes(obs: Observations) -> None:
    def _check_example(example_observations: ExampleObservations) -> None:
        grid = example_observations.input_grid
        border = np.concatenate(
            [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
        )
        bg_color = int(np.bincount(border).argmax())
        for shape in example_observations.input_color_strict_shapes:
            if shape.color == bg_color:
                continue
            if _shape_encloses_cells(shape, grid):
                example_observations.enclosing_shapes.append(shape)
                for s in example_observations.input_shapes:
                    if s.cells & shape.cells:
                        s.encloses_cells = True
                example_observations.has_enclosing_shapes = True

    for ex in obs.example_observations:
        _check_example(ex)
    _check_example(obs.test_observations)

    if (
        all(ex.has_enclosing_shapes for ex in obs.example_observations)
        and obs.test_observations.has_enclosing_shapes
    ):
        obs.has_enclosing_shapes_everywhere = True
    else:
        obs.has_enclosing_shapes_everywhere = False


def check_output_height_half_of_width(obs: Observations) -> None:
    for ex in obs.example_observations:
        ex.output_height_half_of_width = (
            ex.output_grid.shape[0] * 2 == ex.output_grid.shape[1]
        )
    obs.output_height_half_of_width_everywhere = all(
        ex.output_height_half_of_width for ex in obs.example_observations
    )


OBSERVATION_CHECKS: list[ObservationCheck] = [
    check_grid_sizes,
    collect_shapes,
    check_output_size_ratio,
    check_output_height_half_of_width,
    check_dividers,
    check_color_sets,
    check_zeroed_color,
    check_zero_shapes,
    check_cell_counts,
    check_square_abstraction,
    check_opposing_cells,
    check_spaceship,
    check_two_by_two_rays,
    check_enclosing_shapes,
]
