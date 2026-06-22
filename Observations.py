from dataclasses import dataclass

import numpy as np

Grid = np.ndarray


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
    color: int = None
    is_two_by_two: bool = False


@dataclass
class ExampleObservations:
    input_grid: Grid
    input_shape_count: int
    enclosed_zero_shapes: list[Shape]
    non_enclosed_zero_shapes: list[Shape]
    output_colors: set[int]
    output_colors_count: int
    new_output_colors: set[int]
    new_output_colors_count: int
    input_shapes: list[Shape]
    output_shapes: list[Shape]
    output_grid: Grid = None
    input_square_abstraction: Shape = None
    input_square_abstraction_color: int = None
    input_only_two_by_twos: bool = False
    two_by_two_uni_ray_direction_by_color: dict[int, str] = None
    input_cell_count_by_color: dict[int, int] = None
    output_cell_count_by_color: dict[int, int] = None
    cell_count_by_color_identical: bool = None


@dataclass
class Observations:
    grid_size_stays_identical: bool
    grid_size_decreases: bool
    single_shape_everywhere: bool
    all_inputs_empty: bool = False
    single_output_color: int = None
    example_observations: list[ExampleObservations] = None
    test_observations: ExampleObservations = None
    input_square_abstraction_everywhere: bool = False
    all_inputs_only_two_by_twos: bool = False
    consistent_two_by_two_uni_ray_direction_by_color: dict[int, str] = None
    consistent_new_output_colors: list[int] = None
    has_two_new_output_colors: bool = False
    two_new_output_colors_everywhere: bool = False
    enclosed_zero_shapes_everywhere: bool = False
    non_enclosed_zero_shapes_everywhere: bool = False
    cell_count_by_color_identical_everywhere: bool = None


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
    min_row = min(row for row, col in cells)
    max_row = max(row for row, col in cells)
    min_col = min(col for row, col in cells)
    max_col = max(col for row, col in cells)
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

def _get_enclosed_shapes(shapes: list[Shape], grid_shape: tuple[int, int]) -> list[Shape]:
    max_row, max_col = grid_shape[0] - 1, grid_shape[1] - 1
    enclosed_shapes = []
    for shape in shapes:
        if all(
            0 < row < max_row and 0 < col < max_col
            for row, col in shape.cells
        ):
            enclosed_shapes.append(shape)
    return enclosed_shapes


def _get_non_enclosed_shapes(shapes: list[Shape], grid_shape: tuple[int, int]) -> list[Shape]:
    max_row, max_col = grid_shape[0] - 1, grid_shape[1] - 1
    non_enclosed_shapes = []
    for shape in shapes:
        if any(
            row == 0 or col == 0 or row == max_row or col == max_col
            for row, col in shape.cells
        ):
            non_enclosed_shapes.append(shape)
    return non_enclosed_shapes


# For now, we are very picky about the square abstraction.
# We only consider a single one. Also it has the have exactly 4 "1x1" shapes of same color with proper alignment.
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
) -> dict[int, str]:
    direction_by_color: dict[int, str] = {}
    for shape in input_shapes:
        if shape.is_two_by_two:
            color = shape.color
            # Check rays in all four diagonal directions
            directions_info = [
                {"dir": "tl", "offset": (-1, -1)},
                {"dir": "tr", "offset": (-1, 1)},
                {"dir": "bl", "offset": (1, -1)},
                {"dir": "br", "offset": (1, 1)},
            ]
            for direction_info in directions_info:
                direction = direction_info["dir"]
                offset_row, offset_col = direction_info["offset"]
                start_row = shape.row + offset_row
                start_col = shape.col + offset_col
                if _check_ray_from_location(
                    start_row, start_col, direction, color, output_grid
                ):
                    if color in direction_by_color:
                        # If we already have a direction for this color, it means it's not unique
                        direction_by_color[color] = None
                    else:
                        direction_by_color[color] = direction

    return direction_by_color


def _check_ray_from_location(
    start_row: int, start_col: int, direction: str, color: int, output_grid: Grid
) -> str | None:
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
    return True


def observe(examples: list, test_input: Grid) -> Observations:
    example_observations = []
    single_output_color = None

    for input_grid, output_grid in examples:
        input_shapes = get_shapes(input_grid)
        zero_shapes = _get_zero_shapes(input_grid)
        enclosed_zero_shapes = _get_enclosed_shapes(zero_shapes, input_grid.shape)
        non_enclosed_zero_shapes = _get_non_enclosed_shapes(zero_shapes, input_grid.shape)
        input_square_abstraction = get_square_abstraction(input_shapes)
        input_colors = set(np.unique(input_grid)) - {0}
        output_colors = set(np.unique(output_grid)) - {0}
        output_colors_count = len(output_colors)
        new_output_colors = output_colors - input_colors
        new_output_colors_count = len(new_output_colors)
        all_inputs_only_two_by_twos = all(shape.is_two_by_two for shape in input_shapes)
        input_cell_count_by_color = {color: np.sum(input_grid == color) for color in input_colors}
        output_cell_count_by_color = {color: np.sum(output_grid == color) for color in output_colors}   
        cell_count_by_color_identical = input_cell_count_by_color == output_cell_count_by_color

        if input_square_abstraction:
            input_square_abstraction_color = input_square_abstraction.color

        if output_colors_count == 1:
            if single_output_color is None:
                single_output_color = output_colors.pop()
            elif single_output_color != output_colors.pop():
                single_output_color = None

        if all_inputs_only_two_by_twos:
            two_by_two_uni_ray_direction_by_color = (
                get_two_by_two_uni_ray_direction_by_color(input_shapes, output_grid)
            )

        example_observations.append(
            ExampleObservations(
                input_grid=input_grid,
                output_grid=output_grid,
                enclosed_zero_shapes=enclosed_zero_shapes,
                non_enclosed_zero_shapes=non_enclosed_zero_shapes,
                input_square_abstraction=input_square_abstraction,
                input_square_abstraction_color=input_square_abstraction_color
                if input_square_abstraction
                else None,
                input_shape_count=len(input_shapes),
                output_colors=output_colors,
                output_colors_count=output_colors_count,
                new_output_colors=new_output_colors,
                new_output_colors_count=new_output_colors_count,
                input_shapes=input_shapes,
                output_shapes=get_shapes(output_grid),
                input_only_two_by_twos=all_inputs_only_two_by_twos,
                two_by_two_uni_ray_direction_by_color=two_by_two_uni_ray_direction_by_color
                if all_inputs_only_two_by_twos
                else None,
                input_cell_count_by_color=input_cell_count_by_color,
                output_cell_count_by_color=output_cell_count_by_color,
                cell_count_by_color_identical=cell_count_by_color_identical,
            )
        )

    test_shapes = get_shapes(test_input)
    test_input_square_abstraction = get_square_abstraction(test_shapes)
    test_zero_shapes = _get_zero_shapes(test_input)
    test_input_colors = set(np.unique(test_input)) - {0}
    test_input_cell_count_by_color = {color: int(np.sum(test_input == color)) for color in test_input_colors}

    test_observations = ExampleObservations(
        input_grid=test_input,
        input_shape_count=len(test_shapes),
        enclosed_zero_shapes=_get_enclosed_shapes(test_zero_shapes, test_input.shape),
        non_enclosed_zero_shapes=_get_non_enclosed_shapes(test_zero_shapes, test_input.shape),
        output_colors=set(),
        output_colors_count=0,
        new_output_colors=set(),
        new_output_colors_count=0,
        input_shapes=test_shapes,
        output_shapes=[],
        input_square_abstraction=test_input_square_abstraction,
        input_square_abstraction_color=test_input_square_abstraction.color
        if test_input_square_abstraction
        else None,
        input_cell_count_by_color=test_input_cell_count_by_color,
    )

    return Observations(
        input_square_abstraction_everywhere=all(
            obs.input_square_abstraction_color is not None
            for obs in example_observations
        ),
        all_inputs_empty=all(
            example_observations[i].input_shape_count == 0 for i in range(len(examples))
        ),
        single_output_color=single_output_color,
        grid_size_stays_identical=all(
            input_grid.shape == output_grid.shape
            for input_grid, output_grid in examples
        ),
        grid_size_decreases=any(
            input_grid.size > output_grid.size for input_grid, output_grid in examples
        ),
        single_shape_everywhere=all(
            len(obs.input_shapes) == 1 and len(obs.output_shapes) == 1
            for obs in example_observations
        ),
        all_inputs_only_two_by_twos=all(
            obs.input_only_two_by_twos for obs in example_observations
        ),
        consistent_two_by_two_uni_ray_direction_by_color=(
            example_observations[0].two_by_two_uni_ray_direction_by_color
            if all(
                obs.two_by_two_uni_ray_direction_by_color
                == example_observations[0].two_by_two_uni_ray_direction_by_color
                for obs in example_observations
            )
            else None
        ),
        consistent_new_output_colors=(
            example_observations[0].new_output_colors
            if all(
                obs.new_output_colors == example_observations[0].new_output_colors
                for obs in example_observations
            )
            else None
        ),
        two_new_output_colors_everywhere=all(
            obs.new_output_colors_count == 2 for obs in example_observations
        ),
        enclosed_zero_shapes_everywhere=all(len(obs.enclosed_zero_shapes) > 0 for obs in example_observations),
        non_enclosed_zero_shapes_everywhere=all(len(obs.non_enclosed_zero_shapes) > 0 for obs in example_observations),
        example_observations=example_observations,
        test_observations=test_observations,
        cell_count_by_color_identical_everywhere=all(
            obs.cell_count_by_color_identical for obs in example_observations
        ),
    )
