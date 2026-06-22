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


@dataclass
class ExampleObservations:
    input_grid: Grid
    input_shape_count: int
    output_colors: set[int]
    output_colors_count: int
    input_shapes: list[Shape]
    output_shapes: list[Shape]
    output_grid: Grid = None
    input_square_abstraction: Shape = None
    input_square_abstraction_color: int = None


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
    return Shape(
        row=min_row,
        col=min_col,
        width=max_col - min_col + 1,
        height=max_row - min_row + 1,
        cells=cells,
        color=color,
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


def observe(examples: list, test_input: Grid) -> Observations:
    example_observations = []
    single_output_color = None

    for input_grid, output_grid in examples:
        input_shapes = get_shapes(input_grid)
        input_square_abstraction = get_square_abstraction(input_shapes)
        output_colors = set(np.unique(output_grid)) - {0}
        output_colors_count = len(output_colors)

        if input_square_abstraction:
            input_square_abstraction_color = input_square_abstraction.color

        if output_colors_count == 1:
            if single_output_color is None:
                single_output_color = output_colors.pop()
            elif single_output_color != output_colors.pop():
                single_output_color = None

        example_observations.append(
            ExampleObservations(
                input_grid=input_grid,
                output_grid=output_grid,
                input_square_abstraction=input_square_abstraction,
                input_square_abstraction_color=input_square_abstraction_color
                if input_square_abstraction
                else None,
                input_shape_count=len(input_shapes),
                output_colors=output_colors,
                output_colors_count=output_colors_count,
                input_shapes=input_shapes,
                output_shapes=get_shapes(output_grid),
            )
        )

    test_shapes = get_shapes(test_input)
    test_input_square_abstraction = get_square_abstraction(test_shapes)
    
    if test_input_square_abstraction:
        
        test_input_square_abstraction_color = test_input_square_abstraction.color
        
    test_observations = ExampleObservations(
        input_grid=test_input,
        input_square_abstraction=test_input_square_abstraction,
        input_square_abstraction_color=test_input_square_abstraction_color
        if test_input_square_abstraction
        else None,
        input_shape_count=len(test_shapes),
        output_colors=set(),
        output_colors_count=0,
        input_shapes=test_shapes,
        output_shapes=[],
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
        example_observations=example_observations,
        test_observations=test_observations,
    )
