from dataclasses import dataclass

import numpy as np

Grid = np.ndarray


@dataclass
class Shape:
    row: int
    col: int
    width: int
    height: int
    cells: frozenset[tuple[int, int]]


@dataclass
class ExampleObservations:
    input_shape_count: int
    output_colors: set[int]
    output_colors_count: int
    input_shapes: list[Shape]
    output_shapes: list[Shape]


@dataclass
class Observations:
    grid_size_stays_identical: bool
    grid_size_decreases: bool
    single_shape_everywhere: bool
    all_inputs_empty: bool = False
    single_output_color: int = None
    example_observations: list[ExampleObservations] = None
    test_observations: ExampleObservations = None


def _collect_cells(
    grid: Grid, start_row: int, start_col: int, visited: set
) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    queue = [(start_row, start_col)]
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
    return frozenset(cells)


def _make_shape(cells: frozenset[tuple[int, int]]) -> Shape:
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
    )


def get_shapes(grid: Grid) -> list[Shape]:
    visited: set[tuple[int, int]] = set()
    shapes = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] == 0 or (start_row, start_col) in visited:
                continue
            cells = _collect_cells(grid, start_row, start_col, visited)
            shapes.append(_make_shape(cells))
    return shapes


def observe(examples: list) -> Observations:
    example_observations = []
    single_output_color = None

    for input_grid, output_grid in examples:
        input_shapes = get_shapes(input_grid)
        output_colors = set(np.unique(output_grid)) - {0}
        output_colors_count = len(output_colors)

        if output_colors_count == 1:
            if single_output_color is None:
                single_output_color = output_colors.pop()
            elif single_output_color != output_colors.pop():
                single_output_color = None

        example_observations.append(
            ExampleObservations(
                input_shape_count=len(input_shapes),
                output_colors=output_colors,
                output_colors_count=output_colors_count,
                input_shapes=input_shapes,
                output_shapes=get_shapes(output_grid),
            )
        )

    return Observations(
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
    )
