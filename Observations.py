from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from Enums import DiagonalDirection
from Shapes import (
    Grid,
    Shape,
    Spaceship_Shape,
    get_shapes,
    get_color_strict_shapes,
    get_square_abstraction,
    get_two_by_two_uni_ray_direction_by_color,
    _check_spaceship_shape,
    _collect_opposing_same_color_single_cells,
    _get_zero_shapes,
    _get_enclosed_shapes,
    _get_non_enclosed_shapes,
    _shape_encloses_cells,
)


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


ObservationCheck = Callable[["Observations"], None]


def initialize_observations(examples: list, test_input: Grid) -> Observations:
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
        ex.non_enclosed_zero_shapes = _get_non_enclosed_shapes(zero_shapes, ex.input_grid.shape)

    test = obs.test_observations
    test_zero_shapes = _get_zero_shapes(test.input_grid)
    test.enclosed_zero_shapes = _get_enclosed_shapes(test_zero_shapes, test.input_grid.shape)
    test.non_enclosed_zero_shapes = _get_non_enclosed_shapes(test_zero_shapes, test.input_grid.shape)

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
            ex.two_by_two_uni_ray_direction_by_color = get_two_by_two_uni_ray_direction_by_color(
                ex.input_shapes, ex.output_grid
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

        if input_rows == 2 * output_rows + 1 and input_cols == output_cols:
            mid_row = input_rows // 2
            ex.single_horizontal_divider = all(
                ex.input_grid[mid_row, col] != 0 for col in range(input_cols)
            )
        if input_cols == 2 * output_cols + 1 and input_rows == output_rows:
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
    input_color_sets = [set(np.unique(ex.input_grid)) - {0} for ex in obs.example_observations]
    always_in_input = input_color_sets[0].intersection(*input_color_sets[1:])
    output_color_sets = [set(np.unique(ex.output_grid)) - {0} for ex in obs.example_observations]
    always_zeroed = always_in_input - set().union(*output_color_sets)
    obs.input_color_always_zeroed = next(iter(always_zeroed)) if len(always_zeroed) == 1 else None


def check_enclosing_shapes(obs: Observations) -> None:
    def _check_example(example_observations: ExampleObservations) -> None:
        grid = example_observations.input_grid
        border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
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
