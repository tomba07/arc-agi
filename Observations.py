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
    get_diagonal_shapes,
    get_square_abstraction,
    check_spaceship_shape,
    collect_opposing_same_color_single_cells,
    get_zero_shapes,
    get_enclosed_shapes,
    get_non_enclosed_shapes,
    shape_encloses_cells,
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
    two_by_two_uni_ray_direction_by_color: (
        dict[int, DiagonalDirection | None] | None
    ) = None
    input_cell_count_by_color: dict[int, int] | None = None
    output_cell_count_by_color: dict[int, int] | None = None
    cell_count_by_color_identical: bool | None = None
    opposing_same_color_single_cells: list[tuple[Shape, Shape]] | None = None
    spaceship_shape: Spaceship_Shape | None = None
    output_twice_as_large_as_input: bool = False
    single_horizontal_divider: bool = False
    single_vertical_divider: bool = False
    single_implicit_horizontal_divider: bool = False
    single_implicit_vertical_divider: bool = False
    has_enclosing_shapes: bool = False
    enclosing_shapes: list[Shape] = field(default_factory=list)
    input_color_strict_shapes: list[Shape] | None = None
    output_color_strict_shapes: list[Shape] | None = None
    input_diagonal_shapes: list[Shape] | None = None
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
    consistent_two_by_two_uni_ray_direction_by_color: (
        dict[int, DiagonalDirection | None] | None
    ) = None
    consistent_new_output_colors: set[int] | None = None
    consistent_removed_colors: set[int] | None = None
    two_new_output_colors_everywhere: bool | None = None
    enclosed_zero_shapes_everywhere: bool | None = None
    non_enclosed_zero_shapes_everywhere: bool | None = None
    cell_count_by_color_identical_everywhere: bool | None = None
    has_opposing_same_color_single_cells_everywhere: bool | None = None
    has_spaceship_shape_everywhere: bool | None = None
    all_outputs_twice_as_large_as_inputs: bool | None = None
    has_single_horizontal_divider_everywhere: bool | None = None
    has_single_vertical_divider_everywhere: bool | None = None
    has_single_implicit_horizontal_divider_everywhere: bool | None = None
    has_single_implicit_vertical_divider_everywhere: bool | None = None
    removed_input_color: int | None = None
    has_enclosing_shapes_everywhere: bool | None = None
    output_height_half_of_width_everywhere: bool | None = None
    is_recolor_context: bool | None = None
    only_similar_input_shapes: bool | None = None


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


def check_grid_sizes(observations: Observations) -> None:
    examples = observations.example_observations
    observations.grid_size_stays_identical = all(
        ex.input_grid.shape == ex.output_grid.shape for ex in examples
    )
    observations.grid_size_decreases = any(
        ex.input_grid.size > ex.output_grid.size for ex in examples
    )


def collect_shapes(observations: Observations) -> None:
    for ex in observations.example_observations:
        ex.input_shapes = get_shapes(ex.input_grid)
        ex.input_shape_count = len(ex.input_shapes)
        ex.output_shapes = get_shapes(ex.output_grid)
        ex.input_color_strict_shapes = get_color_strict_shapes(ex.input_grid)
        ex.output_color_strict_shapes = get_color_strict_shapes(ex.output_grid)
        ex.input_diagonal_shapes = get_diagonal_shapes(ex.input_grid)

    test = observations.test_observations
    test.input_shapes = get_shapes(test.input_grid)
    test.input_shape_count = len(test.input_shapes)
    test.input_color_strict_shapes = get_color_strict_shapes(test.input_grid)
    test.input_diagonal_shapes = get_diagonal_shapes(test.input_grid)

    observations.single_shape_everywhere = all(
        len(ex.input_shapes) == 1 and len(ex.output_shapes) == 1
        for ex in observations.example_observations
    )
    observations.all_inputs_empty = all(
        ex.input_shape_count == 0 for ex in observations.example_observations
    )
    observations.shapes_collected = True


def check_output_size_ratio(observations: Observations) -> None:
    for ex in observations.example_observations:
        ex.output_twice_as_large_as_input = (
            ex.output_grid.shape[0] == 2 * ex.input_grid.shape[0]
            and ex.output_grid.shape[1] == 2 * ex.input_grid.shape[1]
        )
    observations.all_outputs_twice_as_large_as_inputs = all(
        ex.output_twice_as_large_as_input for ex in observations.example_observations
    )


def check_color_sets(observations: Observations) -> None:
    single_output_color = None
    first = True

    for ex in observations.example_observations:
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

    observations.single_output_color = single_output_color
    observations.two_new_output_colors_everywhere = all(
        ex.new_output_colors_count == 2 for ex in observations.example_observations
    )
    examples = observations.example_observations
    observations.consistent_new_output_colors = (
        examples[0].new_output_colors
        if all(ex.new_output_colors == examples[0].new_output_colors for ex in examples)
        else None
    )
    removed_per_example = [
        set(np.unique(ex.input_grid)) - set(np.unique(ex.output_grid))
        for ex in examples
    ]
    observations.consistent_removed_colors = removed_per_example[0].intersection(
        *removed_per_example[1:]
    )


def check_zero_shapes(observations: Observations) -> None:
    for ex in observations.example_observations:
        zero_shapes = get_zero_shapes(ex.input_grid)
        ex.enclosed_zero_shapes = get_enclosed_shapes(zero_shapes, ex.input_grid.shape)
        ex.non_enclosed_zero_shapes = get_non_enclosed_shapes(
            zero_shapes, ex.input_grid.shape
        )

    test = observations.test_observations
    test_zero_shapes = get_zero_shapes(test.input_grid)
    test.enclosed_zero_shapes = get_enclosed_shapes(
        test_zero_shapes, test.input_grid.shape
    )
    test.non_enclosed_zero_shapes = get_non_enclosed_shapes(
        test_zero_shapes, test.input_grid.shape
    )

    observations.enclosed_zero_shapes_everywhere = all(
        len(ex.enclosed_zero_shapes) > 0 for ex in observations.example_observations
    )
    observations.non_enclosed_zero_shapes_everywhere = all(
        len(ex.non_enclosed_zero_shapes) > 0 for ex in observations.example_observations
    )


def check_cell_counts(observations: Observations) -> None:
    for ex in observations.example_observations:
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

    test = observations.test_observations
    test_input_colors = set(np.unique(test.input_grid)) - {0}
    test.input_cell_count_by_color = {
        color: int(np.sum(test.input_grid == color)) for color in test_input_colors
    }

    observations.cell_count_by_color_identical_everywhere = all(
        ex.cell_count_by_color_identical for ex in observations.example_observations
    )


def check_square_abstraction(observations: Observations) -> None:
    for ex in observations.example_observations:
        ex.input_square_abstraction = get_square_abstraction(ex.input_shapes)
        ex.input_square_abstraction_color = (
            ex.input_square_abstraction.color if ex.input_square_abstraction else None
        )

    test = observations.test_observations
    test.input_square_abstraction = get_square_abstraction(test.input_shapes)
    test.input_square_abstraction_color = (
        test.input_square_abstraction.color if test.input_square_abstraction else None
    )

    observations.input_square_abstraction_everywhere = all(
        ex.input_square_abstraction_color is not None
        for ex in observations.example_observations
    )


def check_opposing_cells(observations: Observations) -> None:
    for ex in observations.example_observations:
        ex.opposing_same_color_single_cells = collect_opposing_same_color_single_cells(
            ex.input_shapes
        )

    test = observations.test_observations
    test.opposing_same_color_single_cells = collect_opposing_same_color_single_cells(
        test.input_shapes
    )

    observations.has_opposing_same_color_single_cells_everywhere = all(
        ex.opposing_same_color_single_cells for ex in observations.example_observations
    ) and bool(test.opposing_same_color_single_cells)


def check_spaceship(observations: Observations) -> None:
    for ex in observations.example_observations:
        ex.spaceship_shape = check_spaceship_shape(ex.input_shapes, ex.input_grid)

    test = observations.test_observations
    test.spaceship_shape = check_spaceship_shape(test.input_shapes, test.input_grid)

    observations.has_spaceship_shape_everywhere = (
        all(ex.spaceship_shape is not None for ex in observations.example_observations)
        and test.spaceship_shape is not None
    )


def _check_ray_from_location(
    start_row: int,
    start_col: int,
    direction: DiagonalDirection,
    color: int,
    output_grid: Grid,
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


def get_two_by_two_uni_ray_direction_by_color(
    input_shapes: list[Shape], output_grid: Grid
) -> dict[int, DiagonalDirection | None]:
    direction_by_color: dict[int, DiagonalDirection | None] = {}
    for shape in input_shapes:
        if shape.is_two_by_two:
            color = shape.color
            for direction, (offset_row, offset_col) in [
                (DiagonalDirection.TL, (-1, -1)),
                (DiagonalDirection.TR, (-1, +1)),
                (DiagonalDirection.BL, (+1, -1)),
                (DiagonalDirection.BR, (+1, +1)),
            ]:
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


def check_two_by_two_rays(observations: Observations) -> None:
    for ex in observations.example_observations:
        ex.input_only_two_by_twos = all(s.is_two_by_two for s in ex.input_shapes)
        if ex.input_only_two_by_twos:
            ex.two_by_two_uni_ray_direction_by_color = (
                get_two_by_two_uni_ray_direction_by_color(
                    ex.input_shapes, ex.output_grid
                )
            )

    observations.all_inputs_only_two_by_twos = all(
        ex.input_only_two_by_twos for ex in observations.example_observations
    )
    examples = observations.example_observations
    observations.consistent_two_by_two_uni_ray_direction_by_color = (
        examples[0].two_by_two_uni_ray_direction_by_color
        if observations.all_inputs_only_two_by_twos
        and all(
            ex.two_by_two_uni_ray_direction_by_color
            == examples[0].two_by_two_uni_ray_direction_by_color
            for ex in examples
        )
        else None
    )


def check_dividers(observations: Observations) -> None:
    for ex in observations.example_observations:
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

    observations.has_single_horizontal_divider_everywhere = all(
        ex.single_horizontal_divider for ex in observations.example_observations
    )
    observations.has_single_vertical_divider_everywhere = all(
        ex.single_vertical_divider for ex in observations.example_observations
    )

    test = observations.test_observations
    test_rows, test_cols = test.input_grid.shape
    if observations.has_single_horizontal_divider_everywhere:
        mid_row = test_rows // 2
        test.single_horizontal_divider = all(
            test.input_grid[mid_row, col] != 0 for col in range(test_cols)
        )
    if observations.has_single_vertical_divider_everywhere:
        mid_col = test_cols // 2
        test.single_vertical_divider = all(
            test.input_grid[row, mid_col] != 0 for row in range(test_rows)
        )


def check_removed_color(observations: Observations) -> None:
    colors_in_every_input: set[int] | None = None
    for example in observations.example_observations:
        input_colors = set(np.unique(example.input_grid)) - {0}
        if colors_in_every_input is None:
            colors_in_every_input = input_colors
        else:
            colors_in_every_input &= input_colors

    colors_in_any_output: set[int] = set()
    for example in observations.example_observations:
        colors_in_any_output |= set(np.unique(example.output_grid)) - {0}

    always_removed = (colors_in_every_input or set()) - colors_in_any_output
    observations.removed_input_color = (
        next(iter(always_removed)) if len(always_removed) == 1 else None
    )


def check_enclosing_shapes(observations: Observations) -> None:
    def _check_example(example_observations: ExampleObservations) -> None:
        grid = example_observations.input_grid
        border = np.concatenate(
            [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
        )
        bg_color = int(np.bincount(border).argmax())
        for shape in example_observations.input_color_strict_shapes:
            if shape.color == bg_color:
                continue
            enclosed_cells = shape_encloses_cells(shape, grid)
            if enclosed_cells:
                shape.enclosed_cells = enclosed_cells
                example_observations.enclosing_shapes.append(shape)
                for s in example_observations.input_shapes:
                    if s.cells & shape.cells:
                        s.encloses_cells = True
                example_observations.has_enclosing_shapes = True

    for ex in observations.example_observations:
        _check_example(ex)
    _check_example(observations.test_observations)

    if (
        all(ex.has_enclosing_shapes for ex in observations.example_observations)
        and observations.test_observations.has_enclosing_shapes
    ):
        observations.has_enclosing_shapes_everywhere = True
    else:
        observations.has_enclosing_shapes_everywhere = False


def check_output_height_half_of_width(observations: Observations) -> None:
    for ex in observations.example_observations:
        ex.output_height_half_of_width = (
            ex.output_grid.shape[0] * 2 == ex.output_grid.shape[1]
        )
    observations.output_height_half_of_width_everywhere = all(
        ex.output_height_half_of_width for ex in observations.example_observations
    )


def check_recolor_context(observations: Observations) -> None:
    observations.is_recolor_context = (
        bool(observations.grid_size_stays_identical and observations.shapes_collected)
        and not observations.cell_count_by_color_identical_everywhere
    )


def check_implicit_color_dividers(observations: Observations) -> None:
    def colors_in_grid(grid: np.ndarray) -> set[int]:
        return set(np.unique(grid)) - {0}

    for example in observations.example_observations:
        grid = example.input_grid
        rowCount, colCount = example.input_grid.shape

        # we only check the middle divisions
        # horizontal divider check
        if rowCount % 2 == 0:
            mid = rowCount // 2
            top = grid[:mid]
            bottom = grid[mid:]

            top_colors = colors_in_grid(top)
            bottom_colors = colors_in_grid(bottom)

            if top_colors.isdisjoint(bottom_colors):
                example.single_implicit_horizontal_divider = True

        # vertical divider check
        if colCount % 2 == 0:
            mid = colCount // 2
            left = grid[:, :mid]
            right = grid[:, mid:]

            left_colors = colors_in_grid(left)
            right_colors = colors_in_grid(right)

            if left_colors.isdisjoint(right_colors):
                example.single_implicit_vertical_divider = True

    # set everywhere flags
    observations.has_single_implicit_horizontal_divider_everywhere = all(
        ex.single_implicit_horizontal_divider
        for ex in observations.example_observations
    )
    observations.has_single_implicit_vertical_divider_everywhere = all(
        ex.single_implicit_vertical_divider for ex in observations.example_observations
    )
    if observations.has_single_implicit_horizontal_divider_everywhere:
        observations.test_observations.single_implicit_horizontal_divider = True
    if observations.has_single_implicit_vertical_divider_everywhere:
        observations.test_observations.single_implicit_vertical_divider = True


def check_only_similar_input_shapes(observations: Observations) -> None:
    if not observations.shapes_collected:
        collect_shapes(observations)

    all_shapes = []

    observations.only_similar_input_shapes = False

    for example in observations.example_observations:
        for shape in example.input_diagonal_shapes:
            if shape.color != 0 and (shape.width > 1 or shape.height > 1):
                all_shapes.append(shape)

    if len(all_shapes) > 0:
        first_shape = all_shapes[0]
        if all(first_shape.is_similar_to(shape) for shape in all_shapes):
            observations.only_similar_input_shapes = True
