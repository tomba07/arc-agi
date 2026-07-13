from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from Enums import AxisDirection, DiagonalDirection
from Shapes import (
    Grid,
    Shape,
    Spaceship_Shape,
    get_shapes,
    get_color_strict_shapes,
    get_diagonal_shapes,
    get_color_strict_diagonal_shapes,
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
    input_colors: set[int] | None = None
    output_colors: set[int] | None = None
    output_colors_count: int = 0
    new_output_colors: set[int] | None = None
    new_output_colors_count: int = 0
    input_shapes: list[Shape] | None = None
    output_shapes: list[Shape] | None = None
    input_square_abstraction: Shape | None = None
    input_square_abstraction_color: int | None = None
    input_only_two_by_twos: bool = False
    input_only_one_by_ones: bool = False
    two_by_two_uni_ray_direction_by_color: (
        dict[int, DiagonalDirection | None] | None
    ) = None
    input_cell_count_by_color: dict[int, int] | None = None
    output_cell_count_by_color: dict[int, int] | None = None
    cell_count_by_color_identical: bool | None = None
    opposing_same_color_single_cells: list[tuple[Shape, Shape]] | None = None
    spaceship_shape: Spaceship_Shape | None = None
    output_twice_as_large_as_input: bool = False
    output_thrice_as_large_as_input: bool = False
    single_horizontal_divider: bool = False
    two_horizontal_dividers: bool = False
    single_vertical_divider: bool = False
    two_vertical_dividers: bool = False
    single_implicit_horizontal_divider: bool = False
    single_implicit_vertical_divider: bool = False
    has_enclosing_shapes: bool = False
    enclosing_shapes: list[Shape] = field(default_factory=list)
    input_color_strict_shapes: list[Shape] | None = None
    output_color_strict_shapes: list[Shape] | None = None
    input_diagonal_shapes: list[Shape] | None = None
    output_diagonal_shapes: list[Shape] | None = None
    input_color_strict_diagonal_shapes: list[Shape] | None = None
    output_color_strict_diagonal_shapes: list[Shape] | None = None
    bottom_gaps: list[tuple[int, int, int]] | None = None
    output_height_half_of_width: bool = False
    has_single_enclosed_shape: bool = None
    input_has_single_one_by_one_shape: bool = None
    output_has_single_one_by_one_shape: bool = None
    has_four_walls: bool = None
    attracted_color: int | None = None
    has_four_horizontally_aligned_shapes: bool = None
    has_four_vertically_aligned_shapes: bool = None
    has_two_single_cells_on_rim: bool = None


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
    all_inputs_only_one_by_ones: bool | None = None
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
    all_outputs_thrice_as_large_as_inputs: bool | None = None
    has_single_horizontal_divider_everywhere: bool | None = None
    has_two_horizontal_dividers_everywhere: bool | None = None
    has_single_vertical_divider_everywhere: bool | None = None
    has_two_vertical_dividers_everywhere: bool | None = None
    has_single_implicit_horizontal_divider_everywhere: bool | None = None
    has_single_implicit_vertical_divider_everywhere: bool | None = None
    removed_input_color: int | None = None
    has_enclosing_shapes_everywhere: bool | None = None
    output_height_half_of_width_everywhere: bool | None = None
    is_recolor_context: bool | None = None
    only_similar_input_shapes: bool | None = None
    bottom_gaps_everywhere: bool | None = None
    two_by_twos_everywhere: bool | None = None
    single_non_by_two_shape_everywhere: bool | None = None
    has_single_enclosed_shape_everywhere: bool | None = None
    input_has_single_one_by_one_shape_everywhere: bool = None
    output_has_single_one_by_one_shape_everywhere: bool = None
    has_four_walls_everywhere: bool | None = None
    has_four_horizontally_aligned_shapes_everywhere: bool | None = None
    has_four_vertically_aligned_shapes_everywhere: bool | None = None
    has_four_aligned_shapes_everywhere: bool | None = None
    has_two_single_cells_on_rim_everywhere: bool | None = None
    consistent_output_grid_size: tuple[int, int] | None = None


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
        ex.output_diagonal_shapes = get_diagonal_shapes(ex.output_grid)
        ex.input_color_strict_diagonal_shapes = get_color_strict_diagonal_shapes(
            ex.input_grid
        )
        ex.output_color_strict_diagonal_shapes = get_color_strict_diagonal_shapes(
            ex.output_grid
        )
        ex.input_has_single_one_by_one_shape = (
            len(ex.input_shapes) == 1 and ex.input_shapes[0].is_one_by_one
        )
        ex.output_has_single_one_by_one_shape = (
            len(ex.output_shapes) == 1 and ex.output_shapes[0].is_one_by_one
        )
        ex.all_input_shapes_are_one_by_one = all(
            shape.is_one_by_one for shape in ex.input_shapes
        )

    test = observations.test_observations
    test.input_shapes = get_shapes(test.input_grid)
    test.input_shape_count = len(test.input_shapes)
    test.input_color_strict_shapes = get_color_strict_shapes(test.input_grid)
    test.input_diagonal_shapes = get_diagonal_shapes(test.input_grid)
    test.input_color_strict_diagonal_shapes = get_color_strict_diagonal_shapes(
        test.input_grid
    )
    test.input_has_single_one_by_one_shape = (
        len(test.input_shapes) == 1 and test.input_shapes[0].is_one_by_one
    )
    test.all_input_shapes_are_one_by_one = all(
        shape.is_one_by_one for shape in test.input_shapes
    )

    observations.single_shape_everywhere = all(
        len(ex.input_shapes) == 1 and len(ex.output_shapes) == 1
        for ex in observations.example_observations
    )
    observations.all_inputs_empty = all(
        ex.input_shape_count == 0 for ex in observations.example_observations
    )
    observations.shapes_collected = True
    observations.two_by_twos_everywhere = all(
        any(s.is_two_by_two for s in ex.input_shapes)
        for ex in observations.example_observations
    )
    # all shapes except one are not 2x2 shapes, and that one shape is not a 2x2 shape
    observations.single_non_by_two_shape_everywhere = all(
        sum(not s.is_two_by_two for s in ex.input_shapes) == 1
        for ex in observations.example_observations
    )
    observations.input_has_single_one_by_one_shape_everywhere = (
        all(
            ex.input_has_single_one_by_one_shape
            for ex in observations.example_observations
        )
        and observations.test_observations.input_has_single_one_by_one_shape
    )
    observations.output_has_single_one_by_one_shape_everywhere = all(
        ex.output_has_single_one_by_one_shape
        for ex in observations.example_observations
    )
    observations.all_inputs_only_one_by_ones = all(
        ex.all_input_shapes_are_one_by_one for ex in observations.example_observations
    )


def check_output_size_ratio(observations: Observations) -> None:
    for ex in observations.example_observations:
        ex.output_twice_as_large_as_input = (
            ex.output_grid.shape[0] == 2 * ex.input_grid.shape[0]
            and ex.output_grid.shape[1] == 2 * ex.input_grid.shape[1]
        )
        ex.output_thrice_as_large_as_input = (
            ex.output_grid.shape[0] == 3 * ex.input_grid.shape[0]
            and ex.output_grid.shape[1] == 3 * ex.input_grid.shape[1]
        )
    observations.all_outputs_twice_as_large_as_inputs = all(
        ex.output_twice_as_large_as_input for ex in observations.example_observations
    )
    observations.all_outputs_thrice_as_large_as_inputs = all(
        ex.output_thrice_as_large_as_input for ex in observations.example_observations
    )


def check_color_sets(observations: Observations) -> None:
    single_output_color = None
    first = True

    for ex in observations.example_observations:
        input_colors = set(np.unique(ex.input_grid)) - {0}
        output_colors = set(np.unique(ex.output_grid)) - {0}
        ex.input_colors = input_colors
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


def check_two_dividers(observations: Observations) -> None:
    for ex in observations.example_observations:
        input_rows, input_cols = ex.input_grid.shape
        output_rows, output_cols = ex.output_grid.shape

        # check horizontal dividers
        if input_rows == 3 * output_rows + 2 and input_cols == output_cols:
            first_row = input_rows // 3
            second_row = 2 * input_rows // 3
            ex.two_horizontal_dividers = all(
                ex.input_grid[first_row, col] != 0
                and ex.input_grid[second_row, col] != 0
                for col in range(input_cols)
            )

        # check vertical dividers
        if input_cols == 3 * output_cols + 2 and input_rows == output_rows:
            first_col = input_cols // 3
            second_col = 2 * input_cols // 3
            ex.two_vertical_dividers = all(
                ex.input_grid[row, first_col] != 0
                and ex.input_grid[row, second_col] != 0
                for row in range(input_rows)
            )

    observations.has_two_horizontal_dividers_everywhere = all(
        ex.two_horizontal_dividers for ex in observations.example_observations
    )
    observations.has_two_vertical_dividers_everywhere = all(
        ex.two_vertical_dividers for ex in observations.example_observations
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
            if shape.width < 3 or shape.height < 3:
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


def check_bottom_gaps(observations: Observations) -> None:
    if not observations.shapes_collected:
        collect_shapes(observations)

    def collect_bottom_gaps(example: ExampleObservations) -> None:
        rows, cols = example.input_grid.shape
        max_row = rows - 1

        for shape in example.input_shapes:
            # shape has to be at bottom of grid, cover full width and have gaps with depth of 1 at top
            is_bottom = shape.row + shape.height - 1 == max_row
            full_width = shape.width == cols
            # there should be some 0 cells at the top of the shape
            has_top_gap = any(
                example.input_grid[shape.row - 1, col] == 0
                for col in range(shape.col, shape.col + shape.width)
            )

            if is_bottom and full_width and has_top_gap:
                # collect gaps with their widths
                gaps = []
                col = shape.col
                while col < shape.col + shape.width:
                    if example.input_grid[shape.row, col] == 0:
                        gap_start = col
                        while (
                            col < shape.col + shape.width
                            and example.input_grid[shape.row, col] == 0
                        ):
                            col += 1
                        gap_end = col - 1
                        gaps.append((shape.row, gap_start, gap_end))
                    else:
                        col += 1
                # store gaps in example observations
                example.bottom_gaps = gaps

    for example in observations.example_observations:
        collect_bottom_gaps(example)

    # do same for test input
    collect_bottom_gaps(observations.test_observations)

    observations.bottom_gaps_everywhere = (
        all(
            ex.bottom_gaps and len(ex.bottom_gaps) > 0
            for ex in observations.example_observations
        )
        and observations.test_observations.bottom_gaps
        and len(observations.test_observations.bottom_gaps) > 0
    )


def check_single_enclosed_shape_in_enclosing_shape(observations: Observations) -> None:
    if not observations.shapes_collected:
        collect_shapes(observations)

    def test_example(example_observations: ExampleObservations) -> None:
        enclosing_shapes = example_observations.enclosing_shapes
        shapes = example_observations.input_diagonal_shapes or []
        non_enclosing_shape_cells = {cell for s in enclosing_shapes for cell in s.cells}
        non_enclosing_shapes = [
            s for s in shapes if not (s.cells & non_enclosing_shape_cells)
        ]

        for enclosing_shape in enclosing_shapes:
            # check if there is exactly one color inside the enclosing shape
            for shape in non_enclosing_shapes:
                # check if shape is inside one of the enclosing shape
                if (
                    shape.row > enclosing_shape.row
                    and shape.col > enclosing_shape.col
                    and shape.row + shape.height
                    < enclosing_shape.row + enclosing_shape.height
                    and shape.col + shape.width
                    < enclosing_shape.col + enclosing_shape.width
                ):
                    enclosing_shape.enclosed_shapes.append(shape)

            enclosed_colors = {s.color for s in enclosing_shape.enclosed_shapes}
            if len(enclosed_colors) == 1:
                example_observations.has_single_enclosed_shape = True
            else:
                example_observations.has_single_enclosed_shape = False

    for example in observations.example_observations:
        test_example(example)

    test_example(observations.test_observations)

    observations.has_single_enclosed_shape_everywhere = (
        all(ex.has_single_enclosed_shape for ex in observations.example_observations)
        and observations.test_observations.has_single_enclosed_shape
    )


def check_walls(observations: Observations) -> None:
    if not observations.shapes_collected:
        collect_shapes(observations)

    def is_wall(shape: Shape, grid: Grid) -> bool:
        length_correct = (
            shape.width == grid.shape[1] - 2 or shape.height == grid.shape[0] - 2
        )
        width_correct = shape.width == 1 or shape.height == 1
        position_correct = (
            shape.row == 0
            or shape.col == 0
            or shape.row == grid.shape[0] - 1
            or shape.col == grid.shape[1] - 1
        )
        return length_correct and width_correct and position_correct

    for example in observations.example_observations:
        wall_count = 0
        for shape in example.input_shapes:
            shape_is_wall = is_wall(shape, example.input_grid)
            if shape_is_wall:
                wall_count += 1
            shape.is_wall = shape_is_wall

        example.has_four_walls = wall_count == 4

    test = observations.test_observations
    wall_count = 0
    for shape in test.input_shapes:
        shape_is_wall = is_wall(shape, test.input_grid)
        if shape_is_wall:
            wall_count += 1
        shape.is_wall = shape_is_wall

    test.has_four_walls = wall_count == 4

    observations.has_four_walls_everywhere = (
        all(ex.has_four_walls for ex in observations.example_observations)
        and test.has_four_walls
    )


def check_four_aligned_shapes(observations: Observations) -> None:
    def _symmetric_colors(s):
        return (
            s[0].color == s[3].color
            and s[1].color == s[2].color
            and s[0].color != s[1].color
        )

    def _four_h_aligned(shapes):
        sorted_shapes = sorted(shapes, key=lambda x: x.col)
        centers = {x.row + x.height // 2 for x in sorted_shapes}

        return len(centers) == 1 and _symmetric_colors(sorted_shapes)

    def _four_v_aligned(shapes):
        sorted_shapes = sorted(shapes, key=lambda x: x.row)
        centers = {x.col + x.width // 2 for x in sorted_shapes}

        return len(centers) == 1 and _symmetric_colors(sorted_shapes)

    for example in observations.example_observations:
        inp = example.input_color_strict_diagonal_shapes or []
        out = example.output_color_strict_diagonal_shapes or []
        has_four = len(inp) == 4 and len(out) == 4

        horizontal_aligned = has_four and _four_h_aligned(inp) and _four_h_aligned(out)
        example.has_four_horizontally_aligned_shapes = horizontal_aligned

        vertical_aligned = has_four and _four_v_aligned(inp) and _four_v_aligned(out)
        example.has_four_vertically_aligned_shapes = vertical_aligned

    observations.has_four_horizontally_aligned_shapes_everywhere = all(
        ex.has_four_horizontally_aligned_shapes
        for ex in observations.example_observations
    )
    observations.has_four_vertically_aligned_shapes_everywhere = all(
        ex.has_four_vertically_aligned_shapes
        for ex in observations.example_observations
    )
    observations.has_four_aligned_shapes_everywhere = all(
        ex.has_four_horizontally_aligned_shapes or ex.has_four_vertically_aligned_shapes
        for ex in observations.example_observations
    )


def check_two_single_cells_on_rim(observations: Observations) -> None:
    if not observations.shapes_collected:
        collect_shapes(observations)

    def is_on_rim(shape: Shape, grid: Grid) -> bool:
        return (
            shape.row == 0
            or shape.col == 0
            or shape.row == grid.shape[0] - 1
            or shape.col == grid.shape[1] - 1
        )

    def exactly_two_same_color_rim_cells(example_obs: ExampleObservations) -> bool:
        grid = example_obs.input_grid
        from collections import defaultdict

        cells_by_color = defaultdict(int)
        for shape in example_obs.input_color_strict_shapes:
            if (
                shape.width == 1
                and shape.height == 1
                and is_on_rim(shape, grid)
                and shape.color != 0
            ):
                cells_by_color[shape.color] += 1
        return any(count == 2 for count in cells_by_color.values())

    for example in observations.example_observations:
        example.has_two_single_cells_on_rim = exactly_two_same_color_rim_cells(example)

    test = observations.test_observations
    test.has_two_single_cells_on_rim = exactly_two_same_color_rim_cells(test)

    observations.has_two_single_cells_on_rim_everywhere = (
        all(ex.has_two_single_cells_on_rim for ex in observations.example_observations)
        and test.has_two_single_cells_on_rim
    )


def check_consistent_output_grid_size(observations: Observations) -> None:
    result = observations.example_observations[0].output_grid.shape

    for example in observations.example_observations:
        example_output_size = example.output_grid.shape
        if example_output_size != result:
            result = None
            break

    observations.consistent_output_grid_size = result
