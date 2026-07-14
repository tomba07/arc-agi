from dataclasses import dataclass, field
from typing import Callable

from Transformations import (
    Theory,
    cast_rays_from_single_cells,
    connect_similar_shapes,
    connect_two_single_cells_on_rim,
    count_enclosed_cells,
    crop_and_color_change,
    crop_and_color_change_reversed,
    expand_enclosing_shapes,
    fill_enclosing_shapes_with_dominant_color,
    grow_and_connect_single_cells,
    grow_one_by_ones,
    make_implicit_divider_operation,
    make_single_divider_overlay_operation,
    make_spiral_transformation_reversed,
    make_two_cell_line_connection,
    make_two_divider_overlay_operation,
    mirror_horizontally_vertically_and_diagonally,
    mirror_single_enclosed_shape,
    move_inner_shapes_outward_horizontal,
    move_inner_shapes_outward_vertical,
    move_inner_shapes_outward,
    move_one_by_ones_to_same_colored_wall,
    overlay_if_no_overlap,
    print_two_by_two_color_count,
    put_shapes_into_bottom_gaps,
    remove_non_enclosed_single_cells,
    rotate_90,
    rotate_180,
    rotate_270,
    single_cell_attraction,
    transpose,
    mirror_across_horizontal_axis,
    mirror_horizontally_and_vertically,
    crop_to_content,
    crop_to_square_abstraction,
    recolor_to_square_abstraction,
    make_hollow,
    change_enclosing_shapes_color,
    fill_with_increasing_rows,
    swap_colors,
    make_recolor_transformation,
    make_recolor_by_enclosure_transformation,
    make_arrange_colored_cells_transformations,
    make_spiral_transformation,
    cast_uni_ray_from_two_by_twos,
    connect_same_color_opposing_cells,
    create_beam_from_spaceship_tip,
    make_divider_operation,
)
from Observations import (
    Observations,
    ObservationCheck,
    check_bottom_gaps,
    check_color_change_indicators,
    check_four_aligned_shapes,
    check_grid_sizes,
    check_implicit_color_dividers,
    check_only_similar_input_shapes,
    check_two_dividers,
    check_walls,
    collect_shapes,
    check_output_size_ratio,
    check_output_height_half_of_width,
    check_dividers,
    check_color_sets,
    check_removed_color,
    check_zero_shapes,
    check_cell_counts,
    check_square_abstraction,
    check_opposing_cells,
    check_spaceship,
    check_two_by_two_rays,
    check_enclosing_shapes,
    check_recolor_context,
    check_single_enclosed_shape_in_enclosing_shape,
    check_two_single_cells_on_rim,
    check_consistent_output_grid_size,
)
from Enums import AxisDirection, Direction, LogicalOperation

ARC_COLORS = range(10)


@dataclass
class TheoryDef:
    name: str
    condition: Callable[[Observations], bool]
    transforms: Theory
    required_checks: list[ObservationCheck] = field(default_factory=list)


ALL_THEORIES: list[TheoryDef] = [
    # single-observation, single-transform theories
    TheoryDef(
        "mirror_horizontally_and_vertically",
        lambda observations: bool(observations.all_outputs_twice_as_large_as_inputs),
        [mirror_horizontally_and_vertically],
        [check_output_size_ratio],
    ),
    TheoryDef(
        "tile_with_mirrors",
        lambda observations: bool(observations.all_outputs_thrice_as_large_as_inputs),
        [mirror_horizontally_vertically_and_diagonally],
        [check_output_size_ratio],
    ),
    TheoryDef(
        "fill_with_increasing_rows",
        lambda observations: bool(observations.output_height_half_of_width_everywhere),
        [fill_with_increasing_rows],
        [check_output_height_half_of_width],
    ),
    TheoryDef(
        "connect_opposing_cells",
        lambda observations: bool(
            observations.has_opposing_same_color_single_cells_everywhere
        ),
        [connect_same_color_opposing_cells],
        [collect_shapes, check_opposing_cells],
    ),
    TheoryDef(
        "cast_uni_ray",
        lambda observations: bool(
            observations.consistent_two_by_two_uni_ray_direction_by_color
        ),
        [cast_uni_ray_from_two_by_twos],
        [collect_shapes, check_two_by_two_rays],
    ),
    TheoryDef(
        "beam_from_spaceship",
        lambda observations: bool(observations.has_spaceship_shape_everywhere),
        [create_beam_from_spaceship_tip],
        [collect_shapes, check_spaceship],
    ),
    # same-size geometric transforms
    *[
        TheoryDef(
            name,
            lambda observations: bool(
                observations.grid_size_stays_identical and observations.shapes_collected
            ),
            transforms,
            [check_grid_sizes, collect_shapes],
        )
        for name, transforms in [
            ("rotate_90", [rotate_90]),
            ("rotate_180", [rotate_180]),
            ("rotate_270", [rotate_270]),
            ("mirror_across_horizontal_axis", [mirror_across_horizontal_axis]),
            ("swap_colors", [swap_colors]),
            ("make_hollow", [make_hollow]),
            ("rotate_90_mirror", [rotate_90, mirror_across_horizontal_axis]),
            ("transpose", [transpose]),
        ]
    ],
    # size-changing theories
    TheoryDef(
        "crop_to_content",
        lambda observations: bool(observations.grid_size_decreases),
        [crop_to_content],
        [collect_shapes],
    ),
    TheoryDef(
        "crop_to_content_and_swap",
        lambda observations: bool(
            observations.grid_size_decreases and observations.single_shape_everywhere
        ),
        [crop_to_content, swap_colors],
        [check_grid_sizes, collect_shapes],
    ),
    TheoryDef(
        "crop_to_square_abstraction",
        lambda observations: bool(observations.input_square_abstraction_everywhere),
        [crop_to_square_abstraction, recolor_to_square_abstraction],
        [collect_shapes, check_square_abstraction],
    ),
    TheoryDef(
        "cast_rays_from_single_cells",
        lambda observations: bool(
            observations.input_has_single_one_by_one_shape_everywhere
            and not observations.output_has_single_one_by_one_shape_everywhere
        ),
        [cast_rays_from_single_cells],
        [collect_shapes],
    ),
    TheoryDef(
        "grow_one_by_ones",
        lambda observations: bool(
            observations.all_inputs_only_one_by_ones
            and not observations.output_has_single_one_by_one_shape_everywhere
            and observations.consistent_new_output_colors is not None
            and len(observations.consistent_new_output_colors) == 1
        ),
        [grow_one_by_ones],
        [collect_shapes, check_color_sets],
    ),
    TheoryDef(
        "move_one_by_ones_to_same_colored_wall",
        lambda observations: bool(observations.has_four_walls_everywhere),
        [move_one_by_ones_to_same_colored_wall],
        [collect_shapes, check_walls],
    ),
    *[
        TheoryDef(
            f"arrange_colored_cells_{direction.value}_{increasing}",
            lambda observations: (
                bool(observations.cell_count_by_color_identical_everywhere)
                and observations.grid_size_stays_identical is False
            ),
            [make_arrange_colored_cells_transformations(direction, increasing)],
            [check_grid_sizes, check_cell_counts],
        )
        for direction in AxisDirection
        for increasing in (True, False)
    ],
    # structural / divider theories
    *[
        TheoryDef(
            f"divider_{op.value}",
            lambda observations: (
                (
                    bool(observations.has_single_horizontal_divider_everywhere)
                    or bool(observations.has_single_vertical_divider_everywhere)
                )
                and observations.single_output_color is not None
            ),
            [make_divider_operation(op)],
            [check_dividers, check_color_sets],
        )
        for op in LogicalOperation
    ],
    *[
        TheoryDef(
            f"implicit_divider_{op.value}",
            lambda observations: (
                (
                    bool(observations.has_single_implicit_horizontal_divider_everywhere)
                    or bool(
                        observations.has_single_implicit_vertical_divider_everywhere
                    )
                )
                and observations.single_output_color is not None
            ),
            [make_implicit_divider_operation(op)],
            [check_implicit_color_dividers, check_color_sets],
        )
        for op in LogicalOperation
    ],
    # color theories
    *[
        TheoryDef(
            f"spiral_color_{color}_rot{rotation}",
            lambda observations, c=color: bool(
                observations.all_inputs_empty and observations.single_output_color == c
            ),
            [make_spiral_transformation(color, rotation)],
            [collect_shapes, check_color_sets],
        )
        for color in ARC_COLORS
        for rotation in range(4)
    ],
    *[
        TheoryDef(
            f"spiral_color_reversed_{color}_rot{rotation}",
            lambda observations, c=color: bool(
                observations.all_inputs_empty and observations.single_output_color == c
            ),
            [make_spiral_transformation_reversed(color, rotation)],
            [collect_shapes, check_color_sets],
        )
        for color in ARC_COLORS
        for rotation in range(4)
    ],
    *[
        TheoryDef(
            f"removed_recolor_{color}",
            lambda observations, color=color: observations.removed_input_color == color,
            [make_recolor_transformation(color, 0)],
            [check_removed_color],
        )
        for color in ARC_COLORS
    ],
    *[
        TheoryDef(
            f"removed_swap_recolor_{color}",
            lambda observations, color=color: observations.removed_input_color == color,
            [swap_colors, make_recolor_transformation(color, 0)],
            [check_removed_color],
        )
        for color in ARC_COLORS
    ],
    TheoryDef(
        "recolor_by_enclosure",
        lambda observations: bool(
            observations.two_new_output_colors_everywhere
            and observations.enclosed_zero_shapes_everywhere
            and observations.non_enclosed_zero_shapes_everywhere
        ),
        [make_recolor_by_enclosure_transformation(flip_colors=False)],
        [check_color_sets, check_zero_shapes],
    ),
    TheoryDef(
        "recolor_by_enclosure_flipped",
        lambda observations: bool(
            observations.two_new_output_colors_everywhere
            and observations.enclosed_zero_shapes_everywhere
            and observations.non_enclosed_zero_shapes_everywhere
        ),
        [make_recolor_by_enclosure_transformation(flip_colors=True)],
        [check_color_sets, check_zero_shapes],
    ),
    TheoryDef(
        "change_enclosing_shapes_color",
        lambda observations: (
            bool(observations.has_enclosing_shapes_everywhere)
            and observations.consistent_new_output_colors is not None
            and len(observations.consistent_new_output_colors) == 1
        ),
        [change_enclosing_shapes_color],
        [check_color_sets, collect_shapes, check_enclosing_shapes],
    ),
    # recolor pair theories — broadest fallback, checked last
    *[
        TheoryDef(
            f"recolor_{from_color}_to_{to_color}",
            lambda observations, from_color=from_color, to_color=to_color: (
                bool(observations.is_recolor_context)
                and (
                    observations.consistent_removed_colors is None
                    or observations.consistent_new_output_colors is None
                    or (
                        from_color in observations.consistent_removed_colors
                        and to_color in observations.consistent_new_output_colors
                    )
                )
            ),
            [make_recolor_transformation(from_color, to_color)],
            [check_recolor_context, check_color_sets],
        )
        for from_color in ARC_COLORS
        for to_color in ARC_COLORS
        if from_color != to_color
    ],
    TheoryDef(
        "fill_enclosing_shapes_with_dominant_color",
        lambda observations: bool(observations.has_enclosing_shapes_everywhere),
        [fill_enclosing_shapes_with_dominant_color, remove_non_enclosed_single_cells],
        [collect_shapes, check_enclosing_shapes],
    ),
    TheoryDef(
        "connect_similar_shapes",
        lambda observations: bool(observations.only_similar_input_shapes),
        [connect_similar_shapes],
        [collect_shapes, check_only_similar_input_shapes, check_color_sets],
    ),
    TheoryDef(
        "put_shapes_in_bottom_gaps",
        lambda observations: bool(observations.bottom_gaps_everywhere),
        [put_shapes_into_bottom_gaps],
        [collect_shapes, check_bottom_gaps],
    ),
    TheoryDef(
        "print_two_by_two_color_count",
        lambda observations: bool(
            observations.single_non_by_two_shape_everywhere
            and observations.two_by_twos_everywhere
        ),
        [print_two_by_two_color_count],
        [collect_shapes],
    ),
    TheoryDef(
        "mirror_single_enclosed_shape",
        lambda observations: bool(observations.has_single_enclosed_shape_everywhere),
        [mirror_single_enclosed_shape],
        [
            collect_shapes,
            check_enclosing_shapes,
            check_single_enclosed_shape_in_enclosing_shape,
        ],
    ),
    *[
        TheoryDef(
            f"single_horizontal_divider_overlay_{direction.value}",
            lambda observations: bool(
                observations.has_single_horizontal_divider_everywhere
            ),
            [make_single_divider_overlay_operation(direction)],
            [check_dividers],
        )
        for direction in [Direction.UP, Direction.DOWN]
    ],
    *[
        TheoryDef(
            f"single_vertical_divider_overlay_{direction.value}",
            lambda observations: bool(
                observations.has_single_vertical_divider_everywhere
            ),
            [make_single_divider_overlay_operation(direction)],
            [check_dividers],
        )
        for direction in [Direction.LEFT, Direction.RIGHT]
    ],
    *[
        TheoryDef(
            f"two_horizontal_dividers_overlay_{direction.value}",
            lambda observations: bool(
                observations.has_two_horizontal_dividers_everywhere
            ),
            [make_two_divider_overlay_operation(direction)],
            [collect_shapes, check_two_dividers],
        )
        for direction in [Direction.UP, Direction.DOWN]
    ],
    *[
        TheoryDef(
            f"two_vertical_dividers_overlay_{direction.value}",
            lambda observations: bool(
                observations.has_two_vertical_dividers_everywhere
            ),
            [make_two_divider_overlay_operation(direction)],
            [collect_shapes, check_two_dividers],
        )
        for direction in [Direction.LEFT, Direction.RIGHT]
    ],
    *[
        TheoryDef(
            "cell_attraction",
            lambda observations: bool(
                all(
                    len(ex.input_shapes) == 2
                    for ex in observations.example_observations
                )
                and observations.test_observations.input_shapes is not None
                and len(observations.test_observations.input_shapes) == 2
            ),
            [single_cell_attraction],
            [collect_shapes],
        )
    ],
    *[
        TheoryDef(
            "move_inner_shapes_outward_horizontal",
            lambda observations: bool(
                observations.has_four_horizontally_aligned_shapes_everywhere
            ),
            [move_inner_shapes_outward_horizontal],
            [collect_shapes, check_four_aligned_shapes],
        )
    ],
    *[
        TheoryDef(
            "move_inner_shapes_outward_vertical",
            lambda observations: bool(
                observations.has_four_vertically_aligned_shapes_everywhere
            ),
            [move_inner_shapes_outward_vertical],
            [collect_shapes, check_four_aligned_shapes],
        )
    ],
    *[
        TheoryDef(
            "move_inner_shapes_outward",
            lambda observations: bool(observations.has_four_aligned_shapes_everywhere),
            [move_inner_shapes_outward],
            [collect_shapes, check_four_aligned_shapes],
        )
    ],
    *[
        TheoryDef(
            "grow_and_connect",
            lambda observations: bool(
                observations.consistent_new_output_colors is not None
                and len(observations.consistent_new_output_colors) == 1
                and observations.all_inputs_only_one_by_ones,
            ),
            [grow_and_connect_single_cells],
            [collect_shapes, check_color_sets],
        )
    ],
    *[
        TheoryDef(
            f"make_two_cell_line_connection_starting_{color}",
            lambda observations: bool(
                # two
                observations.all_inputs_only_one_by_ones
                and len(observations.consistent_new_output_colors) == 1
            ),
            [make_two_cell_line_connection(color)],
            [collect_shapes, check_color_sets],
        )
        for color in ARC_COLORS
    ],
    *[
        TheoryDef(
            "overlay_if_no_overlap",
            lambda observations: bool(
                observations.has_single_horizontal_divider_everywhere
                or observations.has_single_vertical_divider_everywhere
            ),
            [overlay_if_no_overlap],
            [check_dividers],
        )
    ],
    *[
        TheoryDef(
            "connect_two_single_cells_on_rim",
            lambda observations: bool(
                observations.has_two_single_cells_on_rim_everywhere
            ),
            [connect_two_single_cells_on_rim],
            [collect_shapes, check_color_sets, check_two_single_cells_on_rim],
        )
    ],
    *[
        TheoryDef(
            "count_enclosed_cells",
            lambda observations: bool(
                observations.has_enclosing_shapes_everywhere
                and observations.consistent_output_grid_size
            ),
            [count_enclosed_cells],
            [collect_shapes, check_enclosing_shapes, check_consistent_output_grid_size],
        )
    ],
    *[
        TheoryDef(
            "expand_enclosing_shapes",
            lambda observations: bool(
                observations.has_enclosing_shapes_somewhere
                and observations.grid_size_stays_identical
            ),
            [expand_enclosing_shapes],
            [
                collect_shapes,
                check_enclosing_shapes,
                check_grid_sizes,
                check_color_sets,
            ],
        )
    ],
    *[
        TheoryDef(
            "crop_and_color_change_indicators",
            lambda observations: bool(
                observations.color_change_indicator_shapes_everywhere
            ),
            [crop_and_color_change],
            [collect_shapes, check_color_change_indicators],
        )
    ],
    *[
        TheoryDef(
            "crop_and_color_change_indicators_reversed",
            lambda observations: bool(
                observations.color_change_indicator_shapes_everywhere
            ),
            [crop_and_color_change_reversed],
            [collect_shapes, check_color_change_indicators],
        )
    ],
]


def get_theories(observations: Observations) -> list[TheoryDef]:
    return [t for t in ALL_THEORIES if t.condition(observations)]
