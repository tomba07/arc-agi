from Transformations import (
    Theory,
    cast_uni_ray_from_two_by_twos,
    connect_same_color_opposing_cells,
    create_beam_from_spaceship_tip,
    crop_to_square_abstraction,
    make_logical_operation_on_divided_input_transformations,
    make_spiral_transformation,
    make_recolor_transformation,
    make_recolor_by_enclosure_transformation,
    mirror_horizontally_and_vertically,
    recolor_to_square_abstraction,
    rotate_90,
    rotate_180,
    rotate_270,
    mirror_across_horizontal_axis,
    swap_colors,
    make_hollow,
    crop_to_content,
    transpose,
    make_arrange_colored_cells_transformations,
)
from Observations import Observations

ARC_COLORS = range(10)

LOGICAL_OPERATIONS = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]

SAME_SIZE_THEORIES: list[Theory] = [
    [rotate_90],
    [rotate_180],
    [rotate_270],
    [mirror_across_horizontal_axis],
    [swap_colors],
    [make_hollow],
    [rotate_90, mirror_across_horizontal_axis],
    [transpose],
]

SIZE_REDUCING_THEORIES: list[Theory] = [
    [crop_to_content, swap_colors],
]

RECOLOR_THEORIES: list[Theory] = [
    [make_recolor_transformation(from_color, to_color)]
    for from_color in ARC_COLORS
    for to_color in ARC_COLORS
    if from_color != to_color
]

DIVIDER_THEORIES: list[Theory] = [
    [make_logical_operation_on_divided_input_transformations(op)]
    for op in LOGICAL_OPERATIONS
]


def get_theories(observations: Observations) -> list[Theory]:
    theories: list[Theory] = []

    if observations.single_shape_everywhere:
        theories.append([crop_to_content])
    if observations.all_inputs_empty and observations.single_output_color is not None:
        theories.append([make_spiral_transformation(observations.single_output_color)])
    if observations.input_square_abstraction_everywhere:
        theories.append([crop_to_square_abstraction, recolor_to_square_abstraction])
    if observations.consistent_two_by_two_uni_ray_direction_by_color:
        theories.append([cast_uni_ray_from_two_by_twos])
    if (
        observations.two_new_output_colors_everywhere
        and observations.enclosed_zero_shapes_everywhere
        and observations.non_enclosed_zero_shapes_everywhere
    ):
        theories.append([make_recolor_by_enclosure_transformation(flip_colors=False)])
        theories.append([make_recolor_by_enclosure_transformation(flip_colors=True)])
    if (
        observations.cell_count_by_color_identical_everywhere
        and not observations.grid_size_stays_identical
    ):
        directions = ["horizontal", "vertical"]
        for direction in directions:
            theories.append(
                [make_arrange_colored_cells_transformations(direction, True)]
            )
            theories.append(
                [make_arrange_colored_cells_transformations(direction, False)]
            )
    if observations.has_opposing_same_color_single_cells_everywhere:
        theories.append([connect_same_color_opposing_cells])
    if observations.has_spaceship_shape_everywhere:
        theories.append([create_beam_from_spaceship_tip])
    if observations.all_outputs_twice_as_large_as_inputs:
        theories.append([mirror_horizontally_and_vertically])
    if observations.grid_size_stays_identical:
        theories.extend(SAME_SIZE_THEORIES)
    if (observations.has_single_horizontal_divider_everywhere or observations.has_single_vertical_divider_everywhere) and observations.single_output_color is not None:
        theories.extend(DIVIDER_THEORIES)
    # if observations.grid_size_decreases:
    #     theories.extend(SIZE_REDUCING_THEORIES)

    theories.extend(RECOLOR_THEORIES)
    return theories
