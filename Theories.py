from Transformations import (
    Theory,
    cast_uni_ray_from_two_by_twos,
    crop_to_square_abstraction,
    make_spiral_transformation,
    make_recolor_transformation,
    make_recolor_by_enclosure_transformation,
    recolor_to_square_abstraction,
    rotate_90,
    rotate_180,
    rotate_270,
    mirror_horizontally,
    swap_colors,
    make_hollow,
    crop_to_content,
)
from Observations import Observations

ARC_COLORS = range(10)

SAME_SIZE_THEORIES: list[Theory] = [
    [rotate_90],
    [rotate_180],
    [rotate_270],
    [mirror_horizontally],
    [swap_colors],
    [make_hollow],
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
    if observations.two_new_output_colors_everywhere and observations.enclosed_zero_shapes_everywhere and observations.non_enclosed_zero_shapes_everywhere:
        theories.append([make_recolor_by_enclosure_transformation(flip_colors=False)])
        theories.append([make_recolor_by_enclosure_transformation(flip_colors=True)])
    # if observations.grid_size_stays_identical:
    #     theories.extend(SAME_SIZE_THEORIES)
    # if observations.grid_size_decreases:
    #     theories.extend(SIZE_REDUCING_THEORIES)

    theories.extend(RECOLOR_THEORIES)
    return theories
