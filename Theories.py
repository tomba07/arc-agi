from Transformations import (
    Theory,
    rotate_90, rotate_180, rotate_270,
    mirror_horizontally, swap_colors, make_hollow,
    crop_to_content, recolor,
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
    [recolor(from_color, to_color)]
    for from_color in ARC_COLORS
    for to_color in ARC_COLORS
    if from_color != to_color
]


def get_theories(obs: Observations) -> list[Theory]:
    theories: list[Theory] = []
    
    if obs.single_shape_everywhere:
        theories.append([crop_to_content])

    if obs.grid_size_stays_identical:
        theories.extend(SAME_SIZE_THEORIES)

    if obs.grid_size_decreases:
        theories.extend(SIZE_REDUCING_THEORIES)

    theories.extend(RECOLOR_THEORIES)

    return theories
