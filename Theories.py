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
    [crop_to_content],
    [crop_to_content, swap_colors],
]

RECOLOR_THEORIES: list[Theory] = [
    [recolor(fc, tc)]
    for fc in ARC_COLORS
    for tc in ARC_COLORS
    if fc != tc
]


def get_theories(obs: Observations) -> list[Theory]:
    theories: list[Theory] = []

    if obs.same_size:
        theories.extend(SAME_SIZE_THEORIES)

    if obs.size_decreases:
        theories.extend(SIZE_REDUCING_THEORIES)

    theories.extend(RECOLOR_THEORIES)

    return theories
