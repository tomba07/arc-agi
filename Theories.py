from Transformations import (
    Theory,
    rotate_90, rotate_180, rotate_270,
    mirror_horizontally, swap_colors, make_hollow,
    crop_to_content, recolor, make_color_map_fn,
)
from Observations import Observations


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


def get_theories(obs: Observations) -> list[Theory]:
    theories: list[Theory] = []

    if obs.same_size:
        theories.extend(SAME_SIZE_THEORIES)

    if obs.size_decreases:
        theories.extend(SIZE_REDUCING_THEORIES)

    if obs.color_map:
        theories.append([make_color_map_fn(obs.color_map)])

    for fc, tc in obs.recolor_pairs:
        theories.append([recolor(fc, tc)])
        if obs.same_size:
            theories.append([swap_colors, recolor(fc, tc)])

    return theories
