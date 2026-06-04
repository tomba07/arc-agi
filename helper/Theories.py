from typing import Callable, Generator, List, Tuple

import numpy as np

from helper.Transformations import (
    DIRECTIONS_ALL_8,
    DIRECTIONS_DIAGONAL,
    DIRECTIONS_ORTHOGONAL,
    boolean_combine,
    dilate_square,
    draw_clockwise_spiral,
    erode,
    find_divider,
    make_hollow,
    mask_subtract,
    ray_fill,
    recolor_nonzero,
    remap_replace_keep,
    snap_to_border,
    swap_two_nonzero,
)

Grid = np.ndarray
Transform = Tuple[str, Callable[[Grid], Grid]]


def _crop_to_content(grid: Grid) -> Grid:
    rows, cols = np.where(grid != 0)
    if len(rows) == 0:
        return grid
    return grid[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]


def all_transforms(examples: List[Tuple[Grid, Grid]]) -> List[Transform]:
    fns: List[Transform] = [
        # Symmetry
        ("rotate_90",           lambda g: np.rot90(g, k=1)),
        ("rotate_180",          lambda g: np.rot90(g, k=2)),
        ("rotate_270",          lambda g: np.rot90(g, k=3)),
        ("flip_lr",             np.fliplr),
        ("flip_ud",             np.flipud),
        ("transpose",           lambda g: g.T),
        ("anti_transpose",      lambda g: np.rot90(g.T)),
        # Crop
        ("crop_to_content",     _crop_to_content),
        # Morphological
        ("erode",               erode),
        ("make_hollow",         make_hollow),
        ("ray_fill_diagonal",   lambda g: ray_fill(g, DIRECTIONS_DIAGONAL)),
        ("ray_fill_orthogonal", lambda g: ray_fill(g, DIRECTIONS_ORTHOGONAL)),
        ("ray_fill_all8",       lambda g: ray_fill(g, DIRECTIONS_ALL_8)),
        # Overlay / combination
        ("overlay_flip_ud",     lambda g: np.maximum(g, np.flipud(g))),
        ("overlay_flip_lr",     lambda g: np.maximum(g, np.fliplr(g))),
        ("swap_two_nonzero",    swap_two_nonzero),
        # Spatial
        ("snap_to_border",      snap_to_border),
        ("draw_clockwise_spiral", draw_clockwise_spiral),
    ]

    # Color substitutions: enumerate all color pairs
    for a in range(1, 10):
        for b in range(0, 10):
            if a != b:
                fns.append((
                    f"replace_{a}_with_{b}",
                    lambda g, a=a, b=b: np.where(g == a, b, g).astype(g.dtype),
                ))
        fns.append((f"remap_replace_{a}", lambda g, a=a: remap_replace_keep(g, a)))

    # Scale: enumerate plausible scales and output colors
    for scale in [2, 3, 4, 5]:
        for color in range(1, 10):
            fns.append((
                f"scale_{scale}x_color_{color}",
                lambda g, s=scale, c=color: recolor_nonzero(dilate_square(g, s // 2), c),
            ))

    # Boolean combine: derive split point from first training example
    try:
        axis, idx = find_divider(examples[0][0])
        inp_colors = set(np.unique(examples[0][0])) - {0}
        out_colors = set(np.unique(examples[0][1])) - {0}
        output_color = next(iter(out_colors - inp_colors), 1)

        def _split(g, a=axis, i=idx):
            return (g[:, :i], g[:, i + 1:]) if a == "col" else (g[:i, :], g[i + 1:, :])

        for op in ["and", "or", "nor"]:
            fns.append((
                f"divider_{op}",
                lambda g, op=op, oc=output_color: boolean_combine(*_split(g), op, oc),
            ))
    except (ValueError, IndexError):
        pass

    return fns


def search(
    transforms: List[Transform],
    examples: List[Tuple[Grid, Grid]],
    max_depth: int = 2,
) -> Generator[Tuple[str, Callable[[Grid], Grid]], None, None]:
    def validates(fn: Callable[[Grid], Grid]) -> bool:
        try:
            return all(np.array_equal(fn(inp), out) for inp, out in examples)
        except Exception:
            return False

    for name, fn in transforms:
        if validates(fn):
            yield name, fn

    if max_depth < 2:
        return

    for n1, f1 in transforms:
        for n2, f2 in transforms:
            composed = lambda g, a=f1, b=f2: b(a(g))
            if validates(composed):
                yield f"{n1}+{n2}", composed
