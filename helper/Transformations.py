from typing import Tuple

import numpy as np

# Transformations are pure Grid → Grid functions.
# Framings (parameterized decompositions using observation data) are expressed
# as closures in theory factories, not as standalone functions here.

Grid = np.ndarray


def make_hollow(grid: Grid) -> Grid:
    from helper.ObjectUtils import find_objects
    result = grid.copy()
    for obj in find_objects(grid.tolist()):
        min_r, min_c, max_r, max_c = obj.bounding_box
        for r, c in obj.cells:
            if min_r < r < max_r and min_c < c < max_c:
                result[r, c] = 0
    return result


def grow_cells(grid: Grid, scale: int, new_color: int) -> Grid:
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    half = scale // 2
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                result[max(0, r - half):min(rows, r + half + 1),
                       max(0, c - half):min(cols, c + half + 1)] = new_color
    return result


def boolean_combine(left: Grid, right: Grid, op: str, output_color: int) -> Grid:
    l_bool = left != 0
    r_bool = right != 0
    if op == "and":
        mask = l_bool & r_bool
    elif op == "or":
        mask = l_bool | r_bool
    elif op == "nor":
        mask = ~(l_bool | r_bool)
    else:
        raise ValueError(f"Unknown op: {op}")
    return np.where(mask, output_color, 0).astype(left.dtype)
