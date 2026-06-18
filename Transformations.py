from typing import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from Observations import Grid, Shape

Grid = np.ndarray


@dataclass
class ApplyState:
    grid: Grid
    source_grid: Grid
    source_shapes: tuple[Shape, ...]
    offset: tuple[int, int] = field(default=(0, 0))


Primitive = Callable[[ApplyState], ApplyState]
Program = list[Primitive]


def rotate_1(state: ApplyState) -> ApplyState:
    return replace(state, grid=np.rot90(state.grid, k=1))


def rotate_2(state: ApplyState) -> ApplyState:
    return replace(state, grid=np.rot90(state.grid, k=2))


def rotate_3(state: ApplyState) -> ApplyState:
    return replace(state, grid=np.rot90(state.grid, k=3))


def mirror(state: ApplyState) -> ApplyState:
    return replace(state, grid=np.maximum(state.grid, np.flipud(state.grid)))


def swap_two_colors(state: ApplyState) -> ApplyState:
    grid = state.grid
    colors = sorted(set(int(c) for c in np.unique(grid)) - {0})
    if len(colors) != 2:
        return state
    c1, c2 = colors
    return replace(state, grid=np.select([grid == c1, grid == c2], [c2, c1], grid))


def erode(state: ApplyState) -> ApplyState:
    all_cells = {cell for s in state.source_shapes for cell in s.cells}
    result = state.grid.copy()
    for row, col in all_cells:
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if not all(n in all_cells for n in neighbors):
            result[row, col] = 0
    return replace(state, grid=result)


def subtract_from_source(state: ApplyState) -> ApplyState:
    mask = (state.source_grid != 0) & (state.grid == 0)
    return replace(state, grid=np.where(mask, state.source_grid, 0))


def set_offset_to_source_origin(state: ApplyState) -> ApplyState:
    if not state.source_shapes:
        return state
    min_row = min(s.row for s in state.source_shapes)
    min_col = min(s.col for s in state.source_shapes)
    return replace(state, offset=(min_row, min_col))


def new_grid_from_source_bounds(state: ApplyState) -> ApplyState:
    if not state.source_shapes:
        return state
    min_row = min(s.row for s in state.source_shapes)
    max_row = max(s.row + s.height - 1 for s in state.source_shapes)
    min_col = min(s.col for s in state.source_shapes)
    max_col = max(s.col + s.width - 1 for s in state.source_shapes)
    h = max_row - min_row + 1
    w = max_col - min_col + 1
    return replace(state, grid=np.zeros((h, w), dtype=state.source_grid.dtype))


def paint_source_at_offset(state: ApplyState) -> ApplyState:
    dr, dc = state.offset
    result = state.grid.copy()
    rows, cols = np.where(state.source_grid != 0)
    for r, c in zip(rows, cols):
        nr, nc = r - dr, c - dc
        if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
            result[nr, nc] = state.source_grid[r, c]
    return replace(state, grid=result)


def make_recolor(fc: int, tc: int) -> Primitive:
    def fn(state: ApplyState) -> ApplyState:
        return replace(state, grid=np.where(state.grid == fc, tc, state.grid))
    return fn


def make_apply_color_map(mapping: dict) -> Primitive:
    def fn(state: ApplyState) -> ApplyState:
        result = np.zeros_like(state.grid)
        for k, v in mapping.items():
            result[state.grid == k] = v
        return replace(state, grid=result)
    return fn


BASE_PRIMITIVES: list[Primitive] = [
    rotate_1, rotate_2, rotate_3,
    mirror, swap_two_colors,
    erode, subtract_from_source,
    set_offset_to_source_origin, new_grid_from_source_bounds, paint_source_at_offset,
]
