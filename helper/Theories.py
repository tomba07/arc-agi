from typing import Callable, List

import numpy as np

Grid = np.ndarray


def _make_hollow(grid: Grid) -> Grid:
    interior = np.zeros(grid.shape, dtype=bool)
    interior[1:-1, 1:-1] = (
        (grid[1:-1, 1:-1] != 0)
        & (grid[:-2,  1:-1] != 0)
        & (grid[2:,   1:-1] != 0)
        & (grid[1:-1, :-2]  != 0)
        & (grid[1:-1, 2:]   != 0)
    )
    return np.where(interior, 0, grid)


def _swap_colors(grid: Grid) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    a, b = colors
    return np.select([grid == a, grid == b], [b, a], grid).astype(grid.dtype)


def _color_replace_and_erase(grid: Grid, replace_color: int) -> Grid:
    others = sorted(set(np.unique(grid)) - {0, replace_color})
    if len(others) != 1:
        return grid
    other = others[0]
    return np.select([grid == replace_color, grid == other], [other, 0], grid).astype(grid.dtype)


def _crop_to_content(grid: Grid) -> Grid:
    rows, cols = np.where(grid != 0)
    if len(rows) == 0:
        return grid
    return grid[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]


def _rotate_90(grid: Grid) -> Grid:  return np.rot90(grid, k=1)
def _rotate_180(grid: Grid) -> Grid: return np.rot90(grid, k=2)
def _rotate_270(grid: Grid) -> Grid: return np.rot90(grid, k=3)
def _flip_lr(grid: Grid) -> Grid:    return np.fliplr(grid)
def _flip_ud(grid: Grid) -> Grid:    return np.flipud(grid)
def _transpose(grid: Grid) -> Grid:      return grid.T
def _anti_transpose(grid: Grid) -> Grid: return np.rot90(grid.T)
def _overlay_flip_ud(grid: Grid) -> Grid: return np.maximum(grid, np.flipud(grid))


def _recolor(src: int, dst: int) -> Callable[[Grid], Grid]:
    def fn(g: Grid) -> Grid: return np.where(g == src, dst, g).astype(g.dtype)
    return fn


def _erase_replacing(replace: int) -> Callable[[Grid], Grid]:
    def fn(g: Grid) -> Grid: return _color_replace_and_erase(g, replace)
    return fn


def generate_theories() -> List[Callable[[Grid], Grid]]:
    transforms: List[Callable[[Grid], Grid]] = [
        _rotate_90,
        _rotate_180,
        _rotate_270,
        _flip_lr,
        _flip_ud,
        _transpose,
        _anti_transpose,
        _crop_to_content,
        _make_hollow,
        _overlay_flip_ud,
        _swap_colors,
    ]
    for a in range(1, 10):
        for b in range(0, 10):
            if a != b:
                transforms.append(_recolor(a, b))
        transforms.append(_erase_replacing(a))
    return transforms
