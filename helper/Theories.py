from typing import Callable, Generator, List, Tuple

import numpy as np

Grid = np.ndarray
Transform = Tuple[str, Callable[[Grid], Grid]]


def _erode(grid: Grid) -> Grid:
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (grid[r, c] != 0
                    and grid[r - 1, c] != 0 and grid[r + 1, c] != 0
                    and grid[r, c - 1] != 0 and grid[r, c + 1] != 0):
                result[r, c] = grid[r, c]
    return result


def _make_hollow(grid: Grid) -> Grid:
    eroded = _erode(grid)
    return np.where(eroded != 0, 0, grid)


def _swap_two_nonzero(grid: Grid) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    a, b = colors
    return np.select([grid == a, grid == b], [b, a], grid).astype(grid.dtype)


def _remap_replace(grid: Grid, replace_color: int) -> Grid:
    others = sorted(set(np.unique(grid)) - {0, replace_color})
    if len(others) != 1:
        return grid
    other = others[0]
    return np.select(
        [grid == replace_color, grid == other], [other, 0], grid
    ).astype(grid.dtype)


def _crop_to_content(grid: Grid) -> Grid:
    rows, cols = np.where(grid != 0)
    if len(rows) == 0:
        return grid
    return grid[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]


def search(
    examples: List[Tuple[Grid, Grid]],
    max_depth: int = 2,
) -> Generator[Tuple[str, Callable[[Grid], Grid]], None, None]:
    def validates(fn: Callable[[Grid], Grid]) -> bool:
        try:
            return all(np.array_equal(fn(inp), out) for inp, out in examples)
        except Exception:
            return False

    transforms: List[Transform] = [
        ("rotate_90",        lambda g: np.rot90(g, k=1)),
        ("rotate_180",       lambda g: np.rot90(g, k=2)),
        ("rotate_270",       lambda g: np.rot90(g, k=3)),
        ("flip_lr",          np.fliplr),
        ("flip_ud",          np.flipud),
        ("transpose",        lambda g: g.T),
        ("anti_transpose",   lambda g: np.rot90(g.T)),
        ("crop_to_content",  _crop_to_content),
        ("make_hollow",      _make_hollow),
        ("overlay_flip_ud",  lambda g: np.maximum(g, np.flipud(g))),
        ("swap_two_nonzero", _swap_two_nonzero),
    ]
    for a in range(1, 10):
        for b in range(0, 10):
            if a != b:
                transforms.append((
                    f"replace_{a}_with_{b}",
                    lambda g, a=a, b=b: np.where(g == a, b, g).astype(g.dtype),
                ))
        transforms.append((f"remap_replace_{a}", lambda g, a=a: _remap_replace(g, a)))

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
