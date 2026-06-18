from typing import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

Grid = np.ndarray
Theory = list[Callable[[Grid], Grid]]


def apply_theory(theory: Theory, grid: Grid) -> Grid:
    for fn in theory:
        grid = fn(grid)
    return grid


@dataclass
class ArcObject:
    row: int
    col: int
    width: int
    height: int
    cells: frozenset[tuple[int, int]]


def rotate_90(grid: Grid) -> Grid:
    return np.rot90(grid, k=1)


def rotate_180(grid: Grid) -> Grid:
    return np.rot90(grid, k=2)


def rotate_270(grid: Grid) -> Grid:
    return np.rot90(grid, k=3)


def mirror_horizontally(grid: Grid) -> Grid:
    return np.maximum(grid, np.flipud(grid))


def recolor(from_color: int, to_color: int) -> Callable[[Grid], Grid]:
    def fn(grid: Grid) -> Grid:
        return np.where(grid == from_color, to_color, grid)
    return fn


def make_color_map_fn(mapping: dict) -> Callable[[Grid], Grid]:
    def fn(grid: Grid) -> Grid:
        result = np.zeros_like(grid)
        for k, v in mapping.items():
            result[grid == k] = v
        return result
    return fn


def swap_colors(grid: Grid) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    color1, color2 = colors
    return np.select([grid == color1, grid == color2], [color2, color1], grid)


def _collect_cells(
    grid: Grid, start_row: int, start_col: int, visited: set
) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    queue = [(start_row, start_col)]
    while queue:
        row, col = queue.pop(0)
        if (row, col) in visited or not (
            0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        ):
            continue
        if grid[row, col] == 0:
            continue
        visited.add((row, col))
        cells.add((row, col))
        queue.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])
    return frozenset(cells)


def _make_arc_object(cells: frozenset[tuple[int, int]]) -> ArcObject:
    min_row = min(row for row, col in cells)
    max_row = max(row for row, col in cells)
    min_col = min(col for row, col in cells)
    max_col = max(col for row, col in cells)
    return ArcObject(
        row=min_row,
        col=min_col,
        width=max_col - min_col + 1,
        height=max_row - min_row + 1,
        cells=cells,
    )


@lru_cache(maxsize=None)
def _get_objects_cached(grid_bytes: bytes, shape: tuple) -> tuple[ArcObject, ...]:
    grid = np.frombuffer(grid_bytes).reshape(shape)
    visited: set[tuple[int, int]] = set()
    objects = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] == 0 or (start_row, start_col) in visited:
                continue
            cells = _collect_cells(grid, start_row, start_col, visited)
            objects.append(_make_arc_object(cells))
    return tuple(objects)


def get_objects(grid: Grid) -> tuple[ArcObject, ...]:
    return _get_objects_cached(grid.tobytes(), grid.shape)


def crop_to_content(grid: Grid) -> Grid:
    objects = get_objects(grid)
    if not objects:
        return grid
    min_row = min(obj.row for obj in objects)
    max_row = max(obj.row + obj.height - 1 for obj in objects)
    min_col = min(obj.col for obj in objects)
    max_col = max(obj.col + obj.width - 1 for obj in objects)
    return grid[min_row : max_row + 1, min_col : max_col + 1]


def make_hollow(grid: Grid) -> Grid:
    all_cells = set(cell for obj in get_objects(grid) for cell in obj.cells)
    result = grid.copy()
    for row, col in all_cells:
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if all(n in all_cells for n in neighbors):
            result[row, col] = 0
    return result
