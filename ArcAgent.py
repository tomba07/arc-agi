from typing import Callable, List, Optional
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ArcProblem import ArcProblem

Grid = np.ndarray
Theory = list[Callable[[Grid], Grid]]


def _apply_theory(theory: Theory, grid: Grid) -> Grid:
    for fn in theory:
        grid = fn(grid)
    return grid


@dataclass
class Observations:
    same_size: bool
    size_decreases: bool
    same_color_set: bool
    recolor_pairs: list[tuple[int, int]]
    color_map: Optional[
        dict
    ]


def _detect_color_map(examples: list) -> Optional[dict]:
    """Derive a consistent cell-level color mapping from input to output."""
    mapping: dict = {}
    for inp, out in examples:
        if inp.shape != out.shape:
            return None
        for vi, vo in zip(inp.flat, out.flat):
            vi, vo = int(vi), int(vo)
            if vi in mapping:
                if mapping[vi] != vo:
                    return None
            else:
                mapping[vi] = vo

    if not mapping or all(k == v for k, v in mapping.items()):
        return None
    return mapping


def _observe(examples: list) -> Observations:
    same_size = all(inp.shape == out.shape for inp, out in examples)
    size_decreases = any(inp.size > out.size for inp, out in examples)

    source_colors: set[int] = set()
    target_colors: set[int] = set()
    for inp, out in examples:
        in_colors = set(int(c) for c in np.unique(inp) if c != 0)
        out_colors = set(int(c) for c in np.unique(out) if c != 0)
        source_colors |= in_colors - out_colors
        target_colors |= out_colors - in_colors

    same_color_set = not source_colors and not target_colors

    recolor_pairs: list[tuple[int, int]] = []
    for fc in source_colors:
        if target_colors:
            for tc in target_colors:
                recolor_pairs.append((fc, tc))
        else:
            recolor_pairs.append((fc, 0))

    return Observations(
        same_size=same_size,
        size_decreases=size_decreases,
        same_color_set=same_color_set,
        recolor_pairs=recolor_pairs,
        color_map=_detect_color_map(examples),
    )


@dataclass
class ArcObject:
    row: int
    col: int
    width: int
    height: int
    cells: frozenset[tuple[int, int]]


def _rotate_90(grid: Grid) -> Grid:
    return np.rot90(grid, k=1)


def _rotate_180(grid: Grid) -> Grid:
    return np.rot90(grid, k=2)


def _rotate_270(grid: Grid) -> Grid:
    return np.rot90(grid, k=3)


def _mirror_horizontally(grid: Grid) -> Grid:
    return np.maximum(grid, np.flipud(grid))


def _recolor(from_color: int, to_color: int) -> Callable[[Grid], Grid]:
    def fn(grid: Grid) -> Grid:
        return np.where(grid == from_color, to_color, grid)

    return fn


def _make_color_map_fn(mapping: dict) -> Callable[[Grid], Grid]:
    def fn(grid: Grid) -> Grid:
        result = np.zeros_like(grid)
        for k, v in mapping.items():
            result[grid == k] = v
        return result

    return fn


def _swap_colors(grid: Grid) -> Grid:
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


def _get_objects(grid: Grid) -> tuple[ArcObject, ...]:
    return _get_objects_cached(grid.tobytes(), grid.shape)


def _crop_to_content(grid: Grid) -> Grid:
    objects = _get_objects(grid)

    if not objects:
        return grid

    min_row = min(obj.row for obj in objects)
    max_row = max(obj.row + obj.height - 1 for obj in objects)
    min_col = min(obj.col for obj in objects)
    max_col = max(obj.col + obj.width - 1 for obj in objects)

    return grid[min_row : max_row + 1, min_col : max_col + 1]


def _make_hollow(grid: Grid) -> Grid:
    all_cells = set(cell for obj in _get_objects(grid) for cell in obj.cells)
    result = grid.copy()

    for row, col in all_cells:
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if all(n in all_cells for n in neighbors):
            result[row, col] = 0

    return result


_SAME_SIZE_THEORIES: list[Theory] = [
    [_rotate_90],
    [_rotate_180],
    [_rotate_270],
    [_mirror_horizontally],
    [_swap_colors],
    [_make_hollow],
]

_SIZE_REDUCING_THEORIES: list[Theory] = [
    [_crop_to_content],
    [_crop_to_content, _swap_colors],
]


class ArcAgent:
    def __init__(self):
        pass

    def _get_theories(self, obs: Observations) -> list[Theory]:
        theories: list[Theory] = []

        if obs.same_size:
            theories.extend(_SAME_SIZE_THEORIES)

        if obs.size_decreases:
            theories.extend(_SIZE_REDUCING_THEORIES)

        if obs.color_map:
            theories.append([_make_color_map_fn(obs.color_map)])

        for fc, tc in obs.recolor_pairs:
            theories.append([_recolor(fc, tc)])
            if obs.same_size:
                theories.append([_swap_colors, _recolor(fc, tc)])

        return theories

    def _validate_theory(self, theory: Theory, examples) -> bool:
        try:
            return all(
                np.array_equal(_apply_theory(theory, inp), out) for inp, out in examples
            )
        except Exception:
            return False

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = [
            (example.get_input_data().data(), example.get_output_data().data())
            for example in arc_problem.training_set()
        ]
        test_input = arc_problem.test_set().get_input_data().data()

        obs = _observe(examples)

        for theory in self._get_theories(obs):
            if self._validate_theory(theory, examples):
                print(f"{arc_problem.problem_name()}: matched")
                return [_apply_theory(theory, test_input)]

        print(f"{arc_problem.problem_name()}: no match")
        return []
