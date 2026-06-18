from typing import Optional
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

Grid = np.ndarray


@dataclass
class Shape:
    row: int
    col: int
    width: int
    height: int
    color: int
    cells: frozenset[tuple[int, int]]


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


def _make_shape(grid: Grid, cells: frozenset[tuple[int, int]]) -> Shape:
    min_row = min(row for row, col in cells)
    max_row = max(row for row, col in cells)
    min_col = min(col for row, col in cells)
    max_col = max(col for row, col in cells)
    return Shape(
        row=min_row,
        col=min_col,
        width=max_col - min_col + 1,
        height=max_row - min_row + 1,
        color=int(grid[min_row, min_col]),
        cells=cells,
    )


@lru_cache(maxsize=None)
def _shape_cached(grid_bytes: bytes, grid_shape: tuple) -> tuple[Shape, ...]:
    grid = np.frombuffer(grid_bytes).reshape(grid_shape)
    visited: set[tuple[int, int]] = set()
    objects = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] == 0 or (start_row, start_col) in visited:
                continue
            cells = _collect_cells(grid, start_row, start_col, visited)
            objects.append(_make_shape(grid, cells))
    return tuple(objects)


def shape(grid: Grid) -> tuple[Shape, ...]:
    return _shape_cached(grid.tobytes(), grid.shape)


@dataclass
class ExampleObservation:
    input: Grid
    output: Grid
    input_shapes: tuple[Shape, ...]
    output_shapes: tuple[Shape, ...]


@dataclass
class TestObservation:
    input: Grid
    shapes: tuple[Shape, ...]


@dataclass
class Observations:
    examples: list[ExampleObservation]
    test: TestObservation
    same_size: bool
    size_decreases: bool
    recolor_pairs: list[tuple[int, int]]
    color_map: Optional[dict]
    shapes: tuple[Shape, ...] = ()  # active shapes, set by ArcAgent before each apply


def _detect_color_map(examples: list[ExampleObservation]) -> Optional[dict]:
    mapping: dict = {}
    for ex in examples:
        if ex.input.shape != ex.output.shape:
            return None
        for vi, vo in zip(ex.input.flat, ex.output.flat):
            vi, vo = int(vi), int(vo)
            if vi in mapping:
                if mapping[vi] != vo:
                    return None
            else:
                mapping[vi] = vo
    if not mapping or all(k == v for k, v in mapping.items()):
        return None
    return mapping


def observe(examples: list[tuple[Grid, Grid]], test_input: Grid) -> Observations:
    example_obs = [
        ExampleObservation(
            input=inp,
            output=out,
            input_shapes=shape(inp),
            output_shapes=shape(out),
        )
        for inp, out in examples
    ]

    same_size = all(ex.input.shape == ex.output.shape for ex in example_obs)
    size_decreases = any(ex.input.size > ex.output.size for ex in example_obs)

    source_colors: set[int] = set()
    target_colors: set[int] = set()
    for ex in example_obs:
        in_colors = set(int(c) for c in np.unique(ex.input) if c != 0)
        out_colors = set(int(c) for c in np.unique(ex.output) if c != 0)
        source_colors |= in_colors - out_colors
        target_colors |= out_colors - in_colors

    recolor_pairs: list[tuple[int, int]] = []
    for fc in source_colors:
        if target_colors:
            for tc in target_colors:
                recolor_pairs.append((fc, tc))
        else:
            recolor_pairs.append((fc, 0))

    return Observations(
        examples=example_obs,
        test=TestObservation(input=test_input, shapes=shape(test_input)),
        same_size=same_size,
        size_decreases=size_decreases,
        recolor_pairs=recolor_pairs,
        color_map=_detect_color_map(example_obs),
    )
