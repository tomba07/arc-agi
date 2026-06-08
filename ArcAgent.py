from typing import Callable, List
from dataclasses import dataclass

import numpy as np

from ArcProblem import ArcProblem

Grid = np.ndarray


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


def _swap_colors(grid: Grid) -> Grid:
    colors = sorted(set(np.unique(grid)) - {0})

    if len(colors) != 2:
        return grid
    color1, color2 = colors

    return np.select([grid == color1, grid == color2], [color2, color1], grid)


def _collect_cells(grid: Grid, start_row: int, start_col: int, visited: set) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    queue = [(start_row, start_col)]
    while queue:
        row, col = queue.pop(0)
        if (row, col) in visited or not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
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


def _get_objects(grid: Grid) -> list[ArcObject]:
    visited: set[tuple[int, int]] = set()
    objects = []

    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] == 0 or (start_row, start_col) in visited:
                continue
            cells = _collect_cells(grid, start_row, start_col, visited)
            objects.append(_make_arc_object(cells))

    return objects


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


class ArcAgent:
    def __init__(self):
        pass

    def _generate_theories(self) -> List[Callable[[Grid], Grid]]:
        transforms: List[Callable[[Grid], Grid]] = [
            _rotate_90,
            _rotate_180,
            _rotate_270,
            _swap_colors,
            _mirror_horizontally,
            _crop_to_content,
            _make_hollow,
        ]
        for from_color in range(1, 10):
            for to_color in range(0, 10):
                if from_color != to_color:
                    transforms.append(_recolor(from_color, to_color))
        return transforms

    def _validate_theory(self, fn, examples):
        try:
            return all(np.array_equal(fn(inp), out) for inp, out in examples)
        except Exception:
            return False

    def _validate_theories(self, theories, examples, test_input):
        for theory in theories:
            if self._validate_theory(theory, examples):
                return theory(test_input)

        return None

    def _validate_composed_theories(self, theories, examples, test_input):
        for level1 in theories:
            for level2 in theories:
                try:
                    if all(
                        np.array_equal(level2(level1(inp)), out)
                        for inp, out in examples
                    ):
                        return level2(level1(test_input))
                except Exception:
                    continue

        return None

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        examples = [
            (example.get_input_data().data(), example.get_output_data().data())
            for example in arc_problem.training_set()
        ]
        test_input = arc_problem.test_set().get_input_data().data()

        theories = self._generate_theories()

        result = self._validate_theories(theories, examples, test_input)
        if result is None:
            result = self._validate_composed_theories(theories, examples, test_input)

        if result is not None:
            print(f"{arc_problem.problem_name()}: matched")

            return [result]

        print(f"{arc_problem.problem_name()}: no match")

        return []
