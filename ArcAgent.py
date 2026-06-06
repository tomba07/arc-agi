from typing import Callable, List

import numpy as np

from ArcProblem import ArcProblem

Grid = np.ndarray


def _rotate_90(grid: Grid) -> Grid:
    return np.rot90(grid, k=1)


def _rotate_180(grid: Grid) -> Grid:
    return np.rot90(grid, k=2)


def _rotate_270(grid: Grid) -> Grid:
    return np.rot90(grid, k=3)


def _flip_lr(grid: Grid) -> Grid:
    return np.fliplr(grid)


def _flip_ud(grid: Grid) -> Grid:
    return np.flipud(grid)


def _transpose(grid: Grid) -> Grid:
    return grid.T


def _anti_transpose(grid: Grid) -> Grid:
    return np.rot90(grid.T)


def _overlay_flip_ud(grid: Grid) -> Grid:
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


def _color_replace_and_erase(grid: Grid, replace_color: int) -> Grid:
    other_colors = sorted(set(np.unique(grid)) - {0, replace_color})
    if len(other_colors) != 1:
        return grid
    other_color = other_colors[0]
    return np.select(
        [grid == replace_color, grid == other_color], [other_color, 0], grid
    )


def _crop_to_content(grid: Grid) -> Grid:
    rows, cols = np.where(grid != 0)
    if len(rows) == 0:
        return grid
    return grid[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _make_hollow(grid: Grid) -> Grid:
    interior = np.zeros(grid.shape, dtype=bool)
    interior[1:-1, 1:-1] = (
        (grid[1:-1, 1:-1] != 0)
        & (grid[:-2, 1:-1] != 0)
        & (grid[2:, 1:-1] != 0)
        & (grid[1:-1, :-2] != 0)
        & (grid[1:-1, 2:] != 0)
    )
    return np.where(interior, 0, grid)


class ArcAgent:
    def __init__(self):
        pass

    def _generate_theories(self) -> List[Callable[[Grid], Grid]]:
        transforms: List[Callable[[Grid], Grid]] = [
            _rotate_90,
            _rotate_180,
            _rotate_270,
            _flip_lr,
            _flip_ud,
            _transpose,
            _anti_transpose,
            _overlay_flip_ud,
            _swap_colors,
            _crop_to_content,
            _make_hollow,
        ]
        for from_color in range(1, 10):
            for to_color in range(0, 10):
                if from_color != to_color:
                    transforms.append(_recolor(from_color, to_color))
            transforms.append(lambda grid, a=from_color: _color_replace_and_erase(grid, a))
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
