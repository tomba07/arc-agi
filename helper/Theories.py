from dataclasses import dataclass
from typing import Callable, List, Set, Tuple

import numpy as np

from helper.Observations import ProblemObservation

Grid = np.ndarray


@dataclass
class Theory:
    name: str
    apply: Callable[[Grid], Grid]

    def validate(self, examples: List[Tuple[Grid, Grid]]) -> bool:
        return all(np.array_equal(self.apply(inp), out) for inp, out in examples)


SYMMETRY_THEORIES = [
    Theory("rotate_90", lambda g: np.rot90(g, k=1)),
    Theory("rotate_180", lambda g: np.rot90(g, k=2)),
    Theory("rotate_270", lambda g: np.rot90(g, k=3)),
    Theory("flip_lr", lambda g: np.fliplr(g)),
    Theory("flip_ud", lambda g: np.flipud(g)),
    Theory("transpose", lambda g: g.T),
    Theory("anti_transpose", lambda g: np.rot90(g.T)),
]


def _crop_to_content(grid: Grid) -> Grid:
    rows, cols = np.where(grid != 0)
    if len(rows) == 0:
        return grid
    return grid[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


CROP_THEORY = Theory("crop_to_content", _crop_to_content)


def color_substitution_theories(
    problem: ProblemObservation,
) -> List[Theory]:
    first = problem.examples[0]
    all_colors = (
        {obj.color for obj in first.input_objects}
        | {obj.color for obj in first.output_objects}
        | {0}
    )
    source_colors = {obj.color for obj in first.input_objects}

    theories = []
    for a in source_colors:
        for b in all_colors - {a}:
            theories.append(
                Theory(
                    f"replace_{a}_with_{b}",
                    lambda g, a=a, b=b: np.where(g == a, b, g),
                )
            )
    return theories


_INDICATION_MAP: dict[str, Callable[[str], bool]] = {
    "color_substitution": lambda n: n.startswith("replace_"),
    "crop": lambda n: n == "crop_to_content",
    "symmetry": lambda n: n in {
        "rotate_90", "rotate_180", "rotate_270",
        "flip_lr", "flip_ud", "transpose", "anti_transpose",
    },
}


def _reorder(theories: List[Theory], indications: Set[str]) -> List[Theory]:
    if not indications:
        return theories
    indicated = [
        t for t in theories
        if any(_INDICATION_MAP[ind](t.name) for ind in indications if ind in _INDICATION_MAP)
    ]
    rest = [t for t in theories if t not in indicated]
    return [*indicated, *rest]


def generate_phase1_theories(
    problem: ProblemObservation, indications: Set[str] = frozenset()
) -> List[Theory]:
    all_theories = [*SYMMETRY_THEORIES, CROP_THEORY, *color_substitution_theories(problem)]
    return _reorder(all_theories, indications)


def generate_phase2_theories(
    problem: ProblemObservation, indications: Set[str] = frozenset()
) -> List[Theory]:
    return []


def generate_theories(problem: ProblemObservation) -> List[Theory]:
    return [*generate_phase1_theories(problem), *generate_phase2_theories(problem)]
