from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from helper.Indications import Indication
from helper.Observations import ProblemObservation
from helper.Transformations import (
    boolean_combine,
    grow_cells,
    make_hollow,
)

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


def color_substitution_theories(problem: ProblemObservation) -> List[Theory]:
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
    "hollow": lambda n: n == "make_hollow",
    "expand": lambda n: n.startswith("grow_"),
    "divider": lambda n: n.startswith("divider_"),
}


def _score_and_sort(theories: List[Theory], indications: List[Indication]) -> List[Theory]:
    score_map = {ind.name: ind.confidence for ind in indications}

    def score(t: Theory) -> float:
        return max(
            (score_map[name] for name, match in _INDICATION_MAP.items()
             if name in score_map and match(t.name)),
            default=0.0,
        )

    return sorted(theories, key=score, reverse=True)


def _build_all_theories(problem: ProblemObservation) -> List[Theory]:
    theories: List[Theory] = [
        *SYMMETRY_THEORIES,
        CROP_THEORY,
        *color_substitution_theories(problem),
    ]

    theories.append(Theory("make_hollow", make_hollow))

    first = problem.examples[0]
    if first.output_objects:
        out_obj = first.output_objects[0]
        scale = out_obj.height
        new_color = out_obj.color
        theories.append(Theory(
            f"grow_{scale}x_to_{new_color}",
            lambda g, s=scale, c=new_color: grow_cells(g, s, c),
        ))

    if first.divider_axis is not None:
        output_color = next(iter(first.colors_added), 1)
        axis = first.divider_axis
        idx = first.divider_index

        def _split(g, a=axis, i=idx):
            if a == "col":
                return g[:, :i], g[:, i + 1:]
            return g[:i, :], g[i + 1:, :]

        for op in ["and", "or", "nor"]:
            theories.append(Theory(
                f"divider_{op}",
                lambda g, op=op, oc=output_color: boolean_combine(*_split(g), op, oc),
            ))

    return theories


def generate_theories(
    problem: ProblemObservation, indications: List[Indication]
) -> List[Theory]:
    return _score_and_sort(_build_all_theories(problem), indications)
