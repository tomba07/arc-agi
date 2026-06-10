from dataclasses import dataclass
from typing import Callable, Generator, List, Tuple

import numpy as np

from helper.DeltaAbstraction import derive_object_theories
from helper.Observations import ProblemObservation
from helper.Transformations import (
    boolean_combine,
    erode,
    mask_subtract,
    ray_fill,
    DIRECTIONS_DIAGONAL,
    DIRECTIONS_ORTHOGONAL,
    DIRECTIONS_ALL_8,
    draw_clockwise_spiral,
    remap_replace_keep,
    swap_two_nonzero,
)

Grid = np.ndarray


@dataclass
class Theory:
    name: str
    weight: float
    apply: Callable[[Grid], Grid]


@dataclass
class Primitive:
    name: str
    instantiate: Callable[
        [ProblemObservation, List[Tuple[Grid, Grid]]],
        List[Tuple[str, Callable[[Grid], Grid]]],
    ]


def _crop_to_content(grid: Grid) -> Grid:
    rows, cols = np.where(grid != 0)
    if len(rows) == 0:
        return grid
    return grid[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]


def soft_match_score(apply_fn: Callable[[Grid], Grid], examples: List[Tuple[Grid, Grid]]) -> float:
    scores = []
    for inp, out in examples:
        try:
            pred = apply_fn(inp)
            if pred.shape == out.shape:
                scores.append(float(np.mean(pred == out)))
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _always(name: str, fn: Callable[[Grid], Grid]) -> Primitive:
    return Primitive(name, lambda p, ex, _n=name, _f=fn: [(_n, _f)])


def _remap_instances(problem: ProblemObservation, examples: List[Tuple[Grid, Grid]]) -> List[Tuple[str, Callable]]:
    all_input_colors = [{obj.color for obj in e.input_objects} for e in problem.examples]
    all_output_colors = [{obj.color for obj in e.output_objects} for e in problem.examples]
    if not all_input_colors or not all_output_colors:
        return []
    always_in = set.intersection(*all_input_colors)
    never_out = always_in - set.union(*all_output_colors)
    return [
        (f"remap_replace_{sc}", lambda g, sc=sc: remap_replace_keep(g, sc))
        for sc in never_out
    ]


def _boolean_instances(problem: ProblemObservation, examples: List[Tuple[Grid, Grid]]) -> List[Tuple[str, Callable]]:
    if not problem.has_divider or not problem.examples:
        return []
    first = problem.examples[0]
    output_color = next(iter(first.colors_added), 1)
    axis = first.divider_axis
    idx = first.divider_index

    def _split(g, a=axis, i=idx):
        if a == "col":
            return g[:, :i], g[:, i + 1:]
        return g[:i, :], g[i + 1:, :]

    return [
        (f"divider_{op}", lambda g, op=op, oc=output_color: boolean_combine(*_split(g), op, oc))
        for op in ["and", "or", "nor"]
    ]


PRIMITIVES: List[Primitive] = [
    _always("rotate_90",           lambda g: np.rot90(g, k=1)),
    _always("rotate_180",          lambda g: np.rot90(g, k=2)),
    _always("rotate_270",          lambda g: np.rot90(g, k=3)),
    _always("flip_lr",             lambda g: np.fliplr(g)),
    _always("flip_ud",             lambda g: np.flipud(g)),
    _always("transpose",           lambda g: g.T),
    _always("anti_transpose",      lambda g: np.rot90(g.T)),
    _always("crop_to_content",     _crop_to_content),
    _always("erode",               erode),
    _always("swap_two_nonzero",    swap_two_nonzero),
    _always("overlay_flip_ud",     lambda g: np.maximum(g, np.flipud(g))),
    Primitive("ray_fill", lambda p, ex: [
        ("ray_fill_diagonal",   lambda g: ray_fill(g, DIRECTIONS_DIAGONAL)),
        ("ray_fill_orthogonal", lambda g: ray_fill(g, DIRECTIONS_ORTHOGONAL)),
        ("ray_fill_all8",       lambda g: ray_fill(g, DIRECTIONS_ALL_8)),
    ]),
    _always("draw_clockwise_spiral", draw_clockwise_spiral),
    Primitive("remap_replace",     _remap_instances),
    Primitive("boolean_combine",   _boolean_instances),
]


def generate_candidates(
    problem: ProblemObservation,
    examples: List[Tuple[Grid, Grid]],
) -> List[Theory]:
    theories: List[Theory] = []
    base: List[Tuple[str, Callable[[Grid], Grid]]] = []

    for prim in PRIMITIVES:
        for name, apply_fn in prim.instantiate(problem, examples):
            score = soft_match_score(apply_fn, examples)
            theories.append(Theory(name, score, apply_fn))
            base.append((name, apply_fn))

    for name, apply_fn in derive_object_theories(examples):
        score = soft_match_score(apply_fn, examples)
        theories.append(Theory(name, score, apply_fn))
        base.append((name, apply_fn))

    for name, fn in base:
        def _residual(g, f=fn):
            transformed = f(g)
            if transformed.shape != g.shape:
                return g
            return mask_subtract(g, transformed)
        score = soft_match_score(_residual, examples)
        theories.append(Theory(f"original_minus_{name}", score, _residual))

    return theories


def search_compositions(
    theories: List[Theory],
    examples: List[Tuple[Grid, Grid]],
    max_depth: int = 2,
) -> Generator[Tuple[str, Callable[[Grid], Grid], int], None, None]:
    def validates(fn: Callable[[Grid], Grid]) -> bool:
        return all(np.array_equal(fn(inp), out) for inp, out in examples)

    tried = 0
    for t in sorted(theories, key=lambda t: t.weight, reverse=True):
        tried += 1
        if validates(t.apply):
            yield t.name, t.apply, tried

    if max_depth < 2:
        return

    pairs = sorted(
        [(t1, t2) for t1 in theories for t2 in theories],
        key=lambda p: p[0].weight + p[1].weight,
        reverse=True,
    )
    for t1, t2 in pairs:
        tried += 1
        composed = lambda g, a=t1.apply, b=t2.apply: b(a(g))
        if validates(composed):
            yield f"{t1.name}+{t2.name}", composed, tried
