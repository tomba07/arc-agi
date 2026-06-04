from dataclasses import dataclass
from typing import Callable, Generator, List, Tuple

import numpy as np

from helper.Observations import ProblemObservation
from helper.Transformations import (
    boolean_combine,
    ray_fill,
    DIRECTIONS_DIAGONAL,
    DIRECTIONS_ORTHOGONAL,
    DIRECTIONS_ALL_8,
    draw_clockwise_spiral,
    grow_cells,
    make_hollow,
    remap_replace_keep,
    snap_matching_to_walls,
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


def _derive_wall_map(inp: Grid):
    rows, cols = inp.shape
    mid_r, mid_c = rows // 2, cols // 2
    top    = int(inp[0, mid_c])
    bottom = int(inp[rows - 1, mid_c])
    left   = int(inp[mid_r, 0])
    right  = int(inp[mid_r, cols - 1])
    if 0 in {top, bottom, left, right}:
        return None
    if len({top, bottom, left, right}) != 4:
        return None
    if not (
        np.all(inp[0, 1:-1] == top)
        and np.all(inp[rows - 1, 1:-1] == bottom)
        and np.all(inp[1:-1, 0] == left)
        and np.all(inp[1:-1, cols - 1] == right)
    ):
        return None
    return {
        top:    ("row", 1),
        bottom: ("row", rows - 2),
        left:   ("col", 1),
        right:  ("col", cols - 2),
    }


def _snap_apply(g: Grid) -> Grid:
    wm = _derive_wall_map(g)
    if wm is None:
        return g
    return snap_matching_to_walls(g, wm)


def _snap_instances(problem: ProblemObservation, examples: List[Tuple[Grid, Grid]]) -> List[Tuple[str, Callable]]:
    if any(_derive_wall_map(inp) is None for inp, _ in examples):
        return []
    return [("snap_matching_to_walls", _snap_apply)]


def _recolor_instances(problem: ProblemObservation, examples: List[Tuple[Grid, Grid]]) -> List[Tuple[str, Callable]]:
    if not problem.examples:
        return []
    first = problem.examples[0]
    input_colors = {obj.color for obj in first.input_objects}
    output_colors = {obj.color for obj in first.output_objects} | {0}
    return [
        (f"replace_{a}_with_{b}", lambda g, a=a, b=b: np.where(g == a, b, g).astype(g.dtype))
        for a in input_colors
        for b in output_colors - {a}
    ]


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


def _grow_instances(problem: ProblemObservation, examples: List[Tuple[Grid, Grid]]) -> List[Tuple[str, Callable]]:
    if not problem.all_single_cells or not problem.examples:
        return []
    first = problem.examples[0]
    if not first.output_objects:
        return []
    instances = []
    seen = set()
    for obj in first.output_objects:
        key = (obj.height, obj.color)
        if key not in seen:
            seen.add(key)
            scale, color = obj.height, obj.color
            instances.append((
                f"grow_{scale}x_to_{color}",
                lambda g, s=scale, c=color: grow_cells(g, s, c),
            ))
    return instances


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
    _always("rotate_90",      lambda g: np.rot90(g, k=1)),
    _always("rotate_180",     lambda g: np.rot90(g, k=2)),
    _always("rotate_270",     lambda g: np.rot90(g, k=3)),
    _always("flip_lr",        lambda g: np.fliplr(g)),
    _always("flip_ud",        lambda g: np.flipud(g)),
    _always("transpose",      lambda g: g.T),
    _always("anti_transpose", lambda g: np.rot90(g.T)),
    _always("crop_to_content", _crop_to_content),
    _always("make_hollow",    make_hollow),
    _always("swap_two_nonzero", swap_two_nonzero),
    _always("overlay_flip_ud", lambda g: np.maximum(g, np.flipud(g))),
    Primitive("ray_fill", lambda p, ex: [
        ("ray_fill_diagonal",   lambda g: ray_fill(g, DIRECTIONS_DIAGONAL)),
        ("ray_fill_orthogonal", lambda g: ray_fill(g, DIRECTIONS_ORTHOGONAL)),
        ("ray_fill_all8",       lambda g: ray_fill(g, DIRECTIONS_ALL_8)),
    ]),
    _always("draw_clockwise_spiral", draw_clockwise_spiral),
    Primitive("recolor",         _recolor_instances),
    Primitive("remap_replace",   _remap_instances),
    Primitive("grow_cells",      _grow_instances),
    Primitive("boolean_combine", _boolean_instances),
    Primitive("snap_to_walls",   _snap_instances),
]


def generate_candidates(
    problem: ProblemObservation,
    examples: List[Tuple[Grid, Grid]],
) -> List[Theory]:
    theories: List[Theory] = []
    for prim in PRIMITIVES:
        for name, apply_fn in prim.instantiate(problem, examples):
            score = soft_match_score(apply_fn, examples)
            theories.append(Theory(name, score, apply_fn))
    return theories


def search_compositions(
    theories: List[Theory],
    examples: List[Tuple[Grid, Grid]],
    max_depth: int = 2,
) -> Generator[Tuple[str, Callable[[Grid], Grid]], None, None]:
    def validates(fn: Callable[[Grid], Grid]) -> bool:
        return all(np.array_equal(fn(inp), out) for inp, out in examples)

    for t in sorted(theories, key=lambda t: t.weight, reverse=True):
        if validates(t.apply):
            yield t.name, t.apply

    if max_depth < 2:
        return

    pairs = sorted(
        [(t1, t2) for t1 in theories for t2 in theories],
        key=lambda p: p[0].weight + p[1].weight,
        reverse=True,
    )
    for t1, t2 in pairs:
        composed = lambda g, a=t1.apply, b=t2.apply: b(a(g))
        if validates(composed):
            yield f"{t1.name}+{t2.name}", composed
