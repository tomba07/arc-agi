from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from helper.ObjectUtils import find_objects

Grid = np.ndarray


# ---------------------------------------------------------------------------
# Rule dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ColorMapRule:
    """Each object keeps its position/shape; its color is remapped."""
    mapping: dict  # {input_color: output_color}


@dataclass
class TranslationRule:
    """Every object is shifted by the same (Δrow, Δcol)."""
    delta_row: int
    delta_col: int


@dataclass
class SnapToBorderRule:
    """Each interior object moves to the inner edge of the border whose color matches it."""


@dataclass
class ScaleRule:
    """Each single-cell object is expanded to a scale×scale block of new_color."""
    scale: int
    new_color: int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _derive_wall_map(inp: Grid) -> Optional[dict]:
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


def _place_object(
    result: Grid,
    cells,
    origin_row: int,
    origin_col: int,
    new_row: int,
    new_col: int,
    color: int,
    scale: int = 1,
) -> None:
    rows, cols = result.shape
    for r, c in cells:
        dr = r - origin_row
        dc = c - origin_col
        for sr in range(scale):
            for sc in range(scale):
                nr = new_row + dr * scale + sr
                nc = new_col + dc * scale + sc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr, nc] = color


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

def _try_color_map_rule(examples: List[Tuple[Grid, Grid]]) -> Optional[ColorMapRule]:
    """Color changes where every object keeps its exact position and shape."""
    mapping: dict = {}
    for inp, out in examples:
        inp_objs = find_objects(inp.tolist())
        out_objs = find_objects(out.tolist())
        out_by_bbox = {obj.bounding_box: obj for obj in out_objs}
        for obj in inp_objs:
            match = out_by_bbox.get(obj.bounding_box)
            if match is None:
                return None
            if match.normalized_cells != obj.normalized_cells:
                return None
            if obj.color == match.color:
                continue
            if obj.color in mapping and mapping[obj.color] != match.color:
                return None
            mapping[obj.color] = match.color
    return ColorMapRule(mapping) if mapping else None


def _try_translation_rule(examples: List[Tuple[Grid, Grid]]) -> Optional[TranslationRule]:
    """All objects shift by the same constant (Δrow, Δcol)."""
    delta: Optional[Tuple[int, int]] = None
    for inp, out in examples:
        inp_objs = find_objects(inp.tolist())
        out_objs = find_objects(out.tolist())
        if len(inp_objs) != len(out_objs):
            return None
        for obj in inp_objs:
            match = next(
                (o for o in out_objs
                 if o.color == obj.color and o.normalized_cells == obj.normalized_cells),
                None,
            )
            if match is None:
                return None
            dr = match.bounding_box[0] - obj.bounding_box[0]
            dc = match.bounding_box[1] - obj.bounding_box[1]
            if delta is None:
                delta = (dr, dc)
            elif delta != (dr, dc):
                return None
    if delta is None or delta == (0, 0):
        return None
    return TranslationRule(delta[0], delta[1])


def _try_snap_to_border_rule(examples: List[Tuple[Grid, Grid]]) -> Optional[SnapToBorderRule]:
    """Interior objects snap to the inner edge of the border row/col matching their color.
    Objects whose color does not match any wall are discarded.
    """
    for inp, out in examples:
        wall_map = _derive_wall_map(inp)
        if wall_map is None:
            return None
        inp_objs = find_objects(inp.tolist())
        out_objs = find_objects(out.tolist())
        out_cell_set = {cell for obj in out_objs for cell in obj.cells}
        for obj in inp_objs:
            if obj.touches_border:
                continue
            if obj.color not in wall_map:
                continue  # non-matching objects are expected to be discarded
            kind, pos = wall_map[obj.color]
            min_row, min_col = obj.bounding_box[0], obj.bounding_box[1]
            new_row = pos if kind == "row" else min_row
            new_col = pos if kind == "col" else min_col
            for r, c in obj.cells:
                expected = (r - min_row + new_row, c - min_col + new_col)
                if expected not in out_cell_set:
                    return None
    return SnapToBorderRule()


def _try_scale_rule(examples: List[Tuple[Grid, Grid]]) -> Optional[ScaleRule]:
    """Single-cell inputs expand to scale×scale blocks of a fixed output color.
    Scale and color are derived from example 0; correctness is validated by soft_match_score.
    """
    inp0, out0 = examples[0]
    if inp0.shape != out0.shape:
        return None
    inp_objs = find_objects(inp0.tolist())
    out_objs = find_objects(out0.tolist())
    if not inp_objs or not out_objs:
        return None
    if not all(obj.area == 1 for obj in inp_objs):
        return None
    r0, c0 = inp_objs[0].bounding_box[0], inp_objs[0].bounding_box[1]
    match = next(
        (o for o in out_objs
         if o.bounding_box[0] <= r0 <= o.bounding_box[2]
         and o.bounding_box[1] <= c0 <= o.bounding_box[3]),
        None,
    )
    if match is None:
        return None
    scale = match.height
    if match.width != scale or match.area != scale * scale or scale <= 1:
        return None
    return ScaleRule(scale, match.color)


# ---------------------------------------------------------------------------
# Function synthesis
# ---------------------------------------------------------------------------

def _synthesize_color_map(rule: ColorMapRule) -> Callable[[Grid], Grid]:
    def apply(grid: Grid) -> Grid:
        result = grid.copy()
        for src, dst in rule.mapping.items():
            result = np.where(result == src, dst, result)
        return result.astype(grid.dtype)
    return apply


def _synthesize_translation(rule: TranslationRule) -> Callable[[Grid], Grid]:
    def apply(grid: Grid) -> Grid:
        result = np.zeros_like(grid)
        for obj in find_objects(grid.tolist()):
            min_row, min_col = obj.bounding_box[0], obj.bounding_box[1]
            _place_object(
                result, obj.cells,
                min_row, min_col,
                min_row + rule.delta_row, min_col + rule.delta_col,
                obj.color,
            )
        return result
    return apply


def _synthesize_snap_to_border(rule: SnapToBorderRule) -> Callable[[Grid], Grid]:
    def apply(grid: Grid) -> Grid:
        wall_map = _derive_wall_map(grid)
        if wall_map is None:
            return grid
        rows, cols = grid.shape
        result = grid.copy()
        result[1:rows - 1, 1:cols - 1] = 0
        for obj in find_objects(grid.tolist()):
            if obj.touches_border or obj.color not in wall_map:
                continue
            kind, pos = wall_map[obj.color]
            min_row, min_col = obj.bounding_box[0], obj.bounding_box[1]
            new_row = pos if kind == "row" else min_row
            new_col = pos if kind == "col" else min_col
            _place_object(result, obj.cells, min_row, min_col, new_row, new_col, obj.color)
        return result
    return apply


def _synthesize_scale(rule: ScaleRule) -> Callable[[Grid], Grid]:
    from helper.Transformations import dilate_square, recolor_nonzero
    def apply(grid: Grid) -> Grid:
        return recolor_nonzero(dilate_square(grid, rule.scale // 2), rule.new_color)
    return apply


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_EXTRACTORS = [
    ("color_map",       _try_color_map_rule,       _synthesize_color_map),
    ("translation",     _try_translation_rule,     _synthesize_translation),
    ("snap_to_border",  _try_snap_to_border_rule,  _synthesize_snap_to_border),
    ("scale",           _try_scale_rule,            _synthesize_scale),
]


def derive_object_theories(
    examples: List[Tuple[Grid, Grid]],
) -> List[Tuple[str, Callable[[Grid], Grid]]]:
    """Return (name, apply_fn) for every object-level rule consistent with all examples."""
    results = []
    for family, extractor, synthesizer in _EXTRACTORS:
        rule = extractor(examples)
        if rule is None:
            continue
        name = _rule_name(family, rule)
        results.append((name, synthesizer(rule)))
    return results


def _rule_name(family: str, rule) -> str:
    if isinstance(rule, ColorMapRule):
        pairs = ",".join(f"{k}→{v}" for k, v in sorted(rule.mapping.items()))
        return f"color_map({pairs})"
    if isinstance(rule, TranslationRule):
        return f"translate({rule.delta_row:+d},{rule.delta_col:+d})"
    if isinstance(rule, SnapToBorderRule):
        return "snap_to_border"
    if isinstance(rule, ScaleRule):
        return f"scale_{rule.scale}x_color_{rule.new_color}"
    return family
