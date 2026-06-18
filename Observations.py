from typing import Optional
from dataclasses import dataclass

import numpy as np

from Transformations import Grid


@dataclass
class Observations:
    same_size: bool
    size_decreases: bool
    recolor_pairs: list[tuple[int, int]]
    color_map: Optional[dict]


def _detect_color_map(examples: list) -> Optional[dict]:
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


def observe(examples: list) -> Observations:
    same_size = all(inp.shape == out.shape for inp, out in examples)
    size_decreases = any(inp.size > out.size for inp, out in examples)

    source_colors: set[int] = set()
    target_colors: set[int] = set()
    for inp, out in examples:
        in_colors = set(int(c) for c in np.unique(inp) if c != 0)
        out_colors = set(int(c) for c in np.unique(out) if c != 0)
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
        same_size=same_size,
        size_decreases=size_decreases,
        recolor_pairs=recolor_pairs,
        color_map=_detect_color_map(examples),
    )
