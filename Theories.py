import itertools

import numpy as np

from Transformations import (
    Primitive, Program, ApplyState, BASE_PRIMITIVES,
    make_recolor, make_apply_color_map,
)
from Observations import Observations, ExampleObservation


def synthesize(obs: Observations) -> list[Program]:
    candidates: list[Primitive] = list(BASE_PRIMITIVES)
    for fc, tc in obs.recolor_pairs:
        candidates.append(make_recolor(fc, tc))
    if obs.color_map:
        candidates.append(make_apply_color_map(obs.color_map))

    for length in range(1, 5):
        for combo in itertools.product(candidates, repeat=length):
            prog = list(combo)
            if _validates(prog, obs.examples):
                return [prog]
    return []


def _validates(prog: Program, examples: list[ExampleObservation]) -> bool:
    for ex in examples:
        state = ApplyState(
            grid=ex.input.copy(),
            source_grid=ex.input,
            source_shapes=ex.input_shapes,
        )
        try:
            for step in prog:
                state = step(state)
        except Exception:
            return False
        if not np.array_equal(state.grid, ex.output):
            return False
    return True
