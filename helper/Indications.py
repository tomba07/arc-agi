from dataclasses import dataclass
from typing import List

from helper.Observations import ProblemObservation


@dataclass
class Indication:
    name: str
    confidence: float  # 0.0 (weak hint) → 1.0 (certain)


def compute_indications(problem: ProblemObservation) -> List[Indication]:
    indications = []

    if problem.has_divider:
        indications.append(Indication("divider", 1.0))

    if problem.same_grid_size and problem.all_filled_rectangles:
        indications.append(Indication("hollow", 0.9))

    if any(
        d.color_changed and not d.position_changed
        for e in problem.examples
        for d in e.object_deltas
    ):
        indications.append(Indication("color_substitution", 0.9))

    if problem.all_single_cells:
        indications.append(Indication("expand", 0.8))

    if all(
        e.output_grid_size != e.input_grid_size
        and e.output_grid_size[0] <= e.input_grid_size[0]
        and e.output_grid_size[1] <= e.input_grid_size[1]
        for e in problem.examples
    ):
        indications.append(Indication("crop", 0.7))

    if problem.same_grid_size and problem.same_object_count:
        indications.append(Indication("symmetry", 0.5))

    # Fallback: always try symmetry, crop, and color substitution
    names = {ind.name for ind in indications}
    for name, conf in [("symmetry", 0.3), ("crop", 0.3), ("color_substitution", 0.3)]:
        if name not in names:
            indications.append(Indication(name, conf))

    return sorted(indications, key=lambda i: i.confidence, reverse=True)
