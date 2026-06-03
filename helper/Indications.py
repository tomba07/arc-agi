from typing import Set

from helper.Observations import ProblemObservation


def compute_indications(problem: ProblemObservation) -> Set[str]:
    indications = set()

    if any(
        d.color_changed and not d.position_changed
        for e in problem.examples
        for d in e.object_deltas
    ):
        indications.add("color_substitution")

    if all(
        e.output_grid_size != e.input_grid_size
        and e.output_grid_size[0] <= e.input_grid_size[0]
        and e.output_grid_size[1] <= e.input_grid_size[1]
        for e in problem.examples
    ):
        indications.add("crop")

    if problem.same_grid_size and problem.same_object_count:
        indications.add("symmetry")

    return indications
