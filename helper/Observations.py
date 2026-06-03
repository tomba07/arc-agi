from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np

from helper.Object import Object, ObjectRelation
from helper.ObjectUtils import find_objects, compute_relations


@dataclass
class ExampleObservation:
    same_grid_size: bool
    input_grid_size: Tuple[int, int]
    output_grid_size: Tuple[int, int]
    colors_removed: Set[int]
    colors_added: Set[int]
    input_objects: List[Object]
    output_objects: List[Object]
    same_object_count: bool
    input_relations: List[ObjectRelation]
    output_relations: List[ObjectRelation]


@dataclass
class ProblemObservation:
    examples: List[ExampleObservation]
    same_grid_size: bool
    colors_removed: bool
    colors_added: bool
    same_object_count: bool


def observe_problem(examples: List[ExampleObservation]) -> ProblemObservation:
    return ProblemObservation(
        examples=examples,
        same_grid_size=all(e.same_grid_size for e in examples),
        colors_removed=len({frozenset(e.colors_removed) for e in examples}) == 1,
        colors_added=len({frozenset(e.colors_added) for e in examples}) == 1,
        same_object_count=all(e.same_object_count for e in examples),
    )


def observe_example(
    input_grid: np.ndarray, output_grid: np.ndarray
) -> ExampleObservation:
    input_grid_size = input_grid.shape
    output_grid_size = output_grid.shape

    input_colors = set(np.unique(input_grid)) - {0}
    output_colors = set(np.unique(output_grid)) - {0}

    input_objects = find_objects(input_grid.tolist())
    output_objects = find_objects(output_grid.tolist())

    return ExampleObservation(
        same_grid_size=input_grid_size == output_grid_size,
        input_grid_size=input_grid_size,
        output_grid_size=output_grid_size,
        colors_removed=input_colors - output_colors,
        colors_added=output_colors - input_colors,
        input_objects=input_objects,
        output_objects=output_objects,
        same_object_count=len(input_objects) == len(output_objects),
        input_relations=compute_relations(input_objects),
        output_relations=compute_relations(output_objects),
    )
