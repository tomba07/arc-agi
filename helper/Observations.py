from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from helper.Object import Object, ObjectRelation, ObjectDelta
from helper.ObjectUtils import find_objects, compute_relations, compute_object_deltas, find_divider


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
    object_deltas: List[ObjectDelta]
    all_input_objects_are_single_cells: bool
    all_input_objects_are_filled_rectangles: bool
    has_divider: bool
    divider_axis: Optional[str]
    divider_index: Optional[int]


@dataclass
class ProblemObservation:
    examples: List[ExampleObservation]
    same_grid_size: bool
    colors_removed: bool
    colors_added: bool
    same_object_count: bool
    all_single_cells: bool
    all_filled_rectangles: bool
    has_divider: bool


def observe_problem(examples: List[ExampleObservation]) -> ProblemObservation:
    return ProblemObservation(
        examples=examples,
        same_grid_size=all(e.same_grid_size for e in examples),
        colors_removed=len({frozenset(e.colors_removed) for e in examples}) == 1,
        colors_added=len({frozenset(e.colors_added) for e in examples}) == 1,
        same_object_count=all(e.same_object_count for e in examples),
        all_single_cells=all(e.all_input_objects_are_single_cells for e in examples),
        all_filled_rectangles=all(e.all_input_objects_are_filled_rectangles for e in examples),
        has_divider=all(e.has_divider for e in examples),
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

    try:
        divider_axis, divider_index = find_divider(input_grid)
        has_divider = True
    except ValueError:
        divider_axis, divider_index = None, None
        has_divider = False

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
        object_deltas=compute_object_deltas(input_objects, output_objects),
        all_input_objects_are_single_cells=bool(
            input_objects and all(obj.area == 1 for obj in input_objects)
        ),
        all_input_objects_are_filled_rectangles=bool(
            input_objects and all(
                obj.area == obj.height * obj.width for obj in input_objects
            )
        ),
        has_divider=has_divider,
        divider_axis=divider_axis,
        divider_index=divider_index,
    )
