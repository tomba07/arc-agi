from dataclasses import dataclass
from typing import Set, Tuple

import numpy as np


@dataclass
class ExampleObservation:
    same_grid_size: bool
    input_grid_size: Tuple[int, int]
    output_grid_size: Tuple[int, int]
    colors_removed: Set[int]
    colors_added: Set[int]


def observe_example(input_grid: np.ndarray, output_grid: np.ndarray) -> ExampleObservation:
    input_grid_size = input_grid.shape
    output_grid_size = output_grid.shape

    input_colors = set(np.unique(input_grid)) - {0}
    output_colors = set(np.unique(output_grid)) - {0}

    return ExampleObservation(
        same_grid_size=input_grid_size == output_grid_size,
        input_grid_size=input_grid_size,
        output_grid_size=output_grid_size,
        colors_removed=input_colors - output_colors,
        colors_added=output_colors - input_colors,
    )
