from typing import Optional
from dataclasses import dataclass


@dataclass
class Observations:
    grid_size_stays_identical: bool
    grid_size_decreases: bool


def observe(examples: list) -> Observations:
    grid_size_stays_identical = all(inp.shape == out.shape for inp, out in examples)
    grid_size_decreases = any(inp.size > out.size for inp, out in examples)

    return Observations(
        grid_size_stays_identical=grid_size_stays_identical,
        grid_size_decreases=grid_size_decreases,
    )
