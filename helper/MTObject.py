from dataclasses import dataclass
from typing import Tuple, Set

Cell = Tuple[int, int]


@dataclass
class MTObject:
    color: int
    cells: Set[Cell]
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    area: int
    height: int
    width: int
    touches_border: bool
