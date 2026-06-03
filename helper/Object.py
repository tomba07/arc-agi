from dataclasses import dataclass
from typing import Tuple, Set

Cell = Tuple[int, int]


@dataclass
class Object:
    color: int
    cells: Set[Cell]
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    area: int
    height: int
    width: int
    touches_border: bool

    @property
    def normalized_cells(self) -> Set[Cell]:
        min_row, min_col, _, _ = self.bounding_box
        return {(r - min_row, c - min_col) for r, c in self.cells}


@dataclass
class ObjectRelation:
    type: str
    source: Object
    target: Object
