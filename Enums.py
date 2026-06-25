from enum import Enum


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class DiagonalDirection(str, Enum):
    TL = "tl"
    TR = "tr"
    BL = "bl"
    BR = "br"


class AxisDirection(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class LogicalOperation(str, Enum):
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"
    XNOR = "XNOR"
