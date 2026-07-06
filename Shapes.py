from dataclasses import dataclass, field
import numpy as np

from Enums import Direction

Grid = np.ndarray


@dataclass
class CellsInfo:
    cells: set[tuple[int, int]]
    color: int


@dataclass
class Shape:
    row: int
    col: int
    width: int
    height: int
    cells: set[tuple[int, int]]
    is_square_abstraction: bool = False
    color: int | None = None
    is_one_by_one: bool = False
    is_two_by_two: bool = False
    encloses_cells: bool = False
    enclosed_cells: set[tuple[int, int]] | None = None
    enclosed_shapes: list["Shape"] = field(default_factory=list)
    is_wall: bool = False

    # method which returns True
    def is_similar_to(self, other: "Shape") -> bool:
        return (
            self.width == other.width
            and self.height == other.height
            and self.color == other.color
        )


@dataclass
class Spaceship_Shape(Shape):
    is_spaceship_shape: bool = True
    beam_color: int | None = None
    direction: Direction | None = None
    tip_row: int | None = None
    tip_col: int | None = None


def _collect_cells(
    grid: Grid, start_row: int, start_col: int, visited: set
) -> CellsInfo:
    cells: set[tuple[int, int]] = set()
    queue = [(start_row, start_col)]
    color = grid[start_row, start_col]
    while queue:
        row, col = queue.pop(0)
        if (row, col) in visited or not (
            0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        ):
            continue
        if grid[row, col] == 0:
            continue
        visited.add((row, col))
        cells.add((row, col))
        queue.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])
    return CellsInfo(cells=cells, color=color)


def _make_shape(cells: set[tuple[int, int]], color: int) -> Shape:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    return Shape(
        row=min_row,
        col=min_col,
        width=width,
        height=height,
        cells=cells,
        color=color,
        is_two_by_two=(height == 2 and width == 2),
        is_one_by_one=(height == 1 and width == 1),
    )


def get_shapes(grid: Grid) -> list[Shape]:
    visited: set[tuple[int, int]] = set()
    shapes = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] == 0 or (start_row, start_col) in visited:
                continue
            cells_info = _collect_cells(grid, start_row, start_col, visited)
            shapes.append(_make_shape(cells_info.cells, cells_info.color))
    return shapes


def get_diagonal_shapes(grid: Grid) -> list[Shape]:
    visited: set[tuple[int, int]] = set()
    shapes = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] == 0 or (start_row, start_col) in visited:
                continue
            color = grid[start_row, start_col]
            cells: set[tuple[int, int]] = set()
            queue = [(start_row, start_col)]
            while queue:
                row, col = queue.pop()
                if (row, col) in visited or not (
                    0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
                ):
                    continue
                if grid[row, col] == 0:
                    continue
                visited.add((row, col))
                cells.add((row, col))
                queue.extend(
                    [
                        (row - 1, col),
                        (row + 1, col),
                        (row, col - 1),
                        (row, col + 1),
                        (row - 1, col - 1),
                        (row - 1, col + 1),
                        (row + 1, col - 1),
                        (row + 1, col + 1),
                    ]
                )
            shapes.append(_make_shape(cells, color))
    return shapes


def get_color_strict_shapes(grid: Grid) -> list[Shape]:
    visited: set[tuple[int, int]] = set()
    shapes = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if (start_row, start_col) in visited:
                continue
            color = int(grid[start_row, start_col])
            cells: set[tuple[int, int]] = set()
            queue = [(start_row, start_col)]
            while queue:
                r, c = queue.pop()
                if (r, c) in visited or not (
                    0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                ):
                    continue
                if grid[r, c] != color:
                    continue
                visited.add((r, c))
                cells.add((r, c))
                queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
            shapes.append(_make_shape(cells, color))
    return shapes


def get_square_abstraction(shapes: list[Shape]) -> Shape | None:
    shapes_by_color: dict[int, list[Shape]] = {}
    for shape in shapes:
        if shape.color not in shapes_by_color:
            shapes_by_color[shape.color] = []
        shapes_by_color[shape.color].append(shape)

    for color, color_shapes in shapes_by_color.items():
        if len(color_shapes) == 4 and all(
            s.width == 1 and s.height == 1 for s in color_shapes
        ):
            rows = sorted(s.row for s in color_shapes)
            cols = sorted(s.col for s in color_shapes)
            if (
                rows[0] == rows[1]
                and rows[2] == rows[3]
                and cols[0] == cols[1]
                and cols[2] == cols[3]
            ):
                return Shape(
                    row=rows[0],
                    col=cols[0],
                    width=cols[2] - cols[0] + 1,
                    height=rows[2] - rows[0] + 1,
                    cells=set((row, col) for row in rows for col in cols),
                    is_square_abstraction=True,
                    color=color,
                )
    return None


def check_spaceship_shape(shapes: list[Shape], grid: Grid) -> Spaceship_Shape | None:
    if len(shapes) != 1:
        return None
    shape = shapes[0]
    ratio_correct = (
        shape.width == shape.height * 2 - 1 or shape.height == shape.width * 2 - 1
    )
    if not ratio_correct:
        return None
    for direction in Direction:
        beam_color = _check_pyramid_shape_and_beam_color(shape, grid, direction)
        if beam_color is not None:
            tip_row = (
                shape.row
                if direction == Direction.UP
                else shape.row + shape.height - 1
                if direction == Direction.DOWN
                else shape.row + shape.height // 2
            )
            tip_col = (
                shape.col + shape.width // 2
                if direction in (Direction.UP, Direction.DOWN)
                else shape.col
                if direction == Direction.LEFT
                else shape.col + shape.width - 1
            )
            return Spaceship_Shape(
                row=shape.row,
                col=shape.col,
                width=shape.width,
                height=shape.height,
                cells=shape.cells,
                color=shape.color,
                direction=direction,
                beam_color=beam_color,
                tip_row=tip_row,
                tip_col=tip_col,
            )
    return None


def _check_pyramid_shape_and_beam_color(
    shape: Shape, grid: Grid, direction: Direction
) -> int | None:
    center_row = shape.row + shape.height // 2
    center_col = shape.col + shape.width // 2
    if direction in (Direction.UP, Direction.DOWN):
        beam_row = (
            shape.row + shape.height - 1 if direction == Direction.UP else shape.row
        )
        beam_col = center_col
        cells = [
            (
                shape.row + i
                if direction == Direction.UP
                else shape.row + shape.height - 1 - i,
                center_col + dc,
            )
            for i in range(shape.height)
            for dc in range(-i, i + 1)
        ]
    else:
        beam_row = center_row
        beam_col = (
            shape.col + shape.width - 1 if direction == Direction.LEFT else shape.col
        )
        cells = [
            (
                center_row + dr,
                shape.col + i
                if direction == Direction.LEFT
                else shape.col + shape.width - 1 - i,
            )
            for i in range(shape.width)
            for dr in range(-i, i + 1)
        ]
    for row, col in cells:
        if (row, col) == (beam_row, beam_col):
            continue
        if grid[row, col] != shape.color:
            return None
    beam_color = grid[beam_row, beam_col]
    return None if beam_color == shape.color else beam_color


def collect_opposing_same_color_single_cells(
    shapes: list[Shape],
) -> list[tuple[Shape, Shape]]:
    single_cell_shapes = [s for s in shapes if s.width == 1 and s.height == 1]
    opposing_pairs = []
    for i, shape1 in enumerate(single_cell_shapes):
        for j, shape2 in enumerate(single_cell_shapes):
            if i >= j:
                continue
            if shape1.color == shape2.color:
                left_right = (
                    shape1.col == 0 and shape2.col == max(shape1.col, shape2.col)
                ) or (shape1.col == max(shape1.col, shape2.col) and shape2.col == 0)
                top_bottom = (
                    shape1.row == 0 and shape2.row == max(shape1.row, shape2.row)
                ) or (shape1.row == max(shape1.row, shape2.row) and shape2.row == 0)
                if left_right or top_bottom:
                    opposing_pairs.append((shape1, shape2))
    return opposing_pairs


def _collect_zero_cells(
    grid: Grid, start_row: int, start_col: int, visited: set
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    queue = [(start_row, start_col)]
    while queue:
        row, col = queue.pop(0)
        if (row, col) in visited or not (
            0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        ):
            continue
        if grid[row, col] != 0:
            continue
        visited.add((row, col))
        cells.add((row, col))
        queue.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])
    return cells


def get_zero_shapes(grid: Grid) -> list[Shape]:
    visited: set[tuple[int, int]] = set()
    zero_shapes = []
    for start_row in range(grid.shape[0]):
        for start_col in range(grid.shape[1]):
            if grid[start_row, start_col] != 0 or (start_row, start_col) in visited:
                continue
            cells = _collect_zero_cells(grid, start_row, start_col, visited)
            zero_shapes.append(_make_shape(cells, 0))
    return zero_shapes


def get_enclosed_shapes(
    shapes: list[Shape], grid_shape: tuple[int, int]
) -> list[Shape]:
    max_row, max_col = grid_shape[0] - 1, grid_shape[1] - 1
    return [
        s
        for s in shapes
        if all(0 < row < max_row and 0 < col < max_col for row, col in s.cells)
    ]


def get_non_enclosed_shapes(
    shapes: list[Shape], grid_shape: tuple[int, int]
) -> list[Shape]:
    max_row, max_col = grid_shape[0] - 1, grid_shape[1] - 1
    return [
        s
        for s in shapes
        if any(
            row == 0 or col == 0 or row == max_row or col == max_col
            for row, col in s.cells
        )
    ]


def shape_encloses_cells(shape: Shape, grid: Grid) -> set[tuple[int, int]]:
    rows, cols = grid.shape
    shape_cell_set = shape.cells
    visited = set()
    queue = []
    for r in range(rows):
        for c in [0, cols - 1]:
            if (r, c) not in shape_cell_set:
                queue.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if (r, c) not in shape_cell_set:
                queue.append((r, c))
    while queue:
        r, c = queue.pop()
        if (r, c) in visited:
            continue
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        if (r, c) in shape_cell_set:
            continue
        visited.add((r, c))
        queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    cells_without_shape = {
        (r, c) for r in range(rows) for c in range(cols)
    } - shape_cell_set

    return cells_without_shape - visited
