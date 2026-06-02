from collections import deque
from typing import List, Set

from helper.MTObject import MTObject, Cell

Grid = List[List[int]]


def find_objects(grid: Grid, background: int = 0) -> List[MTObject]:
    rows = len(grid)
    cols = len(grid[0])

    visited: Set[Cell] = set()
    objects: List[MTObject] = []

    for row in range(rows):
        for col in range(cols):
            if (row, col) in visited:
                continue

            color = grid[row][col]

            if color == background:
                visited.add((row, col))
                continue

            cells = _collect_connected_cells(grid, row, col, color, rows, cols, visited)
            objects.append(_build_object(cells, color, rows, cols))

    return objects


def _collect_connected_cells(grid: Grid, start_row: int, start_col: int, color: int, rows: int, cols: int, visited: Set[Cell]) -> Set[Cell]:
    queue = deque([(start_row, start_col)])
    visited.add((start_row, start_col))
    cells: Set[Cell] = set()

    while queue:
        row, col = queue.popleft()
        cells.add((row, col))

        for new_row, new_col in _get_neighbors(row, col, rows, cols):
            if (new_row, new_col) not in visited and grid[new_row][new_col] == color:
                visited.add((new_row, new_col))
                queue.append((new_row, new_col))

    return cells


def _build_object(cells: Set[Cell], color: int, rows: int, cols: int) -> MTObject:
    min_row = min(r for r, _ in cells)
    max_row = max(r for r, _ in cells)
    min_col = min(c for _, c in cells)
    max_col = max(c for _, c in cells)

    return MTObject(
        color=color,
        cells=cells,
        bounding_box=(min_row, min_col, max_row, max_col),
        area=len(cells),
        height=max_row - min_row + 1,
        width=max_col - min_col + 1,
        touches_border=(min_row == 0 or min_col == 0 or max_row == rows - 1 or max_col == cols - 1),
    )


def _get_neighbors(row: int, col: int, rows: int, cols: int):
    for row_diff, col_diff in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + row_diff, col + col_diff
        if 0 <= new_row < rows and 0 <= new_col < cols:
            yield new_row, new_col
