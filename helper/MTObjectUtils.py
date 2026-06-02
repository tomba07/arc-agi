from collections import deque
from typing import List, Set

from helper.MTObject import MTObject, Cell

Grid = List[List[int]]


def get_neighbors(r: int, c: int, rows: int, cols: int):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def find_objects(grid: Grid, background: int = 0) -> List[MTObject]:
    rows = len(grid)
    cols = len(grid[0])

    visited: Set[Cell] = set()
    objects: List[MTObject] = []

    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited:
                continue

            color = grid[r][c]

            if color == background:
                visited.add((r, c))
                continue

            queue = deque([(r, c)])
            visited.add((r, c))
            cells: Set[Cell] = set()

            while queue:
                cr, cc = queue.popleft()
                cells.add((cr, cc))

                for nr, nc in get_neighbors(cr, cc, rows, cols):
                    if (nr, nc) in visited:
                        continue

                    if grid[nr][nc] == color:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            min_r = min(r for r, _ in cells)
            max_r = max(r for r, _ in cells)
            min_c = min(c for _, c in cells)
            max_c = max(c for _, c in cells)

            obj = MTObject(
                color=color,
                cells=cells,
                bounding_box=(min_r, min_c, max_r, max_c),
                area=len(cells),
                height=max_r - min_r + 1,
                width=max_c - min_c + 1,
                touches_border=(
                    min_r == 0 or min_c == 0 or max_r == rows - 1 or max_c == cols - 1
                ),
            )

            objects.append(obj)

    return objects


def normalized_shape(obj: MTObject) -> Set[Cell]:
    min_r, min_c, _, _ = obj.bounding_box
    return {(r - min_r, c - min_c) for r, c in obj.cells}


def object_to_mask(obj: MTObject) -> Grid:
    min_r, min_c, max_r, max_c = obj.bounding_box
    height = max_r - min_r + 1
    width = max_c - min_c + 1

    mask = [[0 for _ in range(width)] for _ in range(height)]

    for r, c in obj.cells:
        mask[r - min_r][c - min_c] = obj.color

    return mask
