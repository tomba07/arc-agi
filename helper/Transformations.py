import numpy as np

# Transformations are pure Grid → Grid functions.
# Framings (parameterized decompositions using observation data) are expressed
# as closures in theory factories, not as standalone functions here.

Grid = np.ndarray

# ---------------------------------------------------------------------------
# Nuclear primitives
# ---------------------------------------------------------------------------

def erode(grid: Grid) -> Grid:
    """Keep only cells whose 4 direct neighbors are all non-zero (interior cells)."""
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                if (r > 0 and r < rows - 1 and c > 0 and c < cols - 1
                        and grid[r - 1, c] != 0 and grid[r + 1, c] != 0
                        and grid[r, c - 1] != 0 and grid[r, c + 1] != 0):
                    result[r, c] = grid[r, c]
    return result


def dilate_square(grid: Grid, radius: int) -> Grid:
    """Expand each non-zero cell to a (2*radius+1)² square block."""
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                result[max(0, r - radius):min(rows, r + radius + 1),
                       max(0, c - radius):min(cols, c + radius + 1)] = grid[r, c]
    return result


def mask_subtract(grid: Grid, mask: Grid) -> Grid:
    """Zero out cells in grid wherever mask is non-zero."""
    return np.where(mask != 0, 0, grid)


def recolor_nonzero(grid: Grid, color: int) -> Grid:
    """Replace every non-zero cell with color."""
    return np.where(grid != 0, color, 0).astype(grid.dtype)


# ---------------------------------------------------------------------------
# Composed transformations
# ---------------------------------------------------------------------------

def make_hollow(grid: Grid) -> Grid:
    return mask_subtract(grid, erode(grid))


def grow_cells(grid: Grid, scale: int, new_color: int) -> Grid:
    return recolor_nonzero(dilate_square(grid, scale // 2), new_color)


# ---------------------------------------------------------------------------
# Logical combination
# ---------------------------------------------------------------------------

def swap_two_nonzero(grid: Grid) -> Grid:
    """Swap the two non-zero colors in grid; return unchanged if not exactly two."""
    colors = sorted(set(np.unique(grid)) - {0})
    if len(colors) != 2:
        return grid
    a, b = colors[0], colors[1]
    return np.select([grid == a, grid == b], [b, a], grid).astype(grid.dtype)


def remap_replace_keep(grid: Grid, replace_color: int) -> Grid:
    """Map replace_color → other_color and other_color → 0 (other determined per grid)."""
    others = sorted(set(np.unique(grid)) - {0, replace_color})
    if len(others) != 1:
        return grid
    other = others[0]
    return np.select(
        [grid == replace_color, grid == other], [other, 0], grid
    ).astype(grid.dtype)


def ray_fill(grid: Grid, directions: list) -> Grid:
    """Cast a ray from each non-zero cell in each direction until the grid boundary."""
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                color = grid[r, c]
                for dr, dc in directions:
                    nr, nc = r, c
                    while 0 <= nr < rows and 0 <= nc < cols:
                        result[nr, nc] = color
                        nr += dr
                        nc += dc
    return result


DIRECTIONS_DIAGONAL   = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRECTIONS_ORTHOGONAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS_ALL_8      = DIRECTIONS_DIAGONAL + DIRECTIONS_ORTHOGONAL


def snap_matching_to_walls(grid: Grid, wall_map: dict) -> Grid:
    """Move each interior cell to the inner edge of the wall whose color it matches.

    wall_map: {color: ('row'|'col', inner_position)} — derived externally from observations.
    """
    rows, cols = grid.shape
    interior = [
        (r, c, int(grid[r, c]))
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
        if grid[r, c] != 0
    ]
    result = grid.copy()
    result[1:rows - 1, 1:cols - 1] = 0
    for r, c, color in interior:
        if color in wall_map:
            kind, pos = wall_map[color]
            if kind == "row":
                result[pos, c] = color
            else:
                result[r, pos] = color
    return result


def draw_clockwise_spiral(grid: Grid) -> Grid:
    """Fill grid with a clockwise rectangular spiral of color 3."""
    result = np.zeros_like(grid)
    rows, cols = grid.shape
    n = rows

    arm_lengths = [n]
    length = n - 1
    while length > 0:
        arm_lengths.append(length)
        arm_lengths.append(length)
        length -= 2

    r, c = 0, 0
    dr, dc = 0, 1  # start going right

    for arm_len in arm_lengths:
        for step in range(arm_len):
            if 0 <= r < rows and 0 <= c < cols:
                result[r, c] = 3
            if step < arm_len - 1:
                r += dr
                c += dc
        dr, dc = dc, -dr  # clockwise turn
        r += dr
        c += dc

    return result


def boolean_combine(left: Grid, right: Grid, op: str, output_color: int) -> Grid:
    l_bool = left != 0
    r_bool = right != 0
    if op == "and":
        mask = l_bool & r_bool
    elif op == "or":
        mask = l_bool | r_bool
    elif op == "nor":
        mask = ~(l_bool | r_bool)
    else:
        raise ValueError(f"Unknown op: {op}")
    return np.where(mask, output_color, 0).astype(left.dtype)
