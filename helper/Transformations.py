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
