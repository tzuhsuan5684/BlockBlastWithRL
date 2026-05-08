import numpy as np

# Each shape is a list of (row, col) offsets from the top-left anchor.
# All offsets are non-negative so placement at (r, c) occupies (r+dr, c+dc).
SHAPES = [
    # --- 1-cell ---
    [(0, 0)],                                                           # 0: dot

    # --- horizontal bars ---
    [(0, 0), (0, 1)],                                                   # 1: H2
    [(0, 0), (0, 1), (0, 2)],                                           # 2: H3
    [(0, 0), (0, 1), (0, 2), (0, 3)],                                   # 3: H4
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],                          # 4: H5

    # --- vertical bars ---
    [(0, 0), (1, 0)],                                                   # 5: V2
    [(0, 0), (1, 0), (2, 0)],                                           # 6: V3
    [(0, 0), (1, 0), (2, 0), (3, 0)],                                   # 7: V4
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],                          # 8: V5

    # --- squares ---
    [(0, 0), (0, 1), (1, 0), (1, 1)],                                   # 9:  2×2
    [(r, c) for r in range(3) for c in range(3)],                       # 10: 3×3

    # ✅--- L-shapes (4 rotations) ---
    # █        █ █ █    █ █          █
    # █        █          █      █ █ █
    # █ █                 █
    [(0, 0), (1, 0), (2, 0), (2, 1)],                                   # 11: L-0
    [(0, 0), (0, 1), (0, 2), (1, 0)],                                   # 12: L-90
    [(0, 0), (0, 1), (1, 1), (2, 1)],                                   # 13: L-180
    [(0, 2), (1, 0), (1, 1), (1, 2)],                                   # 14: L-270

    # ✅--- J-shapes (mirror of L, 4 rotations) ---
    #   █      █          █ █    █ █ █
    #   █      █ █ █      █          █
    # █ █                 █
    [(0, 1), (1, 1), (2, 0), (2, 1)],                                   # 15: J-0
    [(0, 0), (1, 0), (1, 1), (1, 2)],                                   # 16: J-90
    [(0, 0), (0, 1), (1, 0), (2, 0)],                                   # 17: J-180
    [(0, 0), (0, 1), (0, 2), (1, 2)],                                   # 18: J-270

    # ✅--- Big-L (5-cell, 4 rotations) ---
    # █        █ █ █    █ █ █        █
    # █        █            █        █
    # █ █ █    █            █    █ █ █
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],                          # 19: BigL-0
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],                          # 20: BigL-90
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],                          # 21: BigL-180
    [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],                          # 22: BigL-270

    # --- T-shapes (4 rotations) ---
    # █ █ █    █        █         █
    #   █      █ █    █ █ █     █ █
    #          █                  █
    [(0, 0), (0, 1), (0, 2), (1, 1)],                                   # 23: T-south
    [(0, 0), (1, 0), (1, 1), (2, 0)],                                   # 24: T-east
    [(0, 1), (1, 0), (1, 1), (1, 2)],                                   # 25: T-north
    [(0, 1), (1, 0), (1, 1), (2, 1)],                                   # 26: T-west

    # --- S-shape (2 rotations) ---
    #   █ █    █
    # █ █      █ █
    #            █
    [(0, 1), (0, 2), (1, 0), (1, 1)],                                   # 27: S-flat
    [(0, 0), (1, 0), (1, 1), (2, 1)],                                   # 28: S-upright

    # --- Z-shape (2 rotations) ---
    # █ █        █
    #   █ █    █ █
    #          █
    [(0, 0), (0, 1), (1, 1), (1, 2)],                                   # 29: Z-flat
    [(0, 1), (1, 0), (1, 1), (2, 0)],                                   # 30: Z-upright

    # --- small corners (3-cell, 4 rotations) ---
    # █        █ █    █ █      █
    # █ █      █        █    █ █
    [(0, 0), (1, 0), (1, 1)],                                           # 31: corner-SE
    [(0, 0), (0, 1), (1, 0)],                                           # 32: corner-SW
    [(0, 0), (0, 1), (1, 1)],                                           # 33: corner-NW
    [(0, 1), (1, 0), (1, 1)],                                           # 34: corner-NE
]

SHAPE_NAMES = [
    "Dot",
    "H2", "H3", "H4", "H5",
    "V2", "V3", "V4", "V5",
    "Sq2x2", "Sq3x3",
    "L-0", "L-90", "L-180", "L-270",
    "J-0", "J-90", "J-180", "J-270",
    "BigL-0", "BigL-90", "BigL-180", "BigL-270",
    "T-S", "T-E", "T-N", "T-W",
    "S-flat", "S-up",
    "Z-flat", "Z-up",
    "C-SE", "C-SW", "C-NW", "C-NE",
]

N_SHAPES = len(SHAPES)


def shape_to_grid(shape, size=5):
    """Convert shape offset list to a (size × size) binary NumPy grid."""
    grid = np.zeros((size, size), dtype=np.float32)
    for r, c in shape:
        grid[r, c] = 1.0
    return grid


def shape_dims(shape):
    """Return (height, width) of a shape."""
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def print_shape(shape, name=""):
    """Debug helper: print shape as ASCII art."""
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h, w = max(rows) + 1, max(cols) + 1
    cells = set(shape)
    print(f"{name}:")
    for r in range(h):
        print("  " + " ".join("█" if (r, c) in cells else "·" for c in range(w)))
