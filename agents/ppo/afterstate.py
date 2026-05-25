"""Afterstate feature extraction for the afterstate-evaluating PPO actor.

Background: Chen et al. 2026 (arXiv:2603.26765) show that for deterministic-
placement games like Tetris, evaluating *afterstates* (board post-action but
before any randomness) with a tiny shared linear layer beats a Q(s,a)-style
actor by a wide margin — fewer parameters, lower gradient variance. Block
Blast has the same property: placing a piece is deterministic; randomness only
enters when all three slots are used and the env re-deals.

This module simulates each legal action on a passed-in board and extracts 9
DT-style features per resulting afterstate. The simulator mirrors
env.block_blast_env._can_place / _clear_lines but never touches the env — it
operates on a copied numpy board. Per CLAUDE.md the env directory is owned by
Member A; if the placement rules change there we have to mirror them here too.

Features are kept at their *raw integer scale* (no [0,1] normalization).
Following the paper, the linear scorer is expected to learn per-feature
weights; pre-normalizing to [0,1] crushes initial logit differences below
the entropy bonus, leaving the policy near-uniform and unable to bootstrap.

Features:
    0  lines_cleared           # rows + cols cleared by this placement  (0-16)
    1  eroded_piece_cells      piece cells participating in cleared lines (0-5)
    2  holes_after             empty cells below filled cells           (0-64)
    3  max_height_after        max column height                        (0-8)
    4  mean_height_after       mean column height                       (0-8)
    5  bumpiness_after         sum |adjacent column height diff|        (0-56)
    6  row_transitions_after   filled↔empty boundaries within rows      (0-72)
    7  col_transitions_after   same vertically                          (0-72)
    8  near_full_count_after   rows + cols at ≥ 7/8 filled              (0-16)
"""
from __future__ import annotations

import numpy as np

from env.shapes import SHAPES
from env.block_blast_env import _decode_action

BOARD_SIZE = 8
N_PIECES = 3
N_ACTIONS = N_PIECES * BOARD_SIZE * BOARD_SIZE      # 192
AFTERSTATE_FEATURE_DIM = 9


# ---------------------------------------------------------------------------
# Single-action simulator
# ---------------------------------------------------------------------------

def _simulate(board: np.ndarray, shape, row: int, col: int):
    """Place `shape` at (row, col) on a copy of `board`, clear full rows/cols.

    Returns (afterstate_board, lines_cleared, eroded_piece_cells).
    Caller must ensure the placement is legal (we don't re-check bounds).
    """
    tmp = board.copy()
    piece_cells = [(row + dr, col + dc) for dr, dc in shape]
    for r, c in piece_cells:
        tmp[r, c] = 1.0

    rows_done = [r for r in range(BOARD_SIZE) if np.all(tmp[r, :] == 1.0)]
    cols_done = [c for c in range(BOARD_SIZE) if np.all(tmp[:, c] == 1.0)]
    lines = len(rows_done) + len(cols_done)

    cleared_rows = set(rows_done)
    cleared_cols = set(cols_done)
    eroded = sum(1 for r, c in piece_cells if r in cleared_rows or c in cleared_cols)

    for r in rows_done:
        tmp[r, :] = 0.0
    for c in cols_done:
        tmp[:, c] = 0.0
    return tmp, lines, eroded


# ---------------------------------------------------------------------------
# Board statistics (mirror env helpers, but operate on a passed-in array)
# ---------------------------------------------------------------------------

def _column_heights(board: np.ndarray) -> np.ndarray:
    heights = np.zeros(BOARD_SIZE, dtype=np.int32)
    for col in range(BOARD_SIZE):
        filled = np.where(board[:, col] == 1.0)[0]
        heights[col] = BOARD_SIZE - int(filled[0]) if len(filled) > 0 else 0
    return heights


def _count_holes(board: np.ndarray) -> int:
    holes = 0
    for col in range(BOARD_SIZE):
        found = False
        for row in range(BOARD_SIZE):
            if board[row, col] == 1.0:
                found = True
            elif found:
                holes += 1
    return holes


def _row_transitions(board: np.ndarray) -> int:
    """Count filled↔empty boundaries per row, treating the board edges as filled
    (matches Tetris DT row_transitions; flags fragmentation along the row)."""
    n = 0
    for r in range(BOARD_SIZE):
        prev = 1
        for c in range(BOARD_SIZE):
            cur = int(board[r, c])
            if cur != prev:
                n += 1
            prev = cur
        if prev != 1:
            n += 1
    return n


def _col_transitions(board: np.ndarray) -> int:
    n = 0
    for c in range(BOARD_SIZE):
        prev = 1
        for r in range(BOARD_SIZE):
            cur = int(board[r, c])
            if cur != prev:
                n += 1
            prev = cur
        if prev != 1:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Feature vector for a single afterstate
# ---------------------------------------------------------------------------

def _features_of_afterstate(board_after: np.ndarray, lines_cleared: int,
                             eroded: int) -> np.ndarray:
    h = _column_heights(board_after)
    holes = _count_holes(board_after)
    bump = int(np.abs(np.diff(h)).sum())
    row_t = _row_transitions(board_after)
    col_t = _col_transitions(board_after)
    row_fill = board_after.sum(axis=1)
    col_fill = board_after.sum(axis=0)
    near_full = int((row_fill >= BOARD_SIZE - 1).sum()) + int((col_fill >= BOARD_SIZE - 1).sum())

    return np.array([
        float(lines_cleared),
        float(eroded),
        float(holes),
        float(h.max()),
        float(h.mean()),
        float(bump),
        float(row_t),
        float(col_t),
        float(near_full),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API — one call per env step
# ---------------------------------------------------------------------------

def compute_afterstate_features(
    board: np.ndarray,
    piece_shape_ids,
    action_mask: np.ndarray,
) -> np.ndarray:
    """Return (192, 9) afterstate features for each action; zero for illegal ones.

    The actor will mask logits to -inf for illegal actions anyway, so the
    illegal-row contents are irrelevant — keeping them zero just avoids work.

    Args:
        board:           (8, 8) float32 current board
        piece_shape_ids: length-3 list, -1 means slot used
        action_mask:     (192,) bool, env-provided legality mask
    """
    feats = np.zeros((N_ACTIONS, AFTERSTATE_FEATURE_DIM), dtype=np.float32)
    legal_idx = np.where(action_mask)[0]
    for a in legal_idx:
        piece_idx, row, col = _decode_action(int(a))
        pid = piece_shape_ids[piece_idx]
        if pid == -1:
            continue
        shape = SHAPES[pid]
        board_after, lines, eroded = _simulate(board, shape, row, col)
        feats[a] = _features_of_afterstate(board_after, lines, eroded)
    return feats


def compute_afterstate_features_batch(
    boards: np.ndarray,
    piece_shape_ids_batch,
    action_masks: np.ndarray,
) -> np.ndarray:
    """Batched over n envs.

    Args:
        boards:                (N, 8, 8) float32
        piece_shape_ids_batch: length-N list of length-3 lists
        action_masks:          (N, 192) bool
    Returns:
        (N, 192, 9) float32
    """
    n = boards.shape[0]
    out = np.zeros((n, N_ACTIONS, AFTERSTATE_FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        out[i] = compute_afterstate_features(boards[i], piece_shape_ids_batch[i], action_masks[i])
    return out
