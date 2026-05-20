"""
Sanity check: import env, run random steps, print stats.
Run with:  python test_env.py
"""

import sys
import numpy as np
from env import BlockBlastEnv, N_ACTIONS, BOARD_SIZE, N_PIECES, N_SHAPES
from env.block_blast_env import HEURISTIC_DIM

def test_basic():
    print("=== Basic environment test ===")
    env = BlockBlastEnv(reward_mode="sparse", render_mode="human")
    obs, info = env.reset(seed=0)

    assert obs["board"].shape   == (BOARD_SIZE, BOARD_SIZE), "board shape error"
    assert obs["pieces"].shape  == (N_PIECES, 5, 5),         "pieces shape error"
    assert obs["heuristics"].shape == (HEURISTIC_DIM,),      "heuristics shape error"
    assert obs["heuristics"].dtype == np.float32,            "heuristics dtype error"
    assert (obs["heuristics"] >= 0.0).all() and (obs["heuristics"] <= 1.0).all(), \
        "heuristics out of [0,1]"
    assert info["action_mask"].shape == (N_ACTIONS,),        "mask shape error"
    assert info["action_mask"].any(),                        "no legal actions at start"
    print("  observation spaces: OK")

    env.render()

    total_reward = 0.0
    for step in range(50):
        mask = info["action_mask"]
        legal = np.where(mask)[0]
        if len(legal) == 0:
            break
        action = int(np.random.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Episode ended at step {step+1} | score={info['score']}")
            break
    else:
        print(f"  Ran 50 steps without termination | score={info['score']}")

    print(f"  Total reward collected: {total_reward}")
    print("  Basic test: PASSED\n")


def test_line_clear():
    print("=== Line-clear test ===")
    from env.block_blast_env import encode_action
    from env.shapes import SHAPES

    env = BlockBlastEnv(reward_mode="sparse")
    env.reset(seed=42)

    # Manually fill row 7 with H-bar pieces, then verify clear
    env.board[7, :] = 1.0
    env.board[7, 0] = 0.0   # leave one gap
    # Force piece 0 to be a single dot, place at (7, 0)
    from env.shapes import N_SHAPES
    env.piece_shape_ids[0] = 0   # shape index 0 = single dot

    action = encode_action(0, 7, 0)
    _, reward, terminated, _, info = env.step(action)

    assert info["lines_cleared"] >= 1, "Expected at least 1 line cleared"
    assert np.all(env.board[7, :] == 0.0), "Row 7 should be cleared"
    print(f"  Cleared {info['lines_cleared']} line(s), reward={reward}")
    print("  Line-clear test: PASSED\n")


def test_dense_reward():
    print("=== Dense reward test ===")
    env = BlockBlastEnv(reward_mode="dense")
    obs, info = env.reset(seed=1)
    mask = info["action_mask"]
    legal = np.where(mask)[0]
    action = int(np.random.choice(legal))
    obs, reward, terminated, _, info = env.step(action)
    print(f"  Dense reward for first step: {reward:.4f}")
    print("  Dense reward test: PASSED\n")


def test_action_mask_consistency():
    print("=== Action mask consistency test ===")
    env = BlockBlastEnv()
    obs, info = env.reset(seed=7)
    mask = info["action_mask"]

    from env.block_blast_env import _decode_action
    from env.shapes import SHAPES
    for action_idx in np.where(mask)[0][:10]:   # spot-check first 10 legal actions
        piece_idx, row, col = _decode_action(int(action_idx))
        pid = env.piece_shape_ids[piece_idx]
        assert pid != -1, f"Mask says legal but piece slot {piece_idx} is used"
        shape = SHAPES[pid]
        for dr, dc in shape:
            r, c = row + dr, col + dc
            assert 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE, \
                f"Action {action_idx} places cell out of bounds"
            assert env.board[r, c] == 0.0, \
                f"Action {action_idx} overlaps a filled cell"

    print("  Mask consistency: PASSED\n")


def test_heuristics():
    print("=== Heuristic features test ===")
    env = BlockBlastEnv()
    env.reset(seed=0)

    # Construct a known board:
    #   col 0 filled rows 6,7 → height 2, 0 holes
    #   col 1 filled rows 5,7 → height 3, 1 hole (row 6 below the top of col 1)
    #   col 2 empty             → height 0, 0 holes
    env.board[:] = 0.0
    env.board[6, 0] = 1.0; env.board[7, 0] = 1.0
    env.board[5, 1] = 1.0; env.board[7, 1] = 1.0
    h = env._compute_heuristics()

    heights = (h[0:8] * BOARD_SIZE).round().astype(int)
    holes   = (h[8:16] * (BOARD_SIZE - 1)).round().astype(int)
    row_fill = (h[16:24] * BOARD_SIZE).round().astype(int)
    col_fill = (h[24:32] * BOARD_SIZE).round().astype(int)
    bump = (h[32:39] * BOARD_SIZE).round().astype(int)
    n_legal_frac = float(h[39])

    assert heights[0] == 2 and heights[1] == 3 and heights[2] == 0, f"heights wrong: {heights}"
    assert holes[1] == 1 and holes[0] == 0 and holes[2] == 0,       f"holes wrong: {holes}"
    assert col_fill[0] == 2 and col_fill[1] == 2 and col_fill[2] == 0, f"col_fill wrong: {col_fill}"
    assert row_fill[7] == 2 and row_fill[6] == 1 and row_fill[5] == 1, f"row_fill wrong: {row_fill}"
    # bumpiness: |2-3|=1, |3-0|=3, |0-0|=0...
    assert bump[0] == 1 and bump[1] == 3, f"bumpiness wrong: {bump}"
    assert 0.0 <= n_legal_frac <= 1.0
    assert (h >= 0.0).all() and (h <= 1.0).all(), "heuristics out of [0,1]"
    print(f"  heights[0..3]={heights[:3].tolist()}, holes[0..3]={holes[:3].tolist()}, "
          f"n_legal_frac={n_legal_frac:.3f}")
    print("  Heuristic test: PASSED\n")


if __name__ == "__main__":
    np.random.seed(0)
    test_basic()
    test_line_clear()
    test_dense_reward()
    test_action_mask_consistency()
    test_heuristics()
    print("All tests passed!")
