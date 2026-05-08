import numpy as np
from env.block_blast_env import _decode_action, BOARD_SIZE
from env.shapes import SHAPES


class GreedyAgent:
    """
    At each step, try every legal action, simulate one step, and pick the
    action that clears the most lines. Ties broken randomly.
    This is a strong one-step-lookahead baseline.
    """

    def __init__(self, env):
        self.env = env

    def select_action(self, obs, action_mask: np.ndarray) -> int:
        board = obs["board"].copy()         # (8, 8)
        piece_ids = self.env.piece_shape_ids[:]

        best_lines = -1
        best_actions = []

        legal = np.where(action_mask)[0]
        for action in legal:
            piece_idx, row, col = _decode_action(int(action))
            pid = piece_ids[piece_idx]
            if pid == -1:
                continue
            shape = SHAPES[pid]
            lines = _simulate_lines(board, shape, row, col)
            if lines > best_lines:
                best_lines = lines
                best_actions = [int(action)]
            elif lines == best_lines:
                best_actions.append(int(action))

        return int(np.random.choice(best_actions))


def _simulate_lines(board: np.ndarray, shape, row: int, col: int) -> int:
    """Count lines that would clear if shape is placed at (row, col), without modifying board."""
    tmp = board.copy()
    for dr, dc in shape:
        tmp[row + dr, col + dc] = 1.0
    rows_done = sum(1 for r in range(BOARD_SIZE) if np.all(tmp[r, :] == 1.0))
    cols_done = sum(1 for c in range(BOARD_SIZE) if np.all(tmp[:, c] == 1.0))
    return rows_done + cols_done


def evaluate(env, n_episodes: int = 100, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    agent = GreedyAgent(env)
    scores, lengths = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(1 << 31)))
        mask = info["action_mask"]
        steps = 0

        while True:
            action = agent.select_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            steps += 1
            if terminated or truncated:
                break

        scores.append(info["score"])
        lengths.append(steps)

    return {
        "mean_score":  float(np.mean(scores)),
        "std_score":   float(np.std(scores)),
        "mean_steps":  float(np.mean(lengths)),
        "std_steps":   float(np.std(lengths)),
    }
