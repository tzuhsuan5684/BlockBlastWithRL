import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .shapes import SHAPES, shape_to_grid, N_SHAPES
from reward_functions import HOLE_PENALTY, BUMPINESS_PENALTY, COMBO_STREAK_BONUS

BOARD_SIZE = 8
N_PIECES = 3          # pieces per round
N_ACTIONS = N_PIECES * BOARD_SIZE * BOARD_SIZE   # 3 × 64 = 192


class BlockBlastEnv(gym.Env):
    """
    Block Blast on an 8×8 grid, Gymnasium-compatible.

    Observation (Dict):
        board       : float32 (8, 8)    – 0 empty, 1 filled
        pieces      : float32 (3, 5, 5) – each piece rendered in a 5×5 grid;
                                          all-zeros when that slot is already used
        pieces_left : float32 (3,)      – binary mask; 1 if slot still has a piece

    Action (Discrete 192):
        action = piece_idx * 64 + row * 8 + col
        piece_idx ∈ {0,1,2}, row/col ∈ {0,…,7}

    Reward (sparse, default):
        +n_lines  for each row or column cleared this step
        –10       on terminal (no legal action left)

    Reward (dense, reward_mode='dense'):
        sparse reward  –0.1 × holes  –0.05 × bumpiness

    Episode ends when no remaining piece can be placed anywhere.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, reward_mode: str = "sparse", render_mode: str = None):
        super().__init__()
        assert reward_mode in ("sparse", "dense"), "reward_mode must be 'sparse' or 'dense'"
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "board":       spaces.Box(0.0, 1.0, (BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "pieces":      spaces.Box(0.0, 1.0, (N_PIECES, 5, 5),         dtype=np.float32),
            "pieces_left": spaces.Box(0.0, 1.0, (N_PIECES,),              dtype=np.float32),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)

        # runtime state – populated by reset()
        self.board: np.ndarray = None
        self.piece_shape_ids: list = None   # length-3 list; –1 means slot used
        self.score: int = 0
        self.steps: int = 0
        self.combo_streak: int = 0

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.piece_shape_ids = self._sample_pieces()
        self.score = 0
        self.steps = 0
        self.combo_streak = 0

        obs = self._get_obs()
        info = {"action_mask": self.action_masks(), "score": 0}
        return obs, info

    def step(self, action: int):
        assert self.board is not None, "Call reset() before step()"

        piece_idx, row, col = _decode_action(action)

        # --- place the piece ---
        shape = SHAPES[self.piece_shape_ids[piece_idx]]
        for dr, dc in shape:
            self.board[row + dr, col + dc] = 1.0

        self.piece_shape_ids[piece_idx] = -1  # mark slot as used

        # --- clear completed lines (rows + cols simultaneously) ---
        lines = self._clear_lines()
        self.score += lines
        self.steps += 1

        # --- refill pieces when all three are used ---
        if all(pid == -1 for pid in self.piece_shape_ids):
            self.piece_shape_ids = self._sample_pieces()

        # --- compute reward ---
        if lines > 0:
            self.combo_streak += 1
            reward = float(lines ** 2) * (1.0 + COMBO_STREAK_BONUS * self.combo_streak)
        else:
            self.combo_streak = 0
            reward = 0.0
        if self.reward_mode == "dense":
            reward += self._dense_shaping()

        # --- check termination ---
        mask = self.action_masks()
        terminated = not np.any(mask)
        if terminated:
            reward -= 10.0

        obs = self._get_obs()
        info = {
            "action_mask":   mask,
            "score":         self.score,
            "steps":         self.steps,
            "lines_cleared": lines,
            "combo_streak":  self.combo_streak,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Action masking  (called by SB3-contrib MaskablePPO and custom DQN)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return a boolean array of shape (N_ACTIONS,) indicating legal actions."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for piece_idx in range(N_PIECES):
            pid = self.piece_shape_ids[piece_idx]
            if pid == -1:
                continue
            shape = SHAPES[pid]
            base = piece_idx * 64
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if self._can_place(shape, row, col):
                        mask[base + row * BOARD_SIZE + col] = True
        return mask

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def _render_text(self):
        sep = "+" + "-" * (BOARD_SIZE * 2 - 1) + "+"
        print(f"\nScore: {self.score}  |  Steps: {self.steps}")
        print(sep)
        for row in self.board:
            print("|" + " ".join("█" if c else "·" for c in row) + "|")
        print(sep)
        from .shapes import SHAPE_NAMES
        for i, pid in enumerate(self.piece_shape_ids):
            name = SHAPE_NAMES[pid] if pid != -1 else "(used)"
            print(f"  Piece {i}: {name}")

    def _render_rgb(self) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB image of the board (requires matplotlib)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        fig, ax = plt.subplots(figsize=(4, 4))
        display = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
        display[self.board == 1] = [70, 130, 180]   # steel-blue for filled
        display[self.board == 0] = [240, 240, 240]  # light-grey for empty
        ax.imshow(display)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Score: {self.score}")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf))[:, :, :3]
        return img

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _can_place(self, shape, row: int, col: int) -> bool:
        for dr, dc in shape:
            r, c = row + dr, col + dc
            if r >= BOARD_SIZE or c >= BOARD_SIZE:
                return False
            if self.board[r, c] == 1.0:
                return False
        return True

    def _clear_lines(self) -> int:
        """Identify complete rows and columns first, then clear all at once."""
        rows_done = [r for r in range(BOARD_SIZE) if np.all(self.board[r, :] == 1.0)]
        cols_done = [c for c in range(BOARD_SIZE) if np.all(self.board[:, c] == 1.0)]
        for r in rows_done:
            self.board[r, :] = 0.0
        for c in cols_done:
            self.board[:, c] = 0.0
        return len(rows_done) + len(cols_done)

    def _sample_pieces(self) -> list:
        return list(self.np_random.integers(0, N_SHAPES, size=N_PIECES))

    def _get_obs(self) -> dict:
        pieces_grid = np.zeros((N_PIECES, 5, 5), dtype=np.float32)
        pieces_left = np.zeros(N_PIECES, dtype=np.float32)
        for i, pid in enumerate(self.piece_shape_ids):
            if pid != -1:
                pieces_grid[i] = shape_to_grid(SHAPES[pid])
                pieces_left[i] = 1.0
        return {"board": self.board.copy(), "pieces": pieces_grid, "pieces_left": pieces_left}

    def _dense_shaping(self) -> float:
        return HOLE_PENALTY * self._count_holes() + BUMPINESS_PENALTY * self._count_bumpiness()

    def _count_holes(self) -> int:
        """A hole is an empty cell below at least one filled cell in the same column."""
        holes = 0
        for col in range(BOARD_SIZE):
            found_block = False
            for row in range(BOARD_SIZE):
                if self.board[row, col] == 1.0:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def _count_bumpiness(self) -> int:
        """Sum of absolute height differences between adjacent columns."""
        heights = []
        for col in range(BOARD_SIZE):
            filled = np.where(self.board[:, col] == 1.0)[0]
            heights.append(BOARD_SIZE - int(filled[0]) if len(filled) > 0 else 0)
        return sum(abs(heights[i] - heights[i + 1]) for i in range(BOARD_SIZE - 1))


# ------------------------------------------------------------------
# Module-level utility (used by agents)
# ------------------------------------------------------------------

def _decode_action(action: int):
    """Decode flat action index → (piece_idx, row, col)."""
    piece_idx = action // 64
    remainder = action % 64
    row = remainder // BOARD_SIZE
    col = remainder % BOARD_SIZE
    return piece_idx, row, col


def encode_action(piece_idx: int, row: int, col: int) -> int:
    return piece_idx * 64 + row * BOARD_SIZE + col
