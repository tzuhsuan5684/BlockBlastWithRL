import torch
import torch.nn as nn


class BlockBlastNet(nn.Module):
    """
    Shared CNN backbone for both DQN (Member B) and PPO (Member C).

    Architecture (Method 1 — dual-head):

        board  (1, 8, 8) ──► CNN ──► flatten ──► 128-d ──┐
        pieces (3×5×5=75) ─► FC  ──────────────► 64-d  ──┴──► FC ──► output

    Args:
        output_dim : number of output units
                     DQN  → N_ACTIONS (192)  — Q-values
                     PPO  → N_ACTIONS (192)  — logits  (actor)
                             1               — value   (critic, separate head)
    """

    def __init__(self, output_dim: int = 192):
        super().__init__()

        # --- board branch: (1, 8, 8) → 128-d ---
        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 8, 8)
            nn.ReLU(),
            nn.Flatten(),                                  # 64×8×8 = 4096
            nn.Linear(4096, 128),
            nn.ReLU(),
        )

        # --- pieces branch: 3×5×5=75 → 64-d ---
        self.pieces_fc = nn.Sequential(
            nn.Flatten(),          # (3,5,5) → 75
            nn.Linear(75, 64),
            nn.ReLU(),
        )

        # --- combined head: 128+64=192 → output_dim ---
        self.head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, board: torch.Tensor, pieces: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board  : (B, 1, 8, 8)  float32
            pieces : (B, 3, 5, 5)  float32
        Returns:
            (B, output_dim) float32
        """
        b_feat = self.board_cnn(board)     # (B, 128)
        p_feat = self.pieces_fc(pieces)    # (B, 64)
        x = torch.cat([b_feat, p_feat], dim=1)   # (B, 192)
        return self.head(x)                # (B, output_dim)


def obs_to_tensor(obs: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a Gymnasium obs dict (numpy) to (board, pieces) tensors.
    Handles both single obs and batched obs from VecEnv.

    Usage:
        board, pieces = obs_to_tensor(obs, device)
        q = net(board, pieces)
    """
    import numpy as np

    board  = obs["board"]
    pieces = obs["pieces"]

    # add batch dim if single obs
    if board.ndim == 2:
        board  = board[np.newaxis]    # (1, 8, 8)
        pieces = pieces[np.newaxis]   # (1, 3, 5, 5)

    # add channel dim for board CNN
    if board.ndim == 3:
        board = board[:, np.newaxis]  # (B, 1, 8, 8)

    board  = torch.tensor(board,  dtype=torch.float32, device=device)
    pieces = torch.tensor(pieces, dtype=torch.float32, device=device)
    return board, pieces


# ---------------------------------------------------------------------------
# Actor-Critic variant for PPO (Member C)
# ---------------------------------------------------------------------------

class BlockBlastActorCritic(nn.Module):
    """
    Shared backbone + separate actor/critic heads for PPO.

    Usage:
        model = BlockBlastActorCritic()
        logits, value = model(board, pieces)
        # apply mask before softmax:
        logits[~mask] = float("-inf")
        dist = Categorical(logits=logits)
    """

    def __init__(self):
        super().__init__()

        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 128),
            nn.ReLU(),
        )

        self.pieces_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(75, 64),
            nn.ReLU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
        )

        self.actor  = nn.Linear(128, 192)   # logits for 192 actions
        self.critic = nn.Linear(128, 1)     # state value

    def forward(self, board: torch.Tensor, pieces: torch.Tensor):
        """
        Returns:
            logits : (B, 192)
            value  : (B, 1)
        """
        b_feat = self.board_cnn(board)
        p_feat = self.pieces_fc(pieces)
        x = self.shared(torch.cat([b_feat, p_feat], dim=1))
        return self.actor(x), self.critic(x)
