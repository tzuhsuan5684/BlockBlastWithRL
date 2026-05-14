import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockBlastNet(nn.Module):
    """
    Shared CNN backbone for both DQN (Member B) and PPO (Member C).

    Architecture:

        board  (1, 8, 8)  ──► board_encoder ──► (64, 8, 8) ──┐
        piece_0 (1, 5, 5) ─┐                                   │ cat on channel
        piece_1 (1, 5, 5) ─┼─► piece_encoder ──► pad to 8×8   │ dim → (112, 8, 8)
        piece_2 (1, 5, 5) ─┘   (16ch each, weight-shared)  ──►┘
                                                               ↓
                                               fusion conv → flatten → 128-d
                                                               ↓
                                          cat pieces_left (3-d) → 131-d → head

    Spatial fusion lets the conv layers compare board features and piece shapes
    at the same grid resolution before compressing to a vector.

    Args:
        output_dim : DQN → 192 (Q-values), PPO actor → 192 (logits)
    """

    def __init__(self, output_dim: int = 192):
        super().__init__()

        # board: (1, 8, 8) → (64, 8, 8)  [no flatten yet]
        self.board_encoder = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1),  # (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 8, 8)
            nn.ReLU(),
        )

        # per-piece: (1, 5, 5) → (16, 5, 5), weight-shared
        self.piece_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (16, 5, 5)
            nn.ReLU(),
        )

        # fusion: (64+48, 8, 8) → 128-d
        self.fusion = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=3, padding=1),  # (64, 8, 8)
            nn.ReLU(),
            nn.Flatten(),                                   # 64×8×8 = 4096
            nn.Linear(4096, 128),
            nn.ReLU(),
        )

        # head: 128 + 3 (pieces_left) → output_dim
        self.head = nn.Sequential(
            nn.Linear(131, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def _encode_pieces(self, pieces: torch.Tensor) -> torch.Tensor:
        """(B, 3, 5, 5) → (B, 48, 8, 8): encode + pad each piece to board size."""
        B = pieces.shape[0]
        p = self.piece_encoder(pieces.view(B * 3, 1, 5, 5))  # (B*3, 16, 5, 5)
        p = F.pad(p, (0, 3, 0, 3))                            # (B*3, 16, 8, 8)
        return p.view(B, 48, 8, 8)                             # (B,   48, 8, 8)

    def forward(self, board: torch.Tensor, pieces: torch.Tensor, pieces_left: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board       : (B, 1, 8, 8)  float32
            pieces      : (B, 3, 5, 5)  float32
            pieces_left : (B, 3)        float32  binary mask of remaining slots
        Returns:
            (B, output_dim) float32
        """
        b = self.board_encoder(board)        # (B, 64, 8, 8)
        p = self._encode_pieces(pieces)      # (B, 48, 8, 8)
        x = self.fusion(torch.cat([b, p], dim=1))       # (B, 128)
        x = torch.cat([x, pieces_left], dim=1)           # (B, 131)
        return self.head(x)                              # (B, output_dim)


def obs_to_tensor(obs: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a Gymnasium obs dict (numpy) to (board, pieces, pieces_left) tensors.
    Handles both single obs and batched obs from VecEnv.

    Usage:
        board, pieces, pieces_left = obs_to_tensor(obs, device)
        q = net(board, pieces, pieces_left)
    """
    import numpy as np

    board       = obs["board"]
    pieces      = obs["pieces"]
    pieces_left = obs["pieces_left"]

    # add batch dim if single obs
    if board.ndim == 2:
        board       = board[np.newaxis]       # (1, 8, 8)
        pieces      = pieces[np.newaxis]      # (1, 3, 5, 5)
        pieces_left = pieces_left[np.newaxis] # (1, 3)

    # add channel dim for board CNN
    if board.ndim == 3:
        board = board[:, np.newaxis]  # (B, 1, 8, 8)

    board       = torch.tensor(board,       dtype=torch.float32, device=device)
    pieces      = torch.tensor(pieces,      dtype=torch.float32, device=device)
    pieces_left = torch.tensor(pieces_left, dtype=torch.float32, device=device)
    return board, pieces, pieces_left


# ---------------------------------------------------------------------------
# Actor-Critic variant for PPO (Member C)
# ---------------------------------------------------------------------------

class BlockBlastActorCritic(nn.Module):
    """
    Shared spatial-fusion backbone + separate actor/critic heads for PPO.

    Usage:
        model = BlockBlastActorCritic()
        logits, value = model(board, pieces, pieces_left)
        logits[~mask] = float("-inf")
        dist = Categorical(logits=logits)
    """

    def __init__(self):
        super().__init__()

        # board: (1, 8, 8) → (64, 8, 8)
        self.board_encoder = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # per-piece: (1, 5, 5) → (16, 5, 5), weight-shared
        self.piece_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # fusion: (112, 8, 8) → 128-d
        self.fusion = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 128),
            nn.ReLU(),
        )

        # shared FC after spatial fusion + pieces_left
        self.shared = nn.Sequential(
            nn.Linear(131, 128),   # 128 (fusion) + 3 (pieces_left)
            nn.ReLU(),
        )

        self.actor  = nn.Linear(128, 192)  # logits for 192 actions
        self.critic = nn.Linear(128, 1)    # state value

    def _encode_pieces(self, pieces: torch.Tensor) -> torch.Tensor:
        """(B, 3, 5, 5) → (B, 48, 8, 8)"""
        B = pieces.shape[0]
        p = self.piece_encoder(pieces.view(B * 3, 1, 5, 5))  # (B*3, 16, 5, 5)
        p = F.pad(p, (0, 3, 0, 3))                            # (B*3, 16, 8, 8)
        return p.view(B, 48, 8, 8)

    def forward(self, board: torch.Tensor, pieces: torch.Tensor, pieces_left: torch.Tensor):
        """
        Args:
            board       : (B, 1, 8, 8)
            pieces      : (B, 3, 5, 5)
            pieces_left : (B, 3)        binary mask of remaining slots
        Returns:
            logits : (B, 192)
            value  : (B, 1)
        """
        b = self.board_encoder(board)                          # (B, 64, 8, 8)
        p = self._encode_pieces(pieces)                        # (B, 48, 8, 8)
        x = self.fusion(torch.cat([b, p], dim=1))              # (B, 128)
        x = self.shared(torch.cat([x, pieces_left], dim=1))   # (B, 128)
        return self.actor(x), self.critic(x)
