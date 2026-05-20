"""Heuristic-aware Actor-Critic for PPO (Member C, Tier-1 features).

Extends the shared `BlockBlastActorCritic` from agents.network without
modifying it (per CLAUDE.md the shared backbone must stay untouched so
the PPO-vs-DQN comparison remains fair). The convolutional backbone and
piece encoder are reused as-is; only the post-fusion FC is replaced so
the network can additionally consume the (40,) heuristic vector emitted
by the env.

Architecture diff vs `BlockBlastActorCritic`:

    fusion(128) + pieces_left(3) + heuristic_mlp(heuristics)(32) → shared(128)

The heuristic vector goes through a small MLP first so its features are
projected into the same scale as the conv-fusion output before being
concatenated.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from agents.network import BlockBlastActorCritic


HEURISTIC_DIM = 40
HEURISTIC_HIDDEN = 32


class BlockBlastActorCriticH(BlockBlastActorCritic):
    """`BlockBlastActorCritic` + Tier-1 heuristic vector input."""

    def __init__(self, heuristic_dim: int = HEURISTIC_DIM):
        super().__init__()

        self.heuristic_mlp = nn.Sequential(
            nn.Linear(heuristic_dim, HEURISTIC_HIDDEN),
            nn.ReLU(),
        )

        # replace the shared layer to also accept the heuristic embedding
        self.shared = nn.Sequential(
            nn.Linear(128 + 3 + HEURISTIC_HIDDEN, 128),
            nn.ReLU(),
        )

    def forward(
        self,
        board: torch.Tensor,
        pieces: torch.Tensor,
        pieces_left: torch.Tensor,
        heuristics: torch.Tensor,
    ):
        """
        Args:
            board       : (B, 1, 8, 8)
            pieces      : (B, 3, 5, 5)
            pieces_left : (B, 3)
            heuristics  : (B, 40)
        Returns:
            logits : (B, 192)
            value  : (B, 1)
        """
        b = self.board_encoder(board)
        p = self._encode_pieces(pieces)
        x = self.fusion(torch.cat([b, p], dim=1))            # (B, 128)
        h = self.heuristic_mlp(heuristics)                    # (B, 32)
        x = self.shared(torch.cat([x, pieces_left, h], dim=1))  # (B, 128)
        return self.actor(x), self.critic(x)
