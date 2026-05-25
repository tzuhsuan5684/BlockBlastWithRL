"""Afterstate-evaluating actor-critic for PPO.

Implements the actor design from Chen et al. 2026 (arXiv:2603.26765) Fig. 11
adapted to Block Blast's 192-action space:

    afterstate_features (B, 192, F) ──► Linear(F, 1) ──► (B, 192) logits
                                       └ shared across all 192 action slots ┘

The critic stays as a standard V(s) head over the env-provided heuristic
vector (40-d). Per CLAUDE.md the shared CNN backbone in agents/network.py
must stay untouched — this is an independent network for the afterstate
ablation only.

The whole module is ~2.7k parameters — orders of magnitude smaller than
BlockBlastActorCritic — which mirrors the paper's "use a tiny model when
features are informative" thesis.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BlockBlastAfterstateActorCritic(nn.Module):
    """Actor scores afterstates; critic estimates V(s_t) from heuristics."""

    def __init__(
        self,
        feature_dim: int = 9,
        heuristic_dim: int = 40,
        critic_hidden: int = 64,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Paper Fig. 11: one shared linear over the per-action feature vectors.
        self.action_scorer = nn.Linear(feature_dim, 1)

        # Standard V(s) head — uses the env's existing 40-d heuristic vector
        # so we don't have to compute a second feature set for the current state.
        self.critic = nn.Sequential(
            nn.Linear(heuristic_dim, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

    def forward(
        self,
        afterstate_features: torch.Tensor,   # (B, 192, F)
        heuristics: torch.Tensor,            # (B, 40)
    ):
        logits = self.action_scorer(afterstate_features).squeeze(-1)   # (B, 192)
        value = self.critic(heuristics)                                # (B, 1)
        return logits, value
