"""Rollout buffer for PPO.

Stores transitions collected from a SubprocVecEnv for one rollout, computes
GAE advantages and bootstrapped returns, and yields shuffled mini-batches.

Key difference from a vanilla PPO buffer: each transition also stores the
action mask used at that state. Member C's policy update re-applies the mask
to the logits before computing log-probs and entropy, so the mask must
survive into the update step.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MiniBatch:
    boards: torch.Tensor          # (B, 8, 8) float32
    pieces: torch.Tensor          # (B, 3, 5, 5) float32
    pieces_left: torch.Tensor     # (B, 3) float32
    heuristics: torch.Tensor      # (B, H) float32 — empty (B, 0) if heuristics disabled
    actions: torch.Tensor         # (B,) int64
    old_log_probs: torch.Tensor   # (B,) float32
    advantages: torch.Tensor      # (B,) float32
    returns: torch.Tensor         # (B,) float32
    action_masks: torch.Tensor    # (B, 192) bool


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        n_actions: int = 192,
        board_shape: tuple = (8, 8),
        pieces_shape: tuple = (3, 5, 5),
        heuristic_dim: int = 0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_actions = n_actions
        self.heuristic_dim = heuristic_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.boards       = np.zeros((n_steps, n_envs, *board_shape),  dtype=np.float32)
        self.pieces       = np.zeros((n_steps, n_envs, *pieces_shape), dtype=np.float32)
        self.pieces_left  = np.zeros((n_steps, n_envs, 3),             dtype=np.float32)
        self.heuristics   = np.zeros((n_steps, n_envs, heuristic_dim), dtype=np.float32)
        self.actions      = np.zeros((n_steps, n_envs),                dtype=np.int64)
        self.log_probs    = np.zeros((n_steps, n_envs),                dtype=np.float32)
        self.values       = np.zeros((n_steps, n_envs),                dtype=np.float32)
        self.rewards      = np.zeros((n_steps, n_envs),                dtype=np.float32)
        self.dones        = np.zeros((n_steps, n_envs),                dtype=np.bool_)
        self.action_masks = np.zeros((n_steps, n_envs, n_actions),     dtype=np.bool_)

        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns    = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, log_prob, value, reward, done, action_mask):
        """obs is a dict from VecEnv: {board, pieces, pieces_left[, heuristics]}.

        The heuristics key is only read when this buffer was constructed with
        heuristic_dim > 0; otherwise it is ignored (so the same call site works
        for the no-heuristics ablation).
        """
        assert not self.full, "buffer is full — call reset() first"
        i = self.ptr
        self.boards[i]       = obs["board"]
        self.pieces[i]       = obs["pieces"]
        self.pieces_left[i]  = obs["pieces_left"]
        if self.heuristic_dim > 0:
            self.heuristics[i] = obs["heuristics"]
        self.actions[i]      = action
        self.log_probs[i]    = log_prob
        self.values[i]       = value
        self.rewards[i]      = reward
        self.dones[i]        = done
        self.action_masks[i] = action_mask
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_values: np.ndarray, last_dones: np.ndarray):
        """GAE-λ backward pass.

        last_values : (n_envs,) value estimate V(s_T) for the obs *after* the
                      final rollout step. Should be 0 for terminal states.
        last_dones  : (n_envs,) bool, whether s_T is terminal.
        """
        assert self.full, "rollout not complete"
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
                next_values = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get(self, batch_size: int, device: torch.device):
        """Yield shuffled mini-batches of the flat rollout (n_steps * n_envs)."""
        assert self.full, "rollout not complete — call compute_returns_and_advantages first"
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        flat_boards       = self.boards.reshape(total, *self.boards.shape[2:])
        flat_pieces       = self.pieces.reshape(total, *self.pieces.shape[2:])
        flat_pieces_left  = self.pieces_left.reshape(total, 3)
        flat_heuristics   = self.heuristics.reshape(total, self.heuristic_dim)
        flat_actions      = self.actions.reshape(total)
        flat_log_probs    = self.log_probs.reshape(total)
        flat_advantages   = self.advantages.reshape(total)
        flat_returns      = self.returns.reshape(total)
        flat_action_masks = self.action_masks.reshape(total, self.n_actions)

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield MiniBatch(
                boards        = torch.from_numpy(flat_boards[idx]).to(device),
                pieces        = torch.from_numpy(flat_pieces[idx]).to(device),
                pieces_left   = torch.from_numpy(flat_pieces_left[idx]).to(device),
                heuristics    = torch.from_numpy(flat_heuristics[idx]).to(device),
                actions       = torch.from_numpy(flat_actions[idx]).to(device),
                old_log_probs = torch.from_numpy(flat_log_probs[idx]).to(device),
                advantages    = torch.from_numpy(flat_advantages[idx]).to(device),
                returns       = torch.from_numpy(flat_returns[idx]).to(device),
                action_masks  = torch.from_numpy(flat_action_masks[idx]).to(device),
            )

    def reset(self):
        self.ptr = 0
        self.full = False
