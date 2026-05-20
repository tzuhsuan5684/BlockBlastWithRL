"""PPO agent using the shared BlockBlastActorCritic backbone.

This is the "方案二" path from README §8: a from-scratch PPO that uses
agents.network.BlockBlastActorCritic so the backbone matches Member B's DQN
exactly. The only RL machinery here is clipped surrogate loss, GAE
(in RolloutBuffer), and entropy regularization. Action masking is applied
to the logits before every Categorical so illegal actions contribute 0
probability and 0 entropy.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.network import BlockBlastActorCritic
from .network_heuristic import BlockBlastActorCriticH
from .rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        use_heuristics: bool = False,
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device)
        self.use_heuristics = use_heuristics
        if use_heuristics:
            self.model = BlockBlastActorCriticH().to(self.device)
        else:
            self.model = BlockBlastActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _to_torch(self, obs: dict, action_mask: np.ndarray):
        """VecEnv obs (numpy) → torch tensors on device.

        board comes in as (N, 8, 8); BlockBlastActorCritic expects (N, 1, 8, 8).
        Returns heuristics as None when the agent is not configured to use them.
        """
        board       = torch.from_numpy(obs["board"]).to(self.device).unsqueeze(1)
        pieces      = torch.from_numpy(obs["pieces"]).to(self.device)
        pieces_left = torch.from_numpy(obs["pieces_left"]).to(self.device)
        heuristics  = (torch.from_numpy(obs["heuristics"]).to(self.device)
                       if self.use_heuristics else None)
        mask        = torch.from_numpy(action_mask).to(self.device)
        return board, pieces, pieces_left, heuristics, mask

    def _forward(self, board: torch.Tensor, pieces: torch.Tensor,
                 pieces_left: torch.Tensor, heuristics: torch.Tensor | None):
        """Forward pass dispatching on whether heuristics are wired in."""
        if self.use_heuristics:
            return self.model(board, pieces, pieces_left, heuristics)
        return self.model(board, pieces, pieces_left)

    def _masked_dist_and_value(self, board: torch.Tensor, pieces: torch.Tensor,
                                pieces_left: torch.Tensor, heuristics: torch.Tensor | None,
                                mask: torch.Tensor):
        """Forward pass + apply mask → (Categorical, value)."""
        logits, value = self._forward(board, pieces, pieces_left, heuristics)
        logits = logits.masked_fill(~mask, float("-inf"))
        return Categorical(logits=logits), value.squeeze(-1)

    # ------------------------------------------------------------------
    # Rollout-time API (called by train loop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: dict, action_mask: np.ndarray):
        """Sample action stochastically for rollout collection.

        Returns numpy arrays so they can go straight into the env and buffer.
        """
        board, pieces, pieces_left, heuristics, mask = self._to_torch(obs, action_mask)
        dist, value = self._masked_dist_and_value(board, pieces, pieces_left, heuristics, mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    @torch.no_grad()
    def predict_values(self, obs: dict, action_mask: np.ndarray) -> np.ndarray:
        """V(s) for bootstrapping the final step of a rollout."""
        board, pieces, pieces_left, heuristics, mask = self._to_torch(obs, action_mask)
        _, value = self._masked_dist_and_value(board, pieces, pieces_left, heuristics, mask)
        return value.cpu().numpy()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> dict:
        pg_losses, vf_losses, ent_terms, kls, clipfracs = [], [], [], [], []
        all_values, all_returns = [], []

        for _ in range(self.n_epochs):
            for batch in buffer.get(self.batch_size, self.device):
                board = batch.boards.unsqueeze(1)  # (B, 1, 8, 8)
                if self.use_heuristics:
                    logits, value = self.model(board, batch.pieces, batch.pieces_left, batch.heuristics)
                else:
                    logits, value = self.model(board, batch.pieces, batch.pieces_left)
                value = value.squeeze(-1)

                logits = logits.masked_fill(~batch.action_masks, float("-inf"))
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch.actions)
                entropy = dist.entropy().mean()

                # advantage normalization per minibatch (SB3 convention)
                adv = batch.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # clipped surrogate loss
                ratio = torch.exp(new_log_probs - batch.old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, batch.returns)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - batch.old_log_probs
                    # Schulman's k3 estimator, less noisy than mean(log_ratio)
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

                pg_losses.append(policy_loss.item())
                vf_losses.append(value_loss.item())
                ent_terms.append(entropy.item())
                kls.append(approx_kl)
                clipfracs.append(clipfrac)
                all_values.append(value.detach().cpu().numpy())
                all_returns.append(batch.returns.detach().cpu().numpy())

        y_pred = np.concatenate(all_values)
        y_true = np.concatenate(all_returns)
        var_y = float(np.var(y_true))
        explained_var = float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

        return {
            "policy_loss":    float(np.mean(pg_losses)),
            "value_loss":     float(np.mean(vf_losses)),
            "entropy":        float(np.mean(ent_terms)),
            "approx_kl":      float(np.mean(kls)),
            "clip_fraction":  float(np.mean(clipfracs)),
            "explained_var":  explained_var,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str, **extra):
        torch.save(
            {"model_state_dict": self.model.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict(),
             "use_heuristics": self.use_heuristics,
             **extra},
            path,
        )

    def set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def load(self, path: str, load_optimizer: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt
