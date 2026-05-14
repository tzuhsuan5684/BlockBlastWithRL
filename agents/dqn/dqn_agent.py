import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.network import BlockBlastNet, obs_to_tensor


class DQNAgent:
    """
    Double DQN agent for Block Blast.

    Two networks:
      q_net      — updated every step via gradient descent
      target_net — hard-copied from q_net every `target_update_freq` steps

    Double DQN trick: q_net picks the best next action, target_net scores it.
    This reduces Q-value overestimation compared to vanilla DQN.
    """

    def __init__(
        self,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
    ):
        self.device = device
        self.gamma = gamma

        self.q_net     = BlockBlastNet(output_dim=192).to(device)
        self.target_net = BlockBlastNet(output_dim=192).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: dict, mask: np.ndarray, epsilon: float) -> int:
        """
        ε-greedy with action masking.

        With probability ε: pick a random legal action.
        Otherwise: pick the legal action with the highest Q-value.
        """
        legal = np.where(mask)[0]
        if random.random() < epsilon:
            return int(np.random.choice(legal))

        self.q_net.eval()
        with torch.no_grad():
            board, pieces, pieces_left = obs_to_tensor(obs, self.device)
            q = self.q_net(board, pieces, pieces_left).squeeze(0)   # (192,)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)
            q[~mask_t] = float("-inf")
        self.q_net.train()
        return int(q.argmax().item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, batch: dict) -> float:
        """
        One gradient step using a sampled batch.

        Bellman target (Double DQN):
            target = r + γ · Q_target(s', argmax_a Q(s', a))   if not done
                   = r                                           if done

        Returns the scalar loss value for logging.
        """
        obs       = batch["obs"]
        actions   = torch.tensor(batch["actions"],   dtype=torch.long,    device=self.device)
        rewards   = torch.tensor(batch["rewards"],   dtype=torch.float32, device=self.device)
        next_obs  = batch["next_obs"]
        dones     = torch.tensor(batch["dones"],     dtype=torch.float32, device=self.device)
        next_masks = torch.tensor(batch["next_masks"], dtype=torch.bool,  device=self.device)

        board, pieces, pieces_left = obs_to_tensor(obs, self.device)
        nb, np_, npl = obs_to_tensor(next_obs, self.device)

        # current Q values for the actions that were actually taken
        q_current = self.q_net(board, pieces, pieces_left)              # (B, 192)
        q_current = q_current.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN target
        with torch.no_grad():
            # q_net chooses the action
            q_next_online = self.q_net(nb, np_, npl)                    # (B, 192)
            q_next_online[~next_masks] = float("-inf")
            best_actions = q_next_online.argmax(dim=1, keepdim=True)    # (B, 1)

            # target_net scores it
            q_next_target = self.target_net(nb, np_, npl)               # (B, 192)
            q_next_val = q_next_target.gather(1, best_actions).squeeze(1)  # (B,)

            target = rewards + self.gamma * q_next_val * (1.0 - dones)

        loss = F.smooth_l1_loss(q_current, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Hard copy q_net weights → target_net."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net":     self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.target_net.eval()
