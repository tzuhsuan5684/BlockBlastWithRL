import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing (obs, action, reward, next_obs, done, mask, next_mask).

    obs / next_obs are dicts with keys: "board" (8,8), "pieces" (3,5,5), "pieces_left" (3,).
    mask / next_mask are (192,) bool arrays used to filter illegal actions when computing
    the Bellman target (Double DQN style).
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ):
        self.buffer.append((
            {k: v.copy() for k, v in obs.items()},
            action,
            reward,
            {k: v.copy() for k, v in next_obs.items()},
            done,
            mask.copy(),
            next_mask.copy(),
        ))

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(self.buffer, batch_size)
        obs_b, actions, rewards, next_obs_b, dones, masks, next_masks = zip(*batch)

        def stack_obs(obs_list):
            return {
                "board":       np.stack([o["board"]       for o in obs_list]),
                "pieces":      np.stack([o["pieces"]      for o in obs_list]),
                "pieces_left": np.stack([o["pieces_left"] for o in obs_list]),
            }

        return {
            "obs":       stack_obs(obs_b),
            "actions":   np.array(actions,  dtype=np.int64),
            "rewards":   np.array(rewards,  dtype=np.float32),
            "next_obs":  stack_obs(next_obs_b),
            "dones":     np.array(dones,    dtype=np.float32),
            "masks":     np.stack(masks),
            "next_masks": np.stack(next_masks),
        }

    def __len__(self) -> int:
        return len(self.buffer)
