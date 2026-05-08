import numpy as np


class RandomAgent:
    """Uniformly sample from legal actions each step."""

    def __init__(self, env):
        self.env = env

    def select_action(self, obs, action_mask: np.ndarray) -> int:
        legal = np.where(action_mask)[0]
        return int(np.random.choice(legal))


def evaluate(env, n_episodes: int = 100, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    agent = RandomAgent(env)
    scores, lengths = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(1 << 31)))
        mask = info["action_mask"]
        total_reward, steps = 0.0, 0

        while True:
            action = agent.select_action(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            total_reward += reward
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
