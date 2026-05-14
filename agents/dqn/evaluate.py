"""
DQN 評估腳本 — 載入 checkpoint，跑 N 集，輸出 JSON 給組員 E.

執行:
    uv run python -m agents.dqn.evaluate \
        --checkpoint checkpoints/dqn_sparse_seed0_step500000.pt \
        --reward sparse --episodes 100 --seed 42 \
        --out results/dqn_sparse.json \
        --notes "DQN 500k steps, lr=1e-4"
"""

import argparse

import numpy as np
import torch

from env import BlockBlastEnv
from agents.dqn.dqn_agent import DQNAgent
from agents.network import obs_to_tensor
from evaluation.metrics_schema import save_metrics


def evaluate(
    checkpoint: str,
    reward_mode: str = "sparse",
    n_episodes: int = 100,
    seed: int = 42,
    out: str = "results/dqn_sparse.json",
    notes: str = "",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    agent = DQNAgent(device=device)
    agent.load(checkpoint)
    agent.q_net.eval()

    env = BlockBlastEnv(reward_mode=reward_mode)

    raw_scores, raw_steps = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_mask"]
        done = False

        while not done:
            # deterministic: argmax of masked Q-values, no epsilon
            with torch.no_grad():
                board, pieces, pieces_left = obs_to_tensor(obs, device)
                q = agent.q_net(board, pieces, pieces_left).squeeze(0)
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                q[~mask_t] = float("-inf")
                action = int(q.argmax().item())

            obs, _, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            done = terminated or truncated

        raw_scores.append(env.score)
        raw_steps.append(env.steps)

    result = save_metrics(
        out,
        agent="DQN",
        reward_mode=reward_mode,
        n_episodes=n_episodes,
        seed=seed,
        raw_scores=raw_scores,
        raw_steps=raw_steps,
        notes=notes,
    )
    print(f"mean_score={result['mean_score']:.2f} ± {result['std_score']:.2f} "
          f"| mean_steps={result['mean_steps']:.1f} | saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reward",     default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--episodes",   default=100, type=int)
    parser.add_argument("--seed",       default=42,  type=int)
    parser.add_argument("--out",        default="results/dqn_sparse.json")
    parser.add_argument("--notes",      default="")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        reward_mode=args.reward,
        n_episodes=args.episodes,
        seed=args.seed,
        out=args.out,
        notes=args.notes,
    )
