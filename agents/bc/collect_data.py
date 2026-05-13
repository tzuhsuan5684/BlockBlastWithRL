"""
Collect Greedy-agent demonstrations for Behavior Cloning.

Runs the GreedyAgent for --n-steps environment steps and saves
(board, pieces, action, mask, MC-discounted-return) tuples to a .npz file.
Returns are computed at episode end and discounted backward with --gamma.

Run:
    uv run python -m agents.bc.collect_data
    uv run python -m agents.bc.collect_data --n-steps 50000 --out agents/bc/data/greedy_50k.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import BlockBlastEnv
from agents.greedy_agent import GreedyAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-steps", type=int, default=50_000)
    p.add_argument("--reward",  type=str, default="dense", choices=["sparse", "dense"])
    p.add_argument("--gamma",   type=float, default=0.99)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out",     type=str, default="agents/bc/data/greedy_50k.npz")
    return p.parse_args()


def collect(args):
    env = BlockBlastEnv(reward_mode=args.reward)
    agent = GreedyAgent(env)
    rng = np.random.default_rng(args.seed)

    all_boards, all_pieces, all_actions, all_masks, all_returns = [], [], [], [], []

    # Per-episode buffers (need full episode to compute MC returns)
    ep_boards, ep_pieces, ep_actions, ep_masks, ep_rewards = [], [], [], [], []

    obs, info = env.reset(seed=int(rng.integers(1 << 31)))
    mask = info["action_mask"]
    total_steps = 0
    n_episodes = 0

    print(f"Collecting {args.n_steps} steps from GreedyAgent (reward={args.reward}) ...")

    while total_steps < args.n_steps:
        action = agent.select_action(obs, mask)

        ep_boards.append(obs["board"].copy())
        ep_pieces.append(obs["pieces"].copy())
        ep_actions.append(action)
        ep_masks.append(mask.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        mask = info["action_mask"]
        ep_rewards.append(float(reward))
        total_steps += 1

        if terminated or truncated:
            # Compute discounted MC returns backward through the episode
            G = 0.0
            ep_returns = np.empty(len(ep_rewards), dtype=np.float32)
            for t in reversed(range(len(ep_rewards))):
                G = ep_rewards[t] + args.gamma * G
                ep_returns[t] = G

            all_boards.extend(ep_boards)
            all_pieces.extend(ep_pieces)
            all_actions.extend(ep_actions)
            all_masks.extend(ep_masks)
            all_returns.extend(ep_returns.tolist())

            ep_boards, ep_pieces, ep_actions, ep_masks, ep_rewards = [], [], [], [], []
            n_episodes += 1
            obs, info = env.reset(seed=int(rng.integers(1 << 31)))
            mask = info["action_mask"]

        if total_steps % 5_000 == 0:
            print(f"  {total_steps:>6d}/{args.n_steps} steps  {n_episodes} episodes")

    # Flush partial episode at budget cutoff (no reliable returns — discard)
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        boards=np.array(all_boards,   dtype=np.float32),   # (N, 8, 8)
        pieces=np.array(all_pieces,   dtype=np.float32),   # (N, 3, 5, 5)
        actions=np.array(all_actions, dtype=np.int64),     # (N,)
        masks=np.array(all_masks,     dtype=bool),         # (N, 192)
        returns=np.array(all_returns, dtype=np.float32),   # (N,)
    )
    print(f"Saved {len(all_actions)} transitions ({n_episodes} episodes) → {out}")


if __name__ == "__main__":
    collect(parse_args())
