"""
跑 Random + Greedy 兩個 baseline 各 N 局,
把每局的 score / steps 收集起來, 透過 metrics_schema.save_metrics
寫成 results/baseline_<agent>_<reward>.json.

用法:
    python -m evaluation.run_baselines                       # 預設 100 局, sparse, seed=42
    python -m evaluation.run_baselines --n-episodes 50
    python -m evaluation.run_baselines --reward-mode dense
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from env import BlockBlastEnv
from evaluation.metrics_schema import save_metrics


AGENT_REGISTRY = {
    "Random": RandomAgent,
    "Greedy": GreedyAgent,
}


def run_one_agent(
    agent_name: str,
    env: BlockBlastEnv,
    n_episodes: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    跑指定 agent N 局, 回傳 (scores, steps) 兩個長度為 n_episodes 的 list.
    這裡不直接呼叫 random_agent.evaluate / greedy_agent.evaluate,
    因為他們只回傳統計量, 不回傳 raw 資料.
    """
    agent_cls = AGENT_REGISTRY[agent_name]
    agent = agent_cls(env)
    rng = np.random.default_rng(seed)

    scores: list[int] = []
    steps_list: list[int] = []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(1 << 31))
        obs, info = env.reset(seed=ep_seed)
        mask = info["action_mask"]
        steps = 0

        while True:
            action = agent.select_action(obs, mask)
            obs, _reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            steps += 1
            if terminated or truncated:
                break

        scores.append(int(info["score"]))
        steps_list.append(steps)

    return scores, steps_list


def main():
    parser = argparse.ArgumentParser(description="跑 Random + Greedy baseline 並產生 metrics JSON")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-mode", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--agents", nargs="+", default=["Random", "Greedy"],
                        choices=list(AGENT_REGISTRY.keys()))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for agent_name in args.agents:
        env = BlockBlastEnv(reward_mode=args.reward_mode)
        print(f"\n[{agent_name}] running {args.n_episodes} episodes "
              f"(reward_mode={args.reward_mode}, seed={args.seed}) ...")
        scores, steps_list = run_one_agent(agent_name, env, args.n_episodes, args.seed)

        out_path = args.output_dir / f"baseline_{agent_name}_{args.reward_mode}.json"
        payload = save_metrics(
            out_path,
            agent=agent_name,
            reward_mode=args.reward_mode,
            n_episodes=args.n_episodes,
            seed=args.seed,
            raw_scores=scores,
            raw_steps=steps_list,
            notes=f"baseline run via evaluation.run_baselines",
        )
        rows.append(payload)
        print(f"  -> {out_path}")

    # 終端對比表
    print("\n" + "=" * 72)
    print(f"{'agent':<10}{'reward':<10}{'mean_score':>14}{'std_score':>12}{'mean_steps':>14}{'std_steps':>12}")
    print("-" * 72)
    for r in rows:
        print(f"{r['agent']:<10}{r['reward_mode']:<10}"
              f"{r['mean_score']:>14.2f}{r['std_score']:>12.2f}"
              f"{r['mean_steps']:>14.2f}{r['std_steps']:>12.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
