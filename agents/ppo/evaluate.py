"""Evaluate a trained PPO checkpoint and emit JSON in Member E's schema.

Schema (per E's spec):
    {
        "agent": "PPO",
        "reward_mode": "sparse"|"dense",
        "n_episodes": int,
        "seed": int,
        "mean_score":  float,  "std_score": float,
        "mean_steps":  float,  "std_steps": float,
        "raw_scores":  [int, ...],
        "raw_steps":   [int, ...],
        "timestamp":   "YYYY-MM-DDTHH:MM:SS",
        "notes":       str
    }

Evaluation uses *deterministic* policy (argmax over masked logits) so the
reported number is the model's best-effort performance, not a sample mean
over its stochastic policy.

Run:
    uv run python -m agents.ppo.evaluate \
        --checkpoint checkpoints/ppo_sparse_seed0_step500000.pt \
        --reward sparse --episodes 100 --seed 42 \
        --out results/ppo_sparse.json --notes "PPO 500k, lr=3e-4, ent=0.01"
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from env import BlockBlastEnv
from agents.network import BlockBlastActorCritic
from agents.ppo.network_heuristic import BlockBlastActorCriticH
from agents.ppo.network_afterstate import BlockBlastAfterstateActorCritic
from agents.ppo.afterstate import compute_afterstate_features


def _obs_to_torch(obs: dict, device: torch.device, use_heuristics: bool):
    board       = torch.from_numpy(obs["board"]).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,8,8)
    pieces      = torch.from_numpy(obs["pieces"]).to(device).unsqueeze(0)              # (1,3,5,5)
    pieces_left = torch.from_numpy(obs["pieces_left"]).to(device).unsqueeze(0)         # (1,3)
    heuristics  = (torch.from_numpy(obs["heuristics"]).to(device).unsqueeze(0)
                   if use_heuristics else None)                                         # (1,40)
    return board, pieces, pieces_left, heuristics


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    reward_mode: str = "sparse",
    n_episodes: int = 100,
    seed: int = 42,
    notes: str = "",
    device: torch.device | str | None = None,
) -> dict:
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    use_heuristics = bool(ckpt.get("use_heuristics", False))
    use_afterstate = bool(ckpt.get("use_afterstate", False))
    if use_afterstate:
        model = BlockBlastAfterstateActorCritic().to(device)
    elif use_heuristics:
        model = BlockBlastActorCriticH().to(device)
    else:
        model = BlockBlastActorCritic().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = BlockBlastEnv(reward_mode=reward_mode)
    rng = np.random.default_rng(seed)

    raw_scores, raw_steps = [], []
    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(1 << 31)))
        mask = info["action_mask"]
        steps = 0
        while True:
            if use_afterstate:
                af = compute_afterstate_features(obs["board"], env.piece_shape_ids, mask)
                af_t = torch.from_numpy(af).to(device).unsqueeze(0)            # (1, 192, F)
                heur_t = torch.from_numpy(obs["heuristics"]).to(device).unsqueeze(0)  # (1, 40)
                logits, _ = model(af_t, heur_t)
            else:
                board, pieces, pieces_left, heuristics = _obs_to_torch(obs, device, use_heuristics)
                if use_heuristics:
                    logits, _ = model(board, pieces, pieces_left, heuristics)
                else:
                    logits, _ = model(board, pieces, pieces_left)
            mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            action = int(logits.argmax(dim=1).item())

            obs, _, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            steps += 1
            if terminated or truncated:
                break
        raw_scores.append(int(info["score"]))
        raw_steps.append(int(steps))

    return {
        "agent":       "PPO",
        "reward_mode": reward_mode,
        "n_episodes":  n_episodes,
        "seed":        seed,
        "mean_score":  float(np.mean(raw_scores)),
        "std_score":   float(np.std(raw_scores)),
        "mean_steps":  float(np.mean(raw_steps)),
        "std_steps":   float(np.std(raw_steps)),
        "raw_scores":  raw_scores,
        "raw_steps":   raw_steps,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "notes":       notes,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--reward",     type=str, default="sparse", choices=["sparse", "dense"])
    p.add_argument("--episodes",   type=int, default=100)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        type=str, required=True,
                   help="output JSON path, e.g. results/ppo_sparse.json")
    p.add_argument("--notes",      type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    result = evaluate(
        checkpoint_path=args.checkpoint,
        reward_mode=args.reward,
        n_episodes=args.episodes,
        seed=args.seed,
        notes=args.notes,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[eval] {args.checkpoint}")
    print(f"  mean_score = {result['mean_score']:.2f} ± {result['std_score']:.2f}")
    print(f"  mean_steps = {result['mean_steps']:.2f} ± {result['std_steps']:.2f}")
    print(f"  wrote → {out_path}")


if __name__ == "__main__":
    main()
