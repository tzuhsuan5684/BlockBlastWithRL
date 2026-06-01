"""PPO training entry point — Member C.

Uses SubprocVecEnv for parallel rollout and the shared BlockBlastActorCritic
backbone from agents.network (Method 2 from README §8). Hyperparameters
match SB3 MaskablePPO defaults so the PPO-vs-DQN comparison can later be
defended as "neither method was hand-tuned to win".

Run:
    uv run python -m agents.ppo.train_ppo --reward sparse --seed 0
    uv run python -m agents.ppo.train_ppo --reward dense  --seed 0
"""
from __future__ import annotations

import argparse
import re
import time
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from env import BlockBlastEnv
from env.block_blast_env import HEURISTIC_DIM
from agents.ppo.ppo_agent import PPOAgent
from agents.ppo.rollout_buffer import RolloutBuffer
from agents.ppo.afterstate import (
    compute_afterstate_features_batch,
    AFTERSTATE_FEATURE_DIM,
)


# Module-level so SubprocVecEnv can pickle it on Windows (spawn start method).
def _make_env(reward_mode: str, seed: int):
    env = BlockBlastEnv(reward_mode=reward_mode)
    env.reset(seed=seed)
    return env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward",      type=str, default="sparse", choices=["sparse", "dense"])
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-envs",      type=int, default=8)
    p.add_argument("--n-steps",     type=int, default=128)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--n-epochs",    type=int, default=10)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-range",    type=float, default=0.2)
    p.add_argument("--ent-coef",      type=float, default=0.01)
    p.add_argument("--vf-coef",       type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--use-heuristics", action="store_true",
                   help="feed Tier-1 heuristic features (40-d) into the actor-critic; "
                        "enables BlockBlastActorCriticH instead of the base network")
    p.add_argument("--use-afterstate", action="store_true",
                   help="afterstate-evaluating actor (paper arXiv:2603.26765 Eq.6): "
                        "score each action's post-placement features with a shared "
                        "linear layer; mutually exclusive with --use-heuristics")
    p.add_argument("--ckpt-dir",  type=str, default="checkpoints")
    p.add_argument("--log-dir",   type=str, default="runs")
    p.add_argument("--ckpt-every", type=int, default=50_000,
                   help="save a checkpoint every N env steps")
    return p.parse_args()


def train(args):
    assert not (args.use_heuristics and args.use_afterstate), \
        "--use-heuristics and --use-afterstate are mutually exclusive"
    suffix = "_h" if args.use_heuristics else ("_af" if args.use_afterstate else "")
    tag = f"ppo_{args.reward}{suffix}_seed{args.seed}"
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    # If ckpt-dir was created by run_ppo.py its basename ends with YYYYMMDD_HHMMSS;
    # reuse that so the TensorBoard run lines up with the checkpoint folder.
    m = re.search(r"(\d{8}_\d{6})$", ckpt_dir.name)
    run_id   = m.group(1) if m else time.strftime("%Y%m%d_%H%M%S")
    log_dir  = Path(args.log_dir) / tag / run_id; log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- env ---
    env_fns = [partial(_make_env, args.reward, args.seed * 1000 + i) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    # --- agent + buffer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        lr=args.lr, clip_range=args.clip_range,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        use_heuristics=args.use_heuristics,
        use_afterstate=args.use_afterstate,
        device=device,
    )
    # Afterstate critic still uses the env's 40-d heuristic vector as its V(s)
    # input, so we need to store it in the buffer in afterstate mode too.
    buffer_heur_dim = HEURISTIC_DIM if (args.use_heuristics or args.use_afterstate) else 0
    buffer = RolloutBuffer(
        n_steps=args.n_steps, n_envs=args.n_envs,
        heuristic_dim=buffer_heur_dim,
        afterstate_feature_dim=AFTERSTATE_FEATURE_DIM if args.use_afterstate else 0,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
    )
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[{tag}] device={device}  rollout/update={args.n_steps * args.n_envs}  "
          f"updates={args.total_steps // (args.n_steps * args.n_envs)}")

    # --- initial state ---
    obs = vec_env.reset()
    masks = np.stack(vec_env.env_method("action_masks"))
    dones = np.zeros(args.n_envs, dtype=bool)

    ep_score_buf  = deque(maxlen=100)
    ep_length_buf = deque(maxlen=100)
    ep_lengths    = np.zeros(args.n_envs, dtype=np.int64)

    total_steps    = 0
    last_ckpt_step = 0
    n_updates      = args.total_steps // (args.n_steps * args.n_envs)
    t0 = time.time()

    for update in range(1, n_updates + 1):
        # --- collect rollout ---
        for _ in range(args.n_steps):
            if args.use_afterstate:
                # Each sub-env knows its current piece slots; fetch via VecEnv.
                piece_ids = vec_env.get_attr("piece_shape_ids")
                af_feats = compute_afterstate_features_batch(obs["board"], piece_ids, masks)
                actions, log_probs, values = agent.select_action(obs, masks, af_feats)
            else:
                af_feats = None
                actions, log_probs, values = agent.select_action(obs, masks)
            next_obs, rewards, dones, infos = vec_env.step(actions)
            next_masks = np.stack([info["action_mask"] for info in infos])

            buffer.add(obs, actions, log_probs, values, rewards, dones, masks,
                       afterstate_features=af_feats)
            ep_lengths += 1

            if dones.any():
                done_idx = np.where(dones)[0]
                for i in done_idx:
                    ep_score_buf.append(int(infos[i]["score"]))
                    ep_length_buf.append(int(ep_lengths[i]))
                ep_lengths[done_idx] = 0
                # info["action_mask"] is the terminal all-False mask; refresh from the (now-reset) sub-envs
                fresh = vec_env.env_method("action_masks", indices=done_idx.tolist())
                for i, m in zip(done_idx, fresh):
                    next_masks[i] = m

            obs = next_obs
            masks = next_masks

        total_steps += args.n_steps * args.n_envs

        # --- GAE + PPO update ---
        last_values = agent.predict_values(obs, masks)
        buffer.compute_returns_and_advantages(last_values, dones)
        # linear LR decay: 3e-4 → 0 over total_steps (matches SB3 default schedule)
        progress = total_steps / args.total_steps
        current_lr = args.lr * (1.0 - progress)
        agent.set_lr(current_lr)
        stats = agent.update(buffer)
        buffer.reset()

        # --- TensorBoard logging ---
        fps = int(total_steps / (time.time() - t0))
        writer.add_scalar("time/fps", fps, total_steps)
        writer.add_scalar("train/lr",            current_lr,             total_steps)
        writer.add_scalar("train/policy_loss",   stats["policy_loss"],   total_steps)
        writer.add_scalar("train/value_loss",    stats["value_loss"],    total_steps)
        writer.add_scalar("train/entropy",       stats["entropy"],       total_steps)
        writer.add_scalar("train/approx_kl",     stats["approx_kl"],     total_steps)
        writer.add_scalar("train/clip_fraction", stats["clip_fraction"], total_steps)
        writer.add_scalar("train/explained_var", stats["explained_var"], total_steps)
        if ep_score_buf:
            writer.add_scalar("rollout/ep_score_mean",  float(np.mean(ep_score_buf)),  total_steps)
            writer.add_scalar("rollout/ep_length_mean", float(np.mean(ep_length_buf)), total_steps)
            writer.add_scalar("rollout/ep_score_max",   int(np.max(ep_score_buf)),     total_steps)

        if update % 10 == 0 or update == n_updates:
            mean_sc = float(np.mean(ep_score_buf)) if ep_score_buf else float("nan")
            mean_len = float(np.mean(ep_length_buf)) if ep_length_buf else float("nan")
            print(f"  upd {update:4d}/{n_updates}  steps={total_steps:>7d}  "
                  f"score={mean_sc:6.2f}  len={mean_len:6.1f}  "
                  f"pl={stats['policy_loss']:+.3f}  vl={stats['value_loss']:.3f}  "
                  f"ent={stats['entropy']:.3f}  kl={stats['approx_kl']:.4f}  "
                  f"fps={fps}")

        # --- checkpoint ---
        if total_steps - last_ckpt_step >= args.ckpt_every or update == n_updates:
            ckpt_path = ckpt_dir / f"{tag}_step{total_steps}.pt"
            agent.save(str(ckpt_path),
                       step=total_steps, args=vars(args))
            last_ckpt_step = total_steps

    vec_env.close()
    writer.close()
    print(f"[{tag}] done. final mean score (last 100 ep) = "
          f"{float(np.mean(ep_score_buf)) if ep_score_buf else float('nan'):.2f}")


if __name__ == "__main__":
    train(parse_args())
