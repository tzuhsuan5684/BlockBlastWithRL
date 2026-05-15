"""
DQN 訓練入口.

執行:
    uv run python -m agents.dqn.train_dqn
    uv run python -m agents.dqn.train_dqn --reward dense --seed 1
    uv run python -m agents.dqn.train_dqn --reward dense --seed 0 --steps 1000000
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env import BlockBlastEnv
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import ReplayBuffer
from agents.network import obs_to_tensor

# ------------------------------------------------------------------
# Hyper-parameters
# ------------------------------------------------------------------
TOTAL_STEPS      = 500_000   # total env steps across all envs
N_ENVS           = 4         # environments running in parallel → batched GPU forward pass
BUFFER_SIZE      = 500_000   # large buffer for diverse experience
BATCH_SIZE       = 256       # larger batch → better GPU throughput per update
LR               = 1e-4
GAMMA            = 0.99
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY_STEPS  = 400_000   # ε linearly drops from 1.0 → 0.05 over first 400k env steps
SOFT_UPDATE_TAU  = 0.005     # polyak coefficient for soft target update (every iteration)
N_UPDATES_PER_ITER = 2       # gradient updates per iteration → 2:1 update:env-step ratio
LEARNING_START   = 2_000     # don't update until buffer has this many transitions
CKPT_EVERY       = 50_000    # save checkpoint every N env steps
LOG_EVERY        = 1_000     # log scalars to TensorBoard every N env steps


def linear_epsilon(step: int) -> float:
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_START + (EPS_END - EPS_START) * step / EPS_DECAY_STEPS


def train(reward_mode: str = "sparse", seed: int = 0, total_steps: int = TOTAL_STEPS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device} | reward_mode: {reward_mode} | seed: {seed} | n_envs: {N_ENVS}")

    # N_ENVS independent environments — each explores independently
    envs = [BlockBlastEnv(reward_mode=reward_mode) for _ in range(N_ENVS)]
    obs_list, mask_list = [], []
    for i, e in enumerate(envs):
        obs, info = e.reset(seed=seed + i)
        obs_list.append(obs)
        mask_list.append(info["action_mask"])

    agent  = DQNAgent(device=device, lr=LR, gamma=GAMMA)
    buffer = ReplayBuffer(capacity=BUFFER_SIZE)

    run_name = f"dqn_{reward_mode}_seed{seed}"
    writer   = SummaryWriter(log_dir=f"runs/{run_name}")
    ckpt_dir = Path("checkpoints")

    # Per-env episode tracking
    ep_rewards   = np.zeros(N_ENVS)
    ep_steps_arr = np.zeros(N_ENVS, dtype=int)
    ep_count     = 0
    recent_scores = []

    t_start = time.time()

    # Each outer iteration processes N_ENVS env steps.
    # global_step counts total env steps for scheduling (epsilon, target update, etc.)
    n_iterations = total_steps // N_ENVS
    for iteration in range(1, n_iterations + 1):
        global_step = iteration * N_ENVS
        epsilon = linear_epsilon(global_step)

        # ── Batched action selection: ONE GPU forward pass for all N_ENVS obs ──
        # Stack obs from all envs into a single batch dict
        obs_batch = {
            "board":       np.stack([o["board"]       for o in obs_list]),   # (N, 8, 8)
            "pieces":      np.stack([o["pieces"]      for o in obs_list]),   # (N, 3, 5, 5)
            "pieces_left": np.stack([o["pieces_left"] for o in obs_list]),   # (N, 3)
        }
        masks_batch = np.stack(mask_list)  # (N, 192)

        agent.q_net.eval()
        with torch.no_grad():
            board, pieces, pieces_left = obs_to_tensor(obs_batch, device)
            q = agent.q_net(board, pieces, pieces_left)        # (N, 192)
            mask_t = torch.tensor(masks_batch, dtype=torch.bool, device=device)
            q[~mask_t] = float("-inf")
            greedy_actions = q.argmax(dim=1).cpu().numpy()    # (N,)
        agent.q_net.train()

        # Per-env ε-greedy: random or greedy per env independently
        actions = np.zeros(N_ENVS, dtype=int)
        for i in range(N_ENVS):
            if random.random() < epsilon:
                legal = np.where(mask_list[i])[0]
                actions[i] = int(np.random.choice(legal))
            else:
                actions[i] = int(greedy_actions[i])

        # ── Step all N_ENVS envs, push N_ENVS transitions to buffer ──
        for i in range(N_ENVS):
            next_obs, reward, terminated, truncated, next_info = envs[i].step(int(actions[i]))
            next_mask = next_info["action_mask"]
            done = terminated or truncated

            buffer.push(obs_list[i], int(actions[i]), float(reward),
                        next_obs, done, mask_list[i], next_mask)

            ep_rewards[i]   += reward
            ep_steps_arr[i] += 1

            if done:
                ep_count += 1
                score = envs[i].score
                recent_scores.append(score)
                writer.add_scalar("rollout/ep_reward", ep_rewards[i],   global_step)
                writer.add_scalar("rollout/ep_score",  score,            global_step)
                writer.add_scalar("rollout/ep_steps",  ep_steps_arr[i], global_step)
                writer.add_scalar("rollout/epsilon",   epsilon,          global_step)

                new_obs, new_info = envs[i].reset()
                obs_list[i]     = new_obs
                mask_list[i]    = new_info["action_mask"]
                ep_rewards[i]   = 0.0
                ep_steps_arr[i] = 0
            else:
                obs_list[i]  = next_obs
                mask_list[i] = next_mask

        # ── Learning: N_UPDATES_PER_ITER gradient updates per iteration (1:1 ratio) ──
        if len(buffer) >= LEARNING_START:
            loss = 0.0
            for _ in range(N_UPDATES_PER_ITER):
                batch = buffer.sample(BATCH_SIZE)
                loss  = agent.update(batch)
            agent.soft_update_target(SOFT_UPDATE_TAU)

            if global_step % LOG_EVERY == 0:
                writer.add_scalar("train/loss",    loss,    global_step)
                writer.add_scalar("train/epsilon", epsilon, global_step)
                if recent_scores:
                    writer.add_scalar("rollout/ep_score_mean",
                                      np.mean(recent_scores[-50:]), global_step)

        if global_step % CKPT_EVERY == 0:
            ckpt_path = ckpt_dir / f"dqn_{reward_mode}_seed{seed}_step{global_step}.pt"
            agent.save(ckpt_path)
            elapsed = (time.time() - t_start) / 60
            mean_score = np.mean(recent_scores[-50:]) if recent_scores else 0.0
            print(f"step {global_step:>7} | ε={epsilon:.3f} | score_mean={mean_score:.2f} "
                  f"| eps={ep_count} | {elapsed:.1f}min | saved {ckpt_path.name}")

    writer.close()
    print("Training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--seed",   default=0, type=int)
    parser.add_argument("--steps",  default=TOTAL_STEPS, type=int)
    args = parser.parse_args()
    train(reward_mode=args.reward, seed=args.seed, total_steps=args.steps)
