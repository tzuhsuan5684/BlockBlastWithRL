"""
Behavioral Cloning — supervised pretraining of BlockBlastActorCritic.

Actor head: cross-entropy loss against Greedy-agent actions (mask applied
before softmax so illegal actions have zero probability, matching inference).

Critic head (--train-critic): MSE against Monte Carlo discounted returns
collected alongside the demonstrations. The critic is optional because PPO
will fine-tune it anyway; enabling it gives a better value baseline at the
start of PPO training.

Output checkpoint is compatible with PPOAgent.load() (contains "model_state_dict").

Run:
    uv run python -m agents.bc.train_bc
    uv run python -m agents.bc.train_bc --data agents/bc/data/greedy_50k.npz --train-critic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from agents.network import BlockBlastActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",         type=str,   default="agents/bc/data/greedy_50k.npz")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--train-critic", action="store_true",
                   help="Also train critic head with MSE on MC returns")
    p.add_argument("--vf-coef",      type=float, default=0.5,
                   help="Weight on critic loss (only used with --train-critic)")
    p.add_argument("--out",          type=str,   default="agents/bc/bc_weights.pt")
    p.add_argument("--log-dir",      type=str,   default="runs/bc")
    p.add_argument("--seed",         type=int,   default=0)
    return p.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load dataset ---
    data = np.load(args.data)
    boards  = torch.tensor(data["boards"],  dtype=torch.float32)  # (N, 8, 8)
    pieces  = torch.tensor(data["pieces"],  dtype=torch.float32)  # (N, 3, 5, 5)
    actions = torch.tensor(data["actions"], dtype=torch.long)      # (N,)
    masks   = torch.tensor(data["masks"],   dtype=torch.bool)      # (N, 192)
    returns = torch.tensor(data["returns"], dtype=torch.float32)   # (N,)

    n = len(actions)
    print(f"Loaded {n} transitions from {args.data}")
    print(f"  device={device}  epochs={args.epochs}  batch={args.batch_size}  "
          f"train_critic={args.train_critic}")

    dataset = TensorDataset(boards, pieces, actions, masks, returns)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # --- model ---
    model = BlockBlastActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        actor_losses, critic_losses, total_losses = [], [], []

        for b_boards, b_pieces, b_actions, b_masks, b_returns in loader:
            b_boards  = b_boards.unsqueeze(1).to(device)  # (B, 1, 8, 8)
            b_pieces  = b_pieces.to(device)
            b_actions = b_actions.to(device)
            b_masks   = b_masks.to(device)
            b_returns = b_returns.to(device)

            logits, value = model(b_boards, b_pieces)

            # Apply mask before CE so the learned distribution respects legality.
            # Greedy actions are always legal, so their logit stays finite.
            masked_logits = logits.masked_fill(~b_masks, float("-inf"))
            actor_loss = F.cross_entropy(masked_logits, b_actions)

            if args.train_critic:
                critic_loss = F.mse_loss(value.squeeze(-1), b_returns)
                loss = actor_loss + args.vf_coef * critic_loss
            else:
                critic_loss = torch.tensor(0.0, device=device)
                loss = actor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            total_losses.append(loss.item())

        mean_actor  = float(np.mean(actor_losses))
        mean_critic = float(np.mean(critic_losses))
        mean_total  = float(np.mean(total_losses))

        writer.add_scalar("bc/actor_loss",  mean_actor,  epoch)
        writer.add_scalar("bc/critic_loss", mean_critic, epoch)
        writer.add_scalar("bc/total_loss",  mean_total,  epoch)

        print(f"  epoch {epoch:3d}/{args.epochs}  "
              f"actor={mean_actor:.4f}  critic={mean_critic:.4f}  total={mean_total:.4f}")

        if mean_total < best_loss:
            best_loss = mean_total
            torch.save({"model_state_dict": model.state_dict()}, out)

    writer.close()
    print(f"\nBC pretraining done. Best checkpoint → {out}")
    print(f"Warm-start PPO with:")
    print(f"  uv run python -m agents.ppo.train_ppo --reward dense --bc-checkpoint {out}")


if __name__ == "__main__":
    train(parse_args())
