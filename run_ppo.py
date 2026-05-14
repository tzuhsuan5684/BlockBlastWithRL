"""
PPO train / eval / demo automation script.

Usage:
    uv run python run_ppo.py                          # default: all, dense
    uv run python run_ppo.py train --reward sparse
    uv run python run_ppo.py train --reward both
    uv run python run_ppo.py eval  --reward dense
    uv run python run_ppo.py eval  --checkpoint path/to.pt
    uv run python run_ppo.py demo
    uv run python run_ppo.py all   --total-steps 1000000
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CKPT_DIR    = Path("checkpoints")
RESULTS_DIR = Path("results")


def run(cmd: list[str]):
    print("$", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def header(msg: str):
    print(f"\n{'='*40}\n  {msg}\n{'='*40}")


def latest_checkpoint(reward: str, seed: int) -> Path | None:
    pattern = f"ppo_{reward}_seed{seed}_step*.pt"
    candidates = sorted(
        CKPT_DIR.rglob(pattern),
        key=lambda p: int(p.stem.split("_step")[-1]),
        reverse=True,
    )
    return candidates[0] if candidates else None


def train(reward: str, seed: int, total_steps: int, n_envs: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CKPT_DIR / f"{reward}_seed{seed}_{ts}"
    header(f"TRAIN  reward={reward}  steps={total_steps}  seed={seed}")
    print(f"  ckpt-dir: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run([
        "uv", "run", "python", "-m", "agents.ppo.train_ppo",
        "--reward",      reward,
        "--seed",        str(seed),
        "--total-steps", str(total_steps),
        "--n-envs",      str(n_envs),
        "--ckpt-dir",    str(run_dir),
    ])


def eval_(reward: str, seed: int, n_episodes: int, total_steps: int, checkpoint: str | None):
    header(f"EVAL  reward={reward}  episodes={n_episodes}")
    ckpt = Path(checkpoint) if checkpoint else latest_checkpoint(reward, seed)
    if ckpt is None:
        print(f"[skip] no checkpoint found for reward={reward}")
        return
    print(f"  checkpoint: {ckpt}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"ppo_{reward}_seed{seed}_{ts}.json"
    run([
        "uv", "run", "python", "-m", "agents.ppo.evaluate",
        "--checkpoint", str(ckpt),
        "--reward",     reward,
        "--episodes",   str(n_episodes),
        "--seed",       "42",
        "--out",        str(out),
        "--notes",      f"auto-eval seed={seed} steps={total_steps}",
    ])


def demo(seed: int, checkpoint: str | None, fps: int):
    header("DEMO  (pygame)")
    ckpt = Path(checkpoint) if checkpoint else (
        latest_checkpoint("dense", seed) or latest_checkpoint("sparse", seed)
    )
    if ckpt:
        print(f"  agent=ppo  checkpoint={ckpt}")
        run(["uv", "run", "python", "demo/play.py", "--agent", "ppo",
             "--checkpoint", str(ckpt), "--fps", str(fps)])
    else:
        print("  no checkpoint found -- running greedy agent")
        run(["uv", "run", "python", "demo/play.py", "--agent", "greedy", "--fps", str(fps)])


def main():
    parser = argparse.ArgumentParser(description="PPO automation script")
    parser.add_argument("mode", nargs="?", default="all",
                        choices=["train", "eval", "demo", "all"])
    parser.add_argument("--reward",      default="dense", choices=["sparse", "dense", "both"])
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--n-episodes",  type=int, default=100)
    parser.add_argument("--n-envs",      type=int, default=8)
    parser.add_argument("--checkpoint",  default="")
    parser.add_argument("--demo-fps",    type=int, default=10)
    args = parser.parse_args()

    rewards = ["sparse", "dense"] if args.reward == "both" else [args.reward]
    ckpt    = args.checkpoint or None

    if args.mode == "train":
        for r in rewards:
            train(r, args.seed, args.total_steps, args.n_envs)
    elif args.mode == "eval":
        for r in rewards:
            eval_(r, args.seed, args.n_episodes, args.total_steps, ckpt)
    elif args.mode == "demo":
        demo(args.seed, ckpt, args.demo_fps)
    elif args.mode == "all":
        for r in rewards:
            train(r, args.seed, args.total_steps, args.n_envs)
        for r in rewards:
            eval_(r, args.seed, args.n_episodes, args.total_steps, ckpt)

    print("\ndone.")


if __name__ == "__main__":
    main()
