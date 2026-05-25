"""End-to-end experiment runner for Block Blast RL.

Single entry point for the five agents in the comparison:
    Random   uniform-from-legal baseline
    Greedy   one-step-lookahead baseline
    DQN      Double-DQN on the shared CNN backbone
    PPO      vanilla PPO on the shared CNN backbone
    PPO-AF   afterstate-evaluating PPO variant

Subcommands
-----------
    eval     evaluate all (or a subset) of agents, write results/*.json,
             then aggregate -> summary.csv + comparison_*.png
    train    train a PPO or DQN agent (wraps the per-agent train_*.py)
    demo     open the pygame visualizer for a chosen agent

Each eval JSON follows evaluation/metrics_schema. Checkpoints are
auto-detected (latest step in the most recent run folder); pass
--<agent>-ckpt to pin a specific file, or --skip-<agent> to drop one
from the comparison.

Run
---
    uv run python run_experiments.py eval
    uv run python run_experiments.py eval --episodes 200 --skip-dqn
    uv run python run_experiments.py eval --ppo-ckpt checkpoints/dense_seed0_<ts>/ppo_dense_seed0_step999424.pt

    uv run python run_experiments.py train ppo --reward dense --total-steps 1000000
    uv run python run_experiments.py train ppo --reward both           # sparse + dense back-to-back
    uv run python run_experiments.py train ppo --use-afterstate        # PPO-AF variant
    uv run python run_experiments.py train dqn --reward dense --seed 0

    uv run python run_experiments.py demo                              # latest PPO checkpoint
    uv run python run_experiments.py demo --agent greedy
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from env import BlockBlastEnv
from evaluation.metrics_schema import save_metrics
from evaluation.run_baselines import run_one_agent
from agents.ppo.evaluate import evaluate as ppo_evaluate
from agents.dqn.evaluate import evaluate as dqn_evaluate


CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# checkpoint auto-detection
# ---------------------------------------------------------------------------

def _step_of(p: Path) -> int:
    m = re.search(r"step(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


def _latest(glob: str) -> Path | None:
    cands = sorted(CKPT_DIR.glob(glob), key=_step_of)
    return cands[-1] if cands else None


def autoselect_ppo_dense() -> Path | None:
    """Latest step inside the most recent dense_seed0_<timestamp>/ folder."""
    runs = sorted(p for p in CKPT_DIR.glob("dense_seed0_*") if p.is_dir())
    if runs:
        latest = _latest(f"{runs[-1].name}/ppo_dense_seed0_step*.pt")
        if latest is not None:
            return latest
    return _latest("ppo_dense_seed0_step*.pt")


def autoselect_ppo_af() -> Path | None:
    return _latest("ppo_*_af_seed*_step*.pt")


def autoselect_dqn() -> Path | None:
    return _latest("dqn_*_seed*_step*.pt")


def _reward_from_ckpt(name: str, default: str = "sparse") -> str:
    m = re.match(r"(?:ppo|dqn)_(sparse|dense)", name)
    return m.group(1) if m else default


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _run_baseline(agent_name: str, reward_mode: str, n_episodes: int, seed: int) -> Path:
    env = BlockBlastEnv(reward_mode=reward_mode)
    print(f"\n[{agent_name}] {n_episodes} ep, reward={reward_mode}, seed={seed} ...")
    scores, steps = run_one_agent(agent_name, env, n_episodes, seed)
    out = RESULTS_DIR / f"baseline_{agent_name}_{reward_mode}.json"
    payload = save_metrics(
        out, agent=agent_name, reward_mode=reward_mode,
        n_episodes=n_episodes, seed=seed,
        raw_scores=scores, raw_steps=steps,
        notes="final-report comparison",
    )
    print(f"  mean_score = {payload['mean_score']:.2f} +/- {payload['std_score']:.2f}"
          f"  mean_steps = {payload['mean_steps']:.1f}")
    print(f"  -> {out}")
    return out


def _run_dqn_eval(ckpt: Path, reward_mode: str, n_episodes: int, seed: int) -> Path:
    out = RESULTS_DIR / f"dqn_{reward_mode}.json"
    print(f"\n[DQN] {ckpt.name}  ({n_episodes} ep, reward={reward_mode}, seed={seed}) ...")
    dqn_evaluate(
        checkpoint=str(ckpt),
        reward_mode=reward_mode,
        n_episodes=n_episodes,
        seed=seed,
        out=str(out),
        notes=f"final-report eval of {ckpt.name}",
    )
    return out


def _run_ppo_eval(ckpt: Path, reward_mode: str, n_episodes: int, seed: int,
                  agent_label: str, file_prefix: str) -> Path:
    out = RESULTS_DIR / f"{file_prefix}_{reward_mode}.json"
    print(f"\n[{agent_label}] {ckpt.name}  ({n_episodes} ep, reward={reward_mode}, seed={seed}) ...")
    result = ppo_evaluate(
        checkpoint_path=str(ckpt),
        reward_mode=reward_mode,
        n_episodes=n_episodes,
        seed=seed,
        notes=f"final-report eval of {ckpt.name}",
    )
    save_metrics(
        out, agent=agent_label, reward_mode=reward_mode,
        n_episodes=n_episodes, seed=seed,
        raw_scores=result["raw_scores"], raw_steps=result["raw_steps"],
        notes=result["notes"],
    )
    print(f"  mean_score = {result['mean_score']:.2f} +/- {result['std_score']:.2f}"
          f"  mean_steps = {result['mean_steps']:.1f}")
    print(f"  -> {out}")
    return out


def cmd_eval(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    missing: list[str] = []

    if not args.skip_random:
        written.append(_run_baseline("Random", "sparse", args.episodes, args.seed))
    if not args.skip_greedy:
        written.append(_run_baseline("Greedy", "sparse", args.episodes, args.seed))

    if not args.skip_dqn:
        ckpt = Path(args.dqn_ckpt) if args.dqn_ckpt else autoselect_dqn()
        if ckpt is None or not ckpt.exists():
            missing.append(
                "DQN: no checkpoint found. Train one with:\n"
                "       uv run python run_experiments.py train dqn --reward dense --seed 0"
            )
        else:
            written.append(_run_dqn_eval(ckpt, args.reward_mode, args.episodes, args.seed))

    if not args.skip_ppo:
        ckpt = Path(args.ppo_ckpt) if args.ppo_ckpt else autoselect_ppo_dense()
        if ckpt is None or not ckpt.exists():
            missing.append("PPO: no checkpoint found under checkpoints/.")
        else:
            written.append(_run_ppo_eval(
                ckpt, args.reward_mode, args.episodes, args.seed,
                agent_label="PPO", file_prefix="ppo",
            ))

    if not args.skip_ppo_af:
        ckpt = Path(args.ppo_af_ckpt) if args.ppo_af_ckpt else autoselect_ppo_af()
        if ckpt is None or not ckpt.exists():
            missing.append("PPO-AF: no afterstate checkpoint found under checkpoints/.")
        else:
            af_reward = _reward_from_ckpt(ckpt.name)
            written.append(_run_ppo_eval(
                ckpt, af_reward, args.episodes, args.seed,
                agent_label="PPO-AF", file_prefix="ppo_af",
            ))

    print(f"\n[done] wrote {len(written)} JSON file(s) to {RESULTS_DIR}/")
    for m in missing:
        print(f"  [skip] {m}")

    if args.no_plot:
        return

    print("\n--- aggregate ---")
    subprocess.run([sys.executable, "-m", "evaluation.aggregate"], check=True)
    print("\n--- plot ---")
    subprocess.run([sys.executable, "-m", "evaluation.plot_comparison"], check=True)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def _shell(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _train_ppo_one(reward: str, args: argparse.Namespace) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_af" if args.use_afterstate else ("_h" if args.use_heuristics else "")
    run_dir = CKPT_DIR / f"{reward}{suffix}_seed{args.seed}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[train PPO] reward={reward}  steps={args.total_steps}  seed={args.seed}")
    print(f"  ckpt-dir: {run_dir}")

    cmd = [
        sys.executable, "-m", "agents.ppo.train_ppo",
        "--reward",      reward,
        "--seed",        str(args.seed),
        "--total-steps", str(args.total_steps),
        "--n-envs",      str(args.n_envs),
        "--ckpt-dir",    str(run_dir),
    ]
    if args.use_heuristics:
        cmd.append("--use-heuristics")
    if args.use_afterstate:
        cmd.append("--use-afterstate")
    _shell(cmd)


def _train_dqn_one(reward: str, args: argparse.Namespace) -> None:
    print(f"\n[train DQN] reward={reward}  steps={args.total_steps}  seed={args.seed}")
    _shell([
        sys.executable, "-m", "agents.dqn.train_dqn",
        "--reward", reward,
        "--seed",   str(args.seed),
        "--steps",  str(args.total_steps),
    ])


def cmd_train(args: argparse.Namespace) -> None:
    rewards = ["sparse", "dense"] if args.reward == "both" else [args.reward]
    train_one = _train_ppo_one if args.agent == "ppo" else _train_dqn_one
    if args.agent == "dqn" and (args.use_heuristics or args.use_afterstate):
        sys.exit("--use-heuristics / --use-afterstate only apply to 'train ppo'")
    for r in rewards:
        train_one(r, args)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> None:
    if args.agent == "ppo":
        ckpt = Path(args.checkpoint) if args.checkpoint else autoselect_ppo_dense()
        if ckpt is None or not ckpt.exists():
            print("[demo] no PPO checkpoint found; falling back to greedy")
            _shell([sys.executable, "demo/play.py", "--agent", "greedy", "--fps", str(args.fps)])
            return
        print(f"[demo] PPO checkpoint = {ckpt}")
        _shell([
            sys.executable, "demo/play.py",
            "--agent", "ppo", "--checkpoint", str(ckpt),
            "--fps", str(args.fps),
        ])
    else:
        _shell([sys.executable, "demo/play.py", "--agent", args.agent, "--fps", str(args.fps)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--episodes", type=int, default=100,
                   help="evaluation episodes per agent (default 100)")
    p.add_argument("--seed", type=int, default=42,
                   help="evaluation seed (default 42)")
    p.add_argument("--reward-mode", choices=["sparse", "dense"], default="dense",
                   help="reward mode for DQN/PPO eval; afterstate uses its training reward")

    p.add_argument("--ppo-ckpt",    type=str, default=None,
                   help="PPO checkpoint (auto: latest in checkpoints/dense_seed0_*/)")
    p.add_argument("--ppo-af-ckpt", type=str, default=None,
                   help="PPO-AF checkpoint (auto: latest ppo_*_af_seed*_step*.pt)")
    p.add_argument("--dqn-ckpt",    type=str, default=None,
                   help="DQN checkpoint (auto: latest dqn_*_seed*_step*.pt)")

    p.add_argument("--skip-random", action="store_true")
    p.add_argument("--skip-greedy", action="store_true")
    p.add_argument("--skip-dqn",    action="store_true")
    p.add_argument("--skip-ppo",    action="store_true")
    p.add_argument("--skip-ppo-af", action="store_true")
    p.add_argument("--no-plot",     action="store_true",
                   help="skip aggregate + bar-chart at end")


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("agent", choices=["ppo", "dqn"], help="which agent to train")
    p.add_argument("--reward",      choices=["sparse", "dense", "both"], default="dense",
                   help="'both' runs sparse then dense back-to-back")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-envs",      type=int, default=8,
                   help="PPO only; DQN uses its in-script N_ENVS")
    p.add_argument("--use-heuristics", action="store_true",
                   help="PPO only: feed 40-d heuristic features into the actor-critic")
    p.add_argument("--use-afterstate", action="store_true",
                   help="PPO only: score actions by their post-placement features")


def _add_demo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--agent", default="ppo", choices=["random", "greedy", "ppo"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="PPO checkpoint (auto-detected if omitted)")
    p.add_argument("--fps", type=float, default=10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_eval_args(sub.add_parser("eval",  help="evaluate agents + plot comparison"))
    _add_train_args(sub.add_parser("train", help="train PPO or DQN"))
    _add_demo_args(sub.add_parser("demo",  help="open pygame visualizer"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
