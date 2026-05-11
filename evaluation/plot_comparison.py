"""
讀 results/*.json 把所有 agent 的指標畫成 bar chart.

產生兩張圖:
  - comparison_score.png : mean_score (含 std error bar)
  - comparison_steps.png : mean_steps (含 std error bar)

如果同一 agent 有 sparse 和 dense 兩個檔案, 會自動畫成 grouped bar.

用法:
    python -m evaluation.plot_comparison
    python -m evaluation.plot_comparison --results-dir results/ --out-dir results/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics_schema import VALID_AGENTS, VALID_REWARD_MODES, load_metrics


REWARD_COLORS = {"sparse": "#4C72B0", "dense": "#DD8452"}


def _collect(results_dir: Path) -> dict[tuple[str, str], dict]:
    """回傳 {(agent, reward_mode): metrics_dict}, 跳過格式不符的檔案."""
    data: dict[tuple[str, str], dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            m = load_metrics(path)
        except (ValueError, OSError) as e:
            print(f"  [skip] {path.name}: {e}")
            continue
        key = (m["agent"], m["reward_mode"])
        if key in data:
            print(f"  [warn] 重複 agent/reward 組合: {key}, 後者覆蓋前者 ({path.name})")
        data[key] = m
    return data


def _plot_metric(
    data: dict[tuple[str, str], dict],
    *,
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    agents_present = [a for a in VALID_AGENTS if any(a == k[0] for k in data)]
    rewards_present = [r for r in VALID_REWARD_MODES if any(r == k[1] for k in data)]
    if not agents_present:
        print(f"  [skip] {out_path.name}: 沒有任何 agent 資料")
        return

    n_agents = len(agents_present)
    n_rewards = len(rewards_present)
    bar_w = 0.8 / n_rewards
    x = np.arange(n_agents)

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * n_agents), 4.5))
    for i, reward in enumerate(rewards_present):
        means = [data.get((a, reward), {}).get(mean_key, np.nan) for a in agents_present]
        stds = [data.get((a, reward), {}).get(std_key, 0.0) for a in agents_present]
        offset = (i - (n_rewards - 1) / 2) * bar_w
        ax.bar(
            x + offset, means, bar_w,
            yerr=stds, capsize=4,
            label=f"{reward} reward",
            color=REWARD_COLORS.get(reward, None),
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(agents_present)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if n_rewards > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="畫跨 agent 比較 bar chart")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"results-dir 不存在: {args.results_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = _collect(args.results_dir)
    if not data:
        raise SystemExit(f"{args.results_dir} 內找不到合法的 metrics JSON")

    print(f"\n讀到 {len(data)} 筆 metrics:")
    for (agent, reward), m in data.items():
        print(f"  {agent:<8} {reward:<7} mean_score={m['mean_score']:.2f} mean_steps={m['mean_steps']:.2f}")

    print()
    _plot_metric(
        data,
        mean_key="mean_score", std_key="std_score",
        ylabel="Mean episode score (lines cleared)",
        title="Block Blast — Score Comparison",
        out_path=args.out_dir / "comparison_score.png",
    )
    _plot_metric(
        data,
        mean_key="mean_steps", std_key="std_steps",
        ylabel="Mean survival steps",
        title="Block Blast — Survival Comparison",
        out_path=args.out_dir / "comparison_steps.png",
    )


if __name__ == "__main__":
    main()
