"""
把 results/*.json 全部讀進來, 輸出 results/summary.csv,
方便寫報告時直接貼表.

用法:
    python -m evaluation.aggregate
    python -m evaluation.aggregate --results-dir results/ --out results/summary.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.metrics_schema import VALID_AGENTS, VALID_REWARD_MODES, load_metrics

COLUMNS = (
    "agent", "reward_mode", "n_episodes", "seed",
    "mean_score", "std_score", "mean_steps", "std_steps",
    "timestamp", "notes",
)


def main():
    parser = argparse.ArgumentParser(description="把所有 metrics JSON 彙整成 CSV")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("*.json")):
        try:
            m = load_metrics(path)
        except (ValueError, OSError) as e:
            print(f"  [skip] {path.name}: {e}")
            continue
        rows.append({k: m.get(k, "") for k in COLUMNS})

    # 排序: 先按 agent (依 VALID_AGENTS 順序), 再按 reward_mode
    agent_rank = {a: i for i, a in enumerate(VALID_AGENTS)}
    reward_rank = {r: i for i, r in enumerate(VALID_REWARD_MODES)}
    rows.sort(key=lambda r: (agent_rank.get(r["agent"], 999),
                             reward_rank.get(r["reward_mode"], 999)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"寫入 {len(rows)} 筆 -> {args.out}")
    for r in rows:
        print(f"  {r['agent']:<8} {r['reward_mode']:<7} "
              f"score={r['mean_score']:.2f}±{r['std_score']:.2f} "
              f"steps={r['mean_steps']:.2f}±{r['std_steps']:.2f}")


if __name__ == "__main__":
    main()
