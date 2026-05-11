"""
跨 agent 統一指標 JSON schema.

所有 agent (Random / Greedy / DQN / PPO) 跑完評估後,
都呼叫 save_metrics(...) 把結果寫成同一格式的 JSON,
組員 E 的圖表腳本 (plot_comparison.py) 會自動讀全部 JSON.

JSON 範例:
{
  "agent": "DQN",
  "reward_mode": "sparse",
  "n_episodes": 100,
  "seed": 42,
  "mean_score": 32.5,
  "std_score": 8.1,
  "mean_steps": 45.2,
  "std_steps": 6.4,
  "raw_scores": [28, 35, 31, ...],
  "raw_steps":  [40, 50, 45, ...],
  "timestamp":  "2026-05-10T14:30:00",
  "notes":      "DQN 500k steps, lr=1e-4"
}
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

VALID_AGENTS = ("Random", "Greedy", "DQN", "PPO")
VALID_REWARD_MODES = ("sparse", "dense")
REQUIRED_FIELDS = (
    "agent", "reward_mode", "n_episodes", "seed",
    "mean_score", "std_score", "mean_steps", "std_steps",
    "raw_scores", "raw_steps", "timestamp",
)


def save_metrics(
    path: str | Path,
    *,
    agent: str,
    reward_mode: str,
    n_episodes: int,
    seed: int,
    raw_scores: Iterable[float],
    raw_steps: Iterable[float],
    notes: str = "",
) -> dict:
    """
    把單一 agent 的評估結果寫成 JSON.
    mean / std 由 raw_scores / raw_steps 自動計算 (避免上游忘了算).
    回傳寫出的 dict.
    """
    if agent not in VALID_AGENTS:
        raise ValueError(f"agent 必須是 {VALID_AGENTS} 之一, got {agent!r}")
    if reward_mode not in VALID_REWARD_MODES:
        raise ValueError(f"reward_mode 必須是 {VALID_REWARD_MODES} 之一, got {reward_mode!r}")

    raw_scores = [float(s) for s in raw_scores]
    raw_steps = [float(s) for s in raw_steps]
    if len(raw_scores) != n_episodes or len(raw_steps) != n_episodes:
        raise ValueError(
            f"raw_scores/raw_steps 長度 ({len(raw_scores)}, {len(raw_steps)}) "
            f"與 n_episodes ({n_episodes}) 不符"
        )

    import statistics
    payload = {
        "agent": agent,
        "reward_mode": reward_mode,
        "n_episodes": n_episodes,
        "seed": seed,
        "mean_score": statistics.fmean(raw_scores),
        "std_score": statistics.pstdev(raw_scores),
        "mean_steps": statistics.fmean(raw_steps),
        "std_steps": statistics.pstdev(raw_steps),
        "raw_scores": raw_scores,
        "raw_steps": raw_steps,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def load_metrics(path: str | Path) -> dict:
    """讀 JSON 並驗證欄位齊全."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_FIELDS if k not in data]
    if missing:
        raise ValueError(f"{path} 缺少欄位: {missing}")
    return data
