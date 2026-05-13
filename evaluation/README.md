# Evaluation 收件規格（給組員 B / C）

組員 E 在這個資料夾蒐集所有 agent（Random / Greedy / DQN / PPO）的評估結果，產出統一的比較圖表與 summary CSV 給期末報告用。

**B（DQN）跟 C（PPO）負責的事**：訓練完模型後，跑自己的 evaluate 程式，把結果存成符合下面 schema 的 JSON 檔，丟到專案根目錄的 `results/` 資料夾。**就這樣，不需要 import 這裡的任何 module。**

> C 同學的 `agents/ppo/evaluate.py` 已經對齊這份 schema，可以當範例參考。

---

## 1. 要交什麼

**一個 JSON 檔**，放在：

```
results/{agent_prefix}_{reward_mode}.json
```

範例檔名：

- `results/dqn_sparse.json`
- `results/dqn_dense.json`
- `results/ppo_sparse.json`
- `results/ppo_dense.json`

兩種 reward_mode 都跑（sparse + dense）會比較完整，但如果時間有限至少先交其中一種。

---

## 2. JSON Schema（12 個欄位，缺一不可）

| 欄位 | 型別 | 說明 | 範例 |
|---|---|---|---|
| `agent` | string | **大小寫敏感**，只能是 `"Random"` / `"Greedy"` / `"DQN"` / `"PPO"` | `"DQN"` |
| `reward_mode` | string | 只能是 `"sparse"` 或 `"dense"` | `"dense"` |
| `n_episodes` | int | 跑了幾局 | `100` |
| `seed` | int | 起始 seed | `42` |
| `mean_score` | float | `raw_scores` 的平均 | `32.5` |
| `std_score` | float | `raw_scores` 的母體標準差（`statistics.pstdev`） | `8.1` |
| `mean_steps` | float | `raw_steps` 的平均 | `45.2` |
| `std_steps` | float | `raw_steps` 的母體標準差 | `6.4` |
| `raw_scores` | list[number] | 每局 final score，長度必須等於 `n_episodes` | `[28, 35, 31, ...]` |
| `raw_steps` | list[number] | 每局 step 數，長度必須等於 `n_episodes` | `[40, 50, 45, ...]` |
| `timestamp` | string | ISO 8601，例 `datetime.now().isoformat(timespec="seconds")` | `"2026-05-13T14:30:00"` |
| `notes` | string | 自由欄位，建議寫超參數摘要 | `"DQN 500k steps, lr=1e-4"` |

### 完整 JSON 範例

```json
{
  "agent": "DQN",
  "reward_mode": "sparse",
  "n_episodes": 100,
  "seed": 42,
  "mean_score": 32.5,
  "std_score": 8.1,
  "mean_steps": 45.2,
  "std_steps": 6.4,
  "raw_scores": [28, 35, 31, "..."],
  "raw_steps":  [40, 50, 45, "..."],
  "timestamp":  "2026-05-13T14:30:00",
  "notes":      "DQN 500k steps, lr=1e-4"
}
```

（schema 權威來源在 `evaluation/metrics_schema.py`，要修改欄位請先跟 E 講。）

---

## 3. 評估時要對齊的參數

為了公平比較，請所有 agent 用相同的評估設定：

| 參數 | 值 | 理由 |
|---|---|---|
| `--episodes` (n_episodes) | `100` | 跟 baseline 一致 |
| `--seed` | `42` | 跟 baseline 一致；用這個 seed 衍生 100 個 per-episode seed |
| 政策 | **deterministic**（argmax） | 報告的是「模型最佳實力」，不是 stochastic 採樣平均 |
| reward_mode | `sparse` 跟 `dense` 分開各跑一個 JSON | 兩種 reward 是不同的實驗條件 |

PPO 同學的 `agents/ppo/evaluate.py` 預設值就是這組，可參考實作方式（`rng = np.random.default_rng(seed)`，每局 `env.reset(seed=int(rng.integers(1 << 31)))`）。

---

## 4. 交件前自我檢查

跑這行驗證你的 JSON 欄位齊不齊、agent name 合不合法：

```bash
uv run python -c "from evaluation.metrics_schema import load_metrics; load_metrics('results/dqn_sparse.json'); print('OK')"
```

如果欄位漏掉或 agent name 拼錯，會直接 `ValueError`。看到 `OK` 才算過關。

---

## 5. 常見錯誤

| 錯誤 | 症狀 | 解法 |
|---|---|---|
| `agent` 拼成 `"dqn"` / `"Dqn"` | `ValueError: agent 必須是 ('Random', 'Greedy', 'DQN', 'PPO') 之一` | 改成大寫 `"DQN"` |
| `reward_mode` 拼成 `"Sparse"` 或 `"dense_v2"` | 同上 ValueError | 只能小寫 `"sparse"` 或 `"dense"` |
| `raw_scores` 長度不等於 `n_episodes` | 之後畫圖會出怪數據 | 確認跑滿 100 局 |
| 缺欄位（少 `timestamp` 或 `notes`） | `load_metrics` 報 `缺少欄位` | 補上即可，`notes` 可以是空字串 |
| `std_score` 用樣本標準差（`numpy.std(..., ddof=1)`） | mean ± std 圖會跟其他人不一致 | 用母體標準差：`numpy.std(...)` 預設或 `statistics.pstdev` |
| 檔名放錯位置（放到 `evaluation/results/`） | E 的 `aggregate.py` 抓不到 | 必須在**專案根目錄**的 `results/` |

---

## 6. 交件後 E 會做的事

1. `uv run python -m evaluation.aggregate` → 產出 `results/summary.csv`
2. `uv run python -m evaluation.plot_comparison` → 產出 `report/figures/score_comparison.png`
3. 把數字 + 圖貼進 `report/outline.md`

所以你交完 JSON 就沒事了，剩下 E 處理。
