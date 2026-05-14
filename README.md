# Block Blast RL — 開發手冊

> 給組員 B、C、D、E 的環境使用說明。  
> 有問題先看這份，再問組員 A。

---

## Git 協作規則

```bash
# 各自開自己的 branch，不要直接在 main 上開發
git checkout -b feature/dqn        # 組員 B
git checkout -b feature/ppo        # 組員 C
git checkout -b feature/reward     # 組員 D
git checkout -b feature/baseline   # 組員 E
```

**禁止修改 `env/` 目錄下的任何檔案。**  
有 bug 或新需求請告訴組員 A，由 A 統一修改並更新 main。

每次開始工作前先 pull：
```bash
git pull origin main
```

開發完成後確認測試通過再 push：
```bash
uv run python test_env.py   # 必須全部 PASSED
git push origin feature/你的branch名稱
```

---

## 1. 安裝

```bash
# uv 管理依賴（pyproject.toml 為主，uv.lock 鎖版本）
uv sync

# 若有 NVIDIA GPU（可選，一次性設定）
uv add torch --index https://download.pytorch.org/whl/cu124 --reinstall
uv run python -c "import torch; print(torch.cuda.is_available())"   # 應印 True
```

執行時 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 自動判斷，有 GPU 就用，不用手動切。CPU 跑 500k steps 約 25 分鐘。

確認可以 import：

```python
from env import BlockBlastEnv
env = BlockBlastEnv()
obs, info = env.reset()
print(obs["board"].shape)        # (8, 8)
print(obs["pieces_left"].shape)  # (3,)
```

---

## 2. 環境基本規則

| 項目 | 說明 |
|------|------|
| 盤面 | 8×8 格，0 = 空，1 = 填滿 |
| 每回合 | 拿到 3 個方塊，依序各放 1 次，放完再拿新的 3 個 |
| 得分 | 消除一行或一列 +1 分，行列同時消除各自計算 |
| 死局 | 剩餘方塊都無法放置 → episode 結束 |
| 方塊庫 | 35 種形狀（含所有旋轉），從 `env/shapes.py` 查看 |

---

## 3. Observation Space

`reset()` 和 `step()` 都回傳一個 `dict`：

```python
obs = {
    "board":       np.ndarray,  # shape (8, 8),  float32,  0.0 或 1.0
    "pieces":      np.ndarray,  # shape (3, 5, 5), float32
                                # 三個方塊各自渲染成 5×5 grid
                                # 已使用的 slot 全為 0
    "pieces_left": np.ndarray,  # shape (3,), float32, binary mask
                                # 1.0 = 該 slot 還有方塊, 0.0 = 已使用
}
```

**pieces 範例：** 形狀 J-0（`[(0,1),(1,1),(2,0),(2,1)]`）對應的 5×5 grid：

```
· █ · · ·
· █ · · ·
█ █ · · ·
· · · · ·
· · · · ·
```

---

## 4. Action Space

```
Discrete(192)
action = piece_idx * 64 + row * 8 + col
```

| 欄位 | 範圍 |
|------|------|
| `piece_idx` | 0、1、2（對應三個方塊 slot） |
| `row` | 0–7（方塊左上角的列） |
| `col` | 0–7（方塊左上角的行） |

解碼：

```python
from env import _decode_action
piece_idx, row, col = _decode_action(action)
```

編碼：

```python
from env import encode_action
action = encode_action(piece_idx=0, row=3, col=5)
```

---

## 5. Action Mask（最重要）

非法動作（超出邊界、格子已填、slot 已用完）必須過濾，不然 agent 會學壞。

### 取得 mask

```python
# 從 info dict 取（reset 和 step 都有）
mask = info["action_mask"]   # shape (192,), dtype bool

# 或直接呼叫 method（內容相同）
mask = env.action_masks()
```

### 組員 B — DQN 用法

```python
q_values = model(obs)
q_values[~mask] = float("-inf")   # 非法動作設為 -inf
action = q_values.argmax()
```

### 組員 C — 自製 PPO 用法

```python
logits, value = model(board, pieces, pieces_left)
logits = logits.masked_fill(~mask_tensor, float("-inf"))
dist = Categorical(logits=logits)
action = dist.sample()
```

---

## 6. Reward

```python
env = BlockBlastEnv(reward_mode="sparse")   # 預設
env = BlockBlastEnv(reward_mode="dense")    # 密集版
```

| 模式 | 公式 |
|------|------|
| sparse | `lines² × (1 + 0.2 × combo_streak)`，無消除時 `0`，死局 `−10` |
| dense | sparse `+ HOLE_PENALTY × holes + BUMPINESS_PENALTY × bumpiness` |

**Combo streak**: 連續消除時 `combo_streak` 每次 +1（放方塊但未消除則歸零）。例如連續三次消除各 1 行：

| 步驟 | streak | reward |
|------|--------|--------|
| 1 | 1 | 1² × 1.2 = 1.2 |
| 2 | 2 | 1² × 1.4 = 1.4 |
| 3 | 3 | 1² × 1.6 = 1.6 |

注意：**score（得分顯示）仍用 `lines` 計算**，reward 才用平方 + streak 乘數。

常數定義在 [reward_functions.py](reward_functions.py)，env 的 `_dense_shaping()` 已自動讀取。改係數只要動那個檔案即可，**不用碰 env**。

**目前係數**: `HOLE_PENALTY = -0.02`, `BUMPINESS_PENALTY = -0.01`, `COMBO_STREAK_BONUS = 0.2`。

> 📜 **歷程注記**: proposal §3.1 原案是 `-0.1 / -0.05`,但 v1 實際用了 `-0.3 / -0.1` 訓練,結果 PPO 的 critic `value_loss` 暴增到 ~150、policy 完全沒學起來。組員 C 診斷後降到 `-0.02 / -0.01`,PPO dense 分數從 1.89 跳到 4.22(+123%)。完整實驗紀錄見 [docs/ppo_journey.md](docs/ppo_journey.md)。

---

## 7. 標準訓練迴圈（不用 SB3）

```python
import numpy as np
from env import BlockBlastEnv

env = BlockBlastEnv(reward_mode="sparse")
obs, info = env.reset(seed=42)

for step in range(500_000):
    mask = info["action_mask"]

    # ← 在這裡換成你的 agent
    action = int(np.random.choice(np.where(mask)[0]))

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

---

## 8. PPO 訓練

### 方式 A：PowerShell 一鍵腳本（推薦 Windows）

```powershell
# 預設：dense reward，500k steps，train + eval
.\run_ppo.ps1

# 指定 reward / steps
.\run_ppo.ps1 -Mode train -Reward sparse
.\run_ppo.ps1 -Mode train -Reward both     # sparse + dense 都跑
.\run_ppo.ps1 -Mode all   -TotalSteps 1000000

# 只做 eval（自動找最新 checkpoint）
.\run_ppo.ps1 -Mode eval  -Reward dense

# 開 pygame demo
.\run_ppo.ps1 -Mode demo
```

### 方式 B：直接 uv 指令

```bash
# 稀疏 reward
uv run python -m agents.ppo.train_ppo --reward sparse --seed 0

# 密集 reward
uv run python -m agents.ppo.train_ppo --reward dense  --seed 0
```

### 超參數（可用 CLI flag 覆寫）

| 參數 | 值 | 備註 |
|------|----|----|
| `--total-steps`   | 500_000 | proposal 規定 |
| `--n-envs`        | 8 | 平行 env 數，SubprocVecEnv |
| `--n-steps`       | 128 | 每個 env 每次 rollout 步數 → 一次 update 用 1024 transitions |
| `--n-epochs`      | 10 | 每次 rollout 重複跑 10 個 epoch |
| `--batch-size`    | 64 | mini-batch |
| `--lr`            | 3e-4 | |
| `--gamma`         | 0.99 | |
| `--gae-lambda`    | 0.95 | |
| `--clip-range`    | 0.2 | PPO ratio 截斷範圍 |
| `--ent-coef`      | 0.01 | 太小會 policy collapse，proposal 也提醒 |
| `--vf-coef`       | 0.5 | value loss 權重 |
| `--max-grad-norm` | 0.5 | gradient clipping |
| `--ckpt-every`    | 50_000 | checkpoint 間隔（env steps） |

> 預設值刻意對齊 SB3 MaskablePPO 預設，理由：「PPO 沒調好」這種質疑可以擋掉一輪。

訓練輸出：
- `checkpoints/ppo_<reward>_seed<S>_step<N>.pt` — 每 50k steps 一次
- `runs/ppo_<reward>_seed<S>/` — TensorBoard event files

---

## 9. 監看訓練曲線

訓練期間另開 terminal：

```bash
uv run tensorboard --logdir runs
```

瀏覽器開 http://localhost:6006，要看的指標：

| 指標 | 應該長怎樣 |
|------|----------|
| `rollout/ep_score_mean` | **必須上升**。沒上升 = agent 沒在學，先檢查 ent_coef、lr |
| `rollout/ep_length_mean` | 通常跟 score 同步上升 |
| `train/entropy` | 緩降。若 1-2 萬 step 內就掉到接近 0 → policy collapse，把 ent_coef 從 0.01 提到 0.02-0.05 |
| `train/approx_kl` | 應在 0.005-0.02 之間。> 0.05 表示 step 太大，降低 lr |
| `train/clip_fraction` | 0.1-0.3 之間健康。長期 > 0.4 表 ratio 常被截斷，降 lr 或加 ent_coef |
| `train/explained_var` | 從 0 慢慢往 1 升。卡在 0 附近 = critic 學不起來 |

---

## 10. 產生評估 JSON（交給組員 E）

實測 sparse 在 ~700k 就 plateau，dense 在 1M 仍有上升；預設 500k 起跳，要更高分用 `--total-steps 1000000`。

```bash
uv run python -m agents.ppo.evaluate \
    --checkpoint checkpoints/ppo_sparse_seed0_step1000000.pt \
    --reward sparse --episodes 100 --seed 42 \
    --out results/ppo_sparse.json \
    --notes "PPO 1M steps, lr=3e-4, ent_coef=0.01, shared backbone with DQN"

uv run python -m agents.ppo.evaluate \
    --checkpoint checkpoints/ppo_dense_seed0_step1000000.pt \
    --reward dense --episodes 100 --seed 42 \
    --out results/ppo_dense.json \
    --notes "PPO 1M steps, dense reward (HOLE_PENALTY=-0.02, BUMPINESS_PENALTY=-0.01)"
```

產生 `results/ppo_sparse.json` 和 `results/ppo_dense.json`，**這兩個檔要 commit**，組員 E 會從 git 收集所有人的 JSON 畫對比圖。

JSON schema（11 欄，組員 E 訂）：

```json
{
  "agent": "PPO",
  "reward_mode": "sparse",
  "n_episodes": 100,
  "seed": 42,
  "mean_score": 32.5,
  "std_score": 8.1,
  "mean_steps": 45.2,
  "std_steps": 6.4,
  "raw_scores": [28, 35, "..."],
  "raw_steps":  [40, 50, "..."],
  "timestamp": "2026-05-11T14:30:00",
  "notes": "PPO 1M steps, ..."
}
```

---

## 11. 各組員對接說明

### 組員 B — DQN

**共用網路（必須用 `agents/network.py`，確保與 C 架構一致）：**

```python
import torch
from agents.network import BlockBlastNet, obs_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net    = BlockBlastNet(output_dim=192).to(device)

# obs_to_tensor 回傳 3-tuple（board, pieces, pieces_left）
board, pieces, pieces_left = obs_to_tensor(obs, device)
q_values = net(board, pieces, pieces_left)   # (1, 192)

mask_t = torch.tensor(mask, device=device)
q_values[~mask_t] = float("-inf")
action = int(q_values.argmax(dim=1).item())
```

**replay buffer 要一起存 mask：**

```python
buffer.push(obs, action, reward, next_obs, terminated,
            action_mask=mask, next_action_mask=next_mask)
```

取樣後，`next_mask` 用來 mask 掉 target Q 值裡的非法動作（Double DQN 標準做法）。

---

### 組員 C — PPO

採用**自製 PPO**，使用 `agents.network.BlockBlastActorCritic` 共用 backbone，與組員 B 的 DQN 完全相同網路結構，確保 RQ1 對比公平。

**`agents/ppo/` 檔案架構：**

| 檔案 | 角色 | 關鍵設計 |
|------|------|----------|
| `rollout_buffer.py` | 收集 rollout、算 advantage | **每筆 transition 連 action_mask 一起存**，update 時要重新 mask logits |
| `ppo_agent.py` | PPO 演算法本身 | `logits.masked_fill(~mask, -inf)` 後接 `Categorical`，非法動作機率 = 0 |
| `train_ppo.py` | 8 個 SubprocVecEnv 平行收集 | 每步從 `info["action_mask"]` 拿 mask；env 剛 reset 時要 `env_method("action_masks")` 重抓 |
| `evaluate.py` | 最終評估 | **deterministic = argmax(masked logits)**，不是隨機 sample |

開發歷程與後續方案：
- **[docs/ppo_journey.md](docs/ppo_journey.md)** — 兩輪訓練 + 關鍵 reward 修復的完整紀錄（報告 PPO 章節寫作素材）
- **[docs/improvement_options.md](docs/improvement_options.md)** — 突破當前分數的三個方案（BC warm start / 大網路 / 加 hand-crafted features），含影響範圍與工程量比較

---

### 組員 D — Reward 設計

✅ **`reward_functions.py` → `env._dense_shaping()` 的 wiring 已經接好**（2026-05 完成），改係數**只動 `reward_functions.py` 那一個檔**，env 會自動讀取：

```python
# reward_functions.py（D 動這裡就好）
HOLE_PENALTY      = -0.02   # 目前值
BUMPINESS_PENALTY = -0.01   # 目前值
COMBO_STREAK_BONUS = 0.2    # 連續消除加成係數
```

跑比較實驗：

```python
env_sparse = BlockBlastEnv(reward_mode="sparse")
env_dense  = BlockBlastEnv(reward_mode="dense")
```

---

### 組員 E — Baseline 評估

```python
from env import BlockBlastEnv
from agents.random_agent import evaluate as random_eval
from agents.greedy_agent import evaluate as greedy_eval

env = BlockBlastEnv()
print("Random:", random_eval(env, n_episodes=100))
print("Greedy:", greedy_eval(env, n_episodes=100))
# 回傳 mean_score, std_score, mean_steps, std_steps
```

---

## 12. 方塊形狀查詢

```python
from env.shapes import SHAPES, SHAPE_NAMES, N_SHAPES, print_shape

print(f"共 {N_SHAPES} 種形狀")   # 35

for i, (shape, name) in enumerate(zip(SHAPES, SHAPE_NAMES)):
    print_shape(shape, f"#{i} {name}")
    print()
```

---

## 13. 快速除錯

```python
# 看盤面狀態
env.render()   # render_mode="human" 時印到 terminal

# 手動設定盤面（測試 line clear）
env.board[7, :] = 1.0        # 填滿第 7 列
env.piece_shape_ids[0] = 0   # 強制第一個方塊為 dot

# 確認 mask 正確
mask = env.action_masks()
print(f"合法動作數：{mask.sum()}")
```

---

## 14. CNN 網路架構（`agents/network.py`）

B 和 C 共用同一份網路，確保對比公平。

```
board  (1,8,8)  → Conv(1→32,3×3)→Conv(32→64,3×3) → (64,8,8) ──┐
                                                                  cat(112,8,8) → Conv(112→64,3×3) → Flatten(4096) → FC(128) ──┐
piece_0 (1,5,5) ─┐                                               │                                                             │ cat(131) → FC(128) → FC(128) → output
piece_1 (1,5,5) ─┼─ weight-shared Conv(1→16,3×3) → pad→(16,8,8)─┘                                                             │
piece_2 (1,5,5) ─┘  → (48,8,8) total                                                         pieces_left (3,) ───────────────┘
```

**關鍵設計：Spatial Fusion**（v2 起）：三個 piece 分支 padding 到 8×8 後，與 board feature map 在 channel 維度拼接，讓 conv 在同一空間解析度上比對盤面與方塊形狀，再壓縮成向量。`pieces_left (3,)` 在最後拼接，給模型「還剩幾個方塊」的顯式信號。

| 類別 | 用途 | 輸出 |
|------|------|------|
| `BlockBlastNet` | DQN Q 網路 | `(B, 192)` Q 值 |
| `BlockBlastActorCritic` | 自製 PPO | `(B, 192)` logits + `(B, 1)` value |
| `obs_to_tensor(obs, device)` | numpy obs → tensor | `(board, pieces, pieces_left)` 3-tuple |

```python
# 正確呼叫方式（三個輸入都要傳）
board, pieces, pieces_left = obs_to_tensor(obs, device)
output = net(board, pieces, pieces_left)
```

---

## 15. Pygame Demo 視覺化

安裝 pygame（若尚未安裝）：

```bash
uv add "pygame>=2.6"
```

### 基本執行

```bash
# greedy agent，每秒 4 步（預設）
uv run python demo/play.py

# random agent
uv run python demo/play.py --agent random

# ★ 跑訓練好的 PPO checkpoint
uv run python demo/play.py --agent ppo --checkpoint checkpoints/ppo_sparse_seed0_step500000.pt

# 慢速，每秒 1 步
uv run python demo/play.py --fps 1

# 手動模式：按 SPACE 逐步，ESC 離開
uv run python demo/play.py --fps 0

# 跑完 N 個 episode 後自動結束
uv run python demo/play.py --episodes 5
```

也可以用 `run_ppo.ps1 -Mode demo` 自動找最新 checkpoint 開啟。

PPO 模式採 **deterministic argmax**（masked logits），跟 `evaluate.py` 產 JSON 的決策方式一致。

### 跨機器跑 demo（在 server 訓練，在筆電播放）

1. 在 server / GPU 機器上跑完訓練，產出 `checkpoints/ppo_*.pt`
2. 把 `.pt` 檔（約 7 MB）拷到筆電（`scp` / 雲端 / USB 都行）
3. 筆電上 `uv sync` 裝好 torch + pygame，跑上面的 PPO 指令

`torch.load(map_location=...)` 會自動處理 GPU→CPU 轉換，不用管。

### 畫面說明

| 區域 | 說明 |
|------|------|
| 左側棋盤 | 8×8 盤面，藍色 = 填滿，消除時閃黃光 |
| 右側面板 | 當前 Score、步數、Episode 編號 |
| 方塊預覽 | 三個待放方塊，各自用不同顏色標示 |

### 串接自己的 Agent（例如組員 B 的 DQN）

`demo/play.py` 使用鴨子型別（duck typing）—— 任何有 `select_action(obs, mask)` 方法的物件都可以直接串。加到 `build_agent()` 即可：

```python
def build_agent(name: str, env, checkpoint: str = None):
    if name == "random":
        return RandomAgent(env)
    if name == "ppo":
        return PPODemoAgent(env, checkpoint)
    if name == "dqn":
        # 組員 B 寫好 DQN 後加這段
        from agents.dqn.dqn_agent import DQNDemoAgent
        return DQNDemoAgent(env, checkpoint)
    return GreedyAgent(env)
```

Agent 只需實作一個方法：

```python
class MyAgent:
    def select_action(self, obs: dict, action_mask: np.ndarray) -> int:
        # obs         : {"board": (8,8), "pieces": (3,5,5), "pieces_left": (3,)}
        # action_mask : (192,) bool，True = 合法動作
        # return      : int，合法 action index
        ...
```

---

## 16. 檔案結構

```
BlockBlastWithRL/
├── env/                          # 組員 A 維護，禁止其他人修改
│   ├── block_blast_env.py        # 已接上 reward_functions 的 dense shaping + combo streak
│   ├── shapes.py
│   └── __init__.py
├── agents/
│   ├── network.py                # 共用 CNN backbone（空間融合版），B 和 C 都從這裡 import
│   ├── random_agent.py           # 組員 E
│   ├── greedy_agent.py           # 組員 E
│   ├── dqn/                      # 組員 B 的地盤
│   │   ├── train_dqn.py          # 訓練入口：uv run python -m agents.dqn.train_dqn
│   │   ├── dqn_agent.py          # (B 自行新增)
│   │   ├── replay_buffer.py      # (B 自行新增)
│   │   └── __init__.py
│   └── ppo/                      # 組員 C 的地盤
│       ├── train_ppo.py          # 訓練入口：uv run python -m agents.ppo.train_ppo
│       ├── ppo_agent.py          # PPO 演算法核心
│       ├── rollout_buffer.py     # GAE-λ buffer，存 action_mask + pieces_left
│       ├── evaluate.py           # 載入 ckpt 跑 N 集 → JSON（E 的 schema）
│       └── __init__.py
├── demo/
│   └── play.py                   # Pygame 視覺化 demo（random / greedy / ppo）
├── docs/                         # 開發歷程 + 討論文件
│   ├── ppo_journey.md            # 組員 C 的 PPO 開發紀錄（報告素材）
│   └── improvement_options.md    # 進一步突破分數的方案討論
├── results/                      # 各人交給組員 E 的 JSON（commit 進 git）
├── reward_functions.py           # 組員 D 修改（env 自動讀取）
├── run_ppo.ps1                   # Windows PowerShell 一鍵訓練/評估/demo 腳本
├── test_env.py                   # 環境驗證，push 前必跑
├── README.md                     # 本文件
├── CLAUDE.md                     # 給 Claude Code 用的 repo 導覽
├── pyproject.toml + uv.lock      # uv 主依賴管理
└── .python-version               # 3.13
```

訓練入口用 module 方式執行（在專案根目錄下）：

```bash
uv run python -m agents.dqn.train_dqn
uv run python -m agents.ppo.train_ppo
```
