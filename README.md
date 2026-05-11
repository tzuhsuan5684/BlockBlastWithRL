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
python test_env.py   # 必須全部 PASSED
git push origin feature/你的branch名稱
```

---

## 1. 安裝

```bash
python -m pip install gymnasium numpy matplotlib pillow stable-baselines3 sb3-contrib torch
```

確認可以 import：

```python
from env import BlockBlastEnv
env = BlockBlastEnv()
obs, info = env.reset()
print(obs["board"].shape)   # (8, 8)
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
    "board":  np.ndarray,  # shape (8, 8),  float32,  0.0 或 1.0
    "pieces": np.ndarray,  # shape (3, 5, 5), float32
                           # 三個方塊各自渲染成 5×5 grid
                           # 已使用的 slot 全為 0
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

### 組員 C — MaskablePPO 用法

SB3-contrib 會自動呼叫 `env.action_masks()`，不需要手動傳。

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

---

## 6. Reward

```python
env = BlockBlastEnv(reward_mode="sparse")   # 預設
env = BlockBlastEnv(reward_mode="dense")    # 組員 D 的密集版
```

| 模式 | 公式 |
|------|------|
| sparse | `+n_lines_cleared`，死局 `−10` |
| dense | sparse `− 0.1×holes − 0.05×bumpiness` |

常數定義在 `reward_functions.py`，組員 D 調整係數請改那個檔案，然後在 `block_blast_env.py` 的 `_dense_shaping()` 引用。

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

## 8. 各組員對接說明

### 組員 B — DQN

**共用網路（必須用 `agents/network.py`，確保與 C 架構一致）：**

```python
import torch
from agents.network import BlockBlastNet, obs_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net    = BlockBlastNet(output_dim=192).to(device)

# 每個 step 的動作選擇
board, pieces = obs_to_tensor(obs, device)
q_values = net(board, pieces)                    # (1, 192)

mask_t = torch.tensor(mask, device=device)
q_values[~mask_t] = float("-inf")               # 非法動作設為 -inf
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

採用 **方案二:自製 PPO**,使用 `agents.network.BlockBlastActorCritic` 共用 backbone,與組員 B 的 DQN 完全相同網路結構,確保 RQ1 對比公平。

#### C-1. 套件安裝

```bash
# torch 預設裝 CPU wheel,500k steps 約 25 分鐘可接受
uv add "gymnasium>=0.29" "numpy>=1.24" "matplotlib>=3.7" \
       "stable-baselines3>=2.3" "sb3-contrib>=2.3" \
       "torch>=2.2" "pillow>=10.0" "tensorboard>=2.16"

# 若想用 GPU(可選,一次性設定):
uv add torch --index https://download.pytorch.org/whl/cu124 --reinstall
uv run python -c "import torch; print(torch.cuda.is_available())"   # 應印 True
```

程式碼 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 自動判斷,不用手動切。

#### C-2. 檔案架構

```
agents/ppo/
├── __init__.py
├── rollout_buffer.py   # RolloutBuffer + MiniBatch,GAE backward pass
├── ppo_agent.py        # PPOAgent class:select_action / update / save / load
├── train_ppo.py        # 訓練主程式(SubprocVecEnv + TensorBoard + checkpoint)
└── evaluate.py         # 載入 checkpoint 跑 N 集,輸出 E 的 JSON schema
```

| 檔案 | 角色 | 關鍵設計 |
|------|------|----------|
| `rollout_buffer.py` | 收集 rollout、算 advantage | **每筆 transition 連 action_mask 一起存**,因為 update 時要重新 mask logits |
| `ppo_agent.py` | PPO 演算法本身 | `logits.masked_fill(~mask, -inf)` 後接 `Categorical`,自動讓非法動作機率 = 0、entropy = 0 |
| `train_ppo.py` | 8 個 SubprocVecEnv 平行收集 | 每步從 `info["action_mask"]` 拿 mask;若該 env 剛 reset,info 是 terminal 的全 False mask,要 `env_method("action_masks")` 重抓 |
| `evaluate.py` | 最終評估 | **deterministic = argmax(masked logits)**,不是隨機 sample,代表「模型最佳表現」 |

#### C-3. 訓練指令

```bash
# 稀疏 reward
uv run python -m agents.ppo.train_ppo --reward sparse --seed 0

# 密集 reward(D 設計的 shaping)
uv run python -m agents.ppo.train_ppo --reward dense  --seed 0
```

預設超參數(可用 CLI flag 覆寫):

| 參數 | 值 | 備註 |
|------|----|----|
| `--total-steps`   | 500_000 | proposal 規定 |
| `--n-envs`        | 8 | 平行 env 數,SubprocVecEnv |
| `--n-steps`       | 128 | 每個 env 每次 rollout 步數 → 一次 update 用 1024 transitions |
| `--n-epochs`      | 10 | 每次 rollout 重複跑 10 個 epoch |
| `--batch-size`    | 64 | mini-batch |
| `--lr`            | 3e-4 | |
| `--gamma`         | 0.99 | |
| `--gae-lambda`    | 0.95 | |
| `--clip-range`    | 0.2 | PPO ratio 截斷範圍 |
| `--ent-coef`      | 0.01 | 太小會 policy collapse,proposal 也提醒 |
| `--vf-coef`       | 0.5 | value loss 權重 |
| `--max-grad-norm` | 0.5 | gradient clipping |
| `--ckpt-every`    | 50_000 | checkpoint 間隔(env steps) |

> 預設值刻意對齊 SB3 MaskablePPO 預設,理由:「PPO 沒調好」這種質疑可以擋掉一輪。

訓練輸出:
- `checkpoints/ppo_<reward>_seed<S>_step<N>.pt` — 每 50k steps 一次
- `runs/ppo_<reward>_seed<S>/` — TensorBoard event files

#### C-4. 監看訓練曲線

訓練期間另開 terminal:

```bash
uv run tensorboard --logdir runs
```

瀏覽器開 http://localhost:6006,要看的指標:

| 指標 | 應該長怎樣 |
|------|----------|
| `rollout/ep_score_mean` | **必須上升**。沒上升 = agent 沒在學,先檢查 ent_coef、lr |
| `rollout/ep_length_mean` | 通常跟 score 同步上升 |
| `train/entropy` | 緩降。若 1-2 萬 step 內就掉到接近 0 → policy collapse,把 ent_coef 從 0.01 提到 0.02-0.05 |
| `train/approx_kl` | 應在 0.005-0.02 之間。> 0.05 表示 step 太大,降低 lr |
| `train/clip_fraction` | 0.1-0.3 之間健康。長期 > 0.4 表 ratio 常被截斷,降 lr 或加 ent_coef |
| `train/explained_var` | 從 0 慢慢往 1 升。卡在 0 附近 = critic 學不起來 |

#### C-5. 產生交給 E 的 JSON

500k 訓練跑完後:

```bash
uv run python -m agents.ppo.evaluate \
    --checkpoint checkpoints/ppo_sparse_seed0_step500000.pt \
    --reward sparse --episodes 100 --seed 42 \
    --out results/ppo_sparse.json \
    --notes "PPO 500k steps, lr=3e-4, ent_coef=0.01, shared backbone with DQN"

uv run python -m agents.ppo.evaluate \
    --checkpoint checkpoints/ppo_dense_seed0_step500000.pt \
    --reward dense --episodes 100 --seed 42 \
    --out results/ppo_dense.json \
    --notes "PPO 500k steps, dense reward (-0.1*holes -0.05*bumpiness)"
```

產生 `results/ppo_sparse.json` 和 `results/ppo_dense.json`,**這兩個檔要 commit**,組員 E 會從 git 收集所有人的 JSON 畫對比圖。

JSON schema(11 欄,組員 E 訂):

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
  "raw_scores": [28, 35, ...],
  "raw_steps":  [40, 50, ...],
  "timestamp": "2026-05-11T14:30:00",
  "notes": "PPO 500k steps, ..."
}
```

### 組員 D — Reward 設計

只需要修改 `reward_functions.py` 的係數，然後在 `block_blast_env.py` 的 `_dense_shaping()` 中引用：

```python
# block_blast_env.py
from reward_functions import HOLE_PENALTY, BUMPINESS_PENALTY

def _dense_shaping(self):
    return HOLE_PENALTY * self._count_holes() + BUMPINESS_PENALTY * self._count_bumpiness()
```

跑比較實驗：

```python
env_sparse = BlockBlastEnv(reward_mode="sparse")
env_dense  = BlockBlastEnv(reward_mode="dense")
```

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

## 9. 方塊形狀查詢

```python
from env.shapes import SHAPES, SHAPE_NAMES, N_SHAPES, print_shape

print(f"共 {N_SHAPES} 種形狀")   # 35

# 印出所有形狀
for i, (shape, name) in enumerate(zip(SHAPES, SHAPE_NAMES)):
    print_shape(shape, f"#{i} {name}")
    print()
```

---

## 10. 快速除錯

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

## 11. CNN 網路架構（`agents/network.py`）

B 和 C 共用同一份網路，確保對比公平。

```
board  (1, 8, 8) → Conv(1→32,3×3) → Conv(32→64,3×3) → Flatten(4096) → FC(128) ──┐
                                                                                    cat(192) → FC(128) → output
pieces (3, 5, 5) ──────────────────────────────────────── Flatten(75) → FC(64)  ──┘
```

| 類別 | 用途 | 輸出 |
|------|------|------|
| `BlockBlastNet` | DQN Q 網路 | `(B, 192)` Q 值 |
| `BlockBlastActorCritic` | 自製 PPO | `(B, 192)` logits + `(B, 1)` value |
| `obs_to_tensor(obs, device)` | numpy obs → tensor | `(board, pieces)` tuple |

---

## 12. 檔案結構

```
BlockBlastWithRL/
├── env/                          # 組員 A 維護，禁止其他人修改
│   ├── block_blast_env.py
│   ├── shapes.py
│   └── __init__.py
├── agents/
│   ├── network.py                # 共用 CNN backbone，B 和 C 都從這裡 import
│   ├── random_agent.py           # 組員 E
│   ├── greedy_agent.py           # 組員 E
│   ├── dqn/                      # 組員 B 的地盤
│   │   ├── train_dqn.py          # 訓練入口：python -m agents.dqn.train_dqn
│   │   ├── dqn_agent.py          # (B 自行新增)
│   │   ├── replay_buffer.py      # (B 自行新增)
│   │   └── __init__.py
│   └── ppo/                      # 組員 C 的地盤
│       ├── train_ppo.py          # 訓練入口：python -m agents.ppo.train_ppo
│       ├── ppo_agent.py          # (C 自行新增，若不用 SB3)
│       └── __init__.py
├── reward_functions.py           # 組員 D 修改
├── test_env.py                   # 環境驗證，push 前必跑
├── README.md                     # 本文件
└── requirements.txt
```

訓練入口用 module 方式執行（在專案根目錄下）：

```bash
python -m agents.dqn.train_dqn
python -m agents.ppo.train_ppo
```
