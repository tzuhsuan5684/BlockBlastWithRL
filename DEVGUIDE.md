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

- observation 有兩個 key（`board`、`pieces`），需要分別輸入 CNN 或 flatten 後合併
- replay buffer 存的 `obs` 是 dict，注意存取方式
- mask 從 `info["action_mask"]` 拿，每個 transition 都要一起存進 buffer

```python
# buffer 存法範例
buffer.push(obs, action, reward, next_obs, terminated,
            action_mask=mask, next_action_mask=next_mask)
```

### 組員 C — PPO

使用 `sb3_contrib.MaskablePPO` + `MultiInputPolicy`：

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

def make_env():
    return BlockBlastEnv(reward_mode="sparse")

vec_env = SubprocVecEnv([make_env] * 8)   # 8 個並行環境
model = MaskablePPO(
    "MultiInputPolicy", vec_env,
    learning_rate=3e-4,
    ent_coef=0.01,        # 太小會 policy collapse
    clip_range=0.2,
    verbose=1,
)
model.learn(total_timesteps=500_000)
```

> `SubprocVecEnv` 需要 `env` 可以 pickle，目前環境沒有問題。

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

## 11. 檔案結構

```
BlockBlastWithRL/
├── env/
│   ├── block_blast_env.py   # 主環境（組員 A 維護）
│   ├── shapes.py            # 35 種方塊形狀
│   └── __init__.py
├── agents/
│   ├── random_agent.py      # Random baseline（組員 E）
│   └── greedy_agent.py      # Greedy baseline（組員 E）
├── reward_functions.py      # Reward 係數（組員 D 修改）
├── test_env.py              # 環境驗證，改完環境請先跑這個
├── DEVGUIDE.md              # 本文件
└── requirements.txt
```
