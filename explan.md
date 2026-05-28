1. 強化學習是什麼？
想像你在教一隻狗學坐下：

狗做對了 → 給零食（正 reward）
狗做錯了 → 沒零食（負 reward）
狗慢慢就學會了
RL 就是這樣，差別是「狗」變成了神經網路（agent），「遊戲」是環境（env）。


Agent 看到畫面（observation）
→ Agent 決定要做什麼（action）
→ 環境給分數（reward）
→ Agent 調整策略（policy）
→ 重複幾百萬次
2. Policy 是什麼？
Policy 就是「決策規則」，輸入是你看到的畫面，輸出是「我要做哪個動作」。

用神經網路表示的話：


board (8×8) ──┐
              ├──► 神經網路 ──► 每個動作的分數（logits）
pieces (3×5×5)┘
分數最高的動作 → agent 採取。

3. PPO 是什麼？
PPO = Proximal Policy Optimization，一種訓練 policy 的演算法。

為什麼不直接讓 agent「做對事情就往那方向走」就好？
問題是：如果每次都大幅更新網路，很容易更新過頭，原本好的策略反而被破壞。

PPO 的核心想法：每次只能更新一點點（proximal = 靠近），用一個 clip 機制限制更新幅度：


# ratio = 新policy / 舊policy 的機率比
ratio = new_prob / old_prob

# clip 強制 ratio 不能偏離 1.0 太遠（預設 ±0.2）
loss = -min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage)
4. Advantage 是什麼？
Advantage = 「這個動作比我預期的好多少？」


Advantage = 實際得到的分數 - 我原本預期的分數
Advantage > 0 → 比預期好，下次多做這個動作
Advantage < 0 → 比預期差，下次少做這個動作
「預期分數」是由另一個叫做 Critic 的網路負責估計的。

5. Actor-Critic 架構
PPO 的網路分兩個 head：


                    ┌──► Actor  → 輸出「每個動作的機率」
神經網路主幹（backbone）┤
                    └──► Critic → 輸出「這個狀態值多少分」（V(s)）
Actor = 決定要做什麼
Critic = 評估現在的局面好不好，給 Actor 當參考
6. SB3 是什麼？
Stable Baselines 3 = 幫你把 PPO 這些演算法寫好的 library，你不用自己實作。


# 用 SB3 的話，三行就能跑
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1_000_000)
這個 project 為什麼不用現成的 SB3？
SB3 的 PPO 會自己建立一個內部 policy network。但這個 project 要讓 PPO 和 DQN 用同一個 network，才能公平比較。SB3 不讓你直接替換 network，所以 Member C 自己重寫了 PPO 的訓練邏輯，但 超參數預設值刻意對齊 SB3，讓結果可辯護。

7. Action Masking 是什麼？
Block Blast 的動作空間有 192 個，但大多數都是非法的（超出邊界、格子已填滿）。

如果不過濾，agent 會「學」到去選非法動作，結果一片混亂。

Action Masking = 把非法動作的機率強制設為 0：


logits[非法動作] = -inf   # softmax 後機率 = 0
dist = Categorical(logits=logits)  # 只從合法動作中選
8. GAE 是什麼？
GAE = Generalized Advantage Estimation

計算 Advantage 的時候有一個問題：要看「未來多遠的 reward」？

只看一步 → 太短視，估計噪音很大
看到 episode 結束 → 太長遠，前期的動作很難分辨好壞
GAE 用 λ（預設 0.95）做折衷，對未來的影響做指數衰減：


Advantage ≈ r₀ + γ·r₁ + γ²·r₂ + ... (越遠的 reward 影響越小)
9. 整個訓練流程（對照這個 project）

┌─────────────────────────────────────────────────────────────┐
│ 一次大迴圈（n_updates 次）                                   │
│                                                             │
│  1. 收集 rollout                                            │
│     8 個平行環境（SubprocVecEnv）各跑 128 步                │
│     每步：obs → Actor → action → env.step() → reward        │
│     存進 RolloutBuffer（1024 筆資料）                        │
│                                                             │
│  2. 計算 GAE Advantage                                       │
│     從後往前掃 RolloutBuffer，算每步的 Advantage             │
│                                                             │
│  3. PPO update                                              │
│     把 1024 筆資料切成 64 筆的 minibatch                     │
│     重複 10 次（n_epochs）                                   │
│     每次算 loss → 更新網路                                   │
│                                                             │
│  4. 存 checkpoint，記錄 TensorBoard                          │
└─────────────────────────────────────────────────────────────┘
10. 現在這個 project 在基礎上加了什麼？
有了上面的概念，再看這個 project 的優化就清楚了：

優化	解決什麼問題
Action Masking	防止 agent 選非法動作
Dense Reward	line clear 太稀疏（平均 10 步才出現），dense 每步都有信號給 Critic 學
Reward 係數調小（v2）	係數太大 → 每步 shaping 比 line clear 還重 → Critic 無法收斂
Heuristic obs（40維）	讓 agent 不用從 raw board 自己學「哪欄高、哪有洞」，直接給答案
Afterstate evaluation	不評估「現在的狀態」，而是評估「放了這塊之後的狀態」，更直接
pieces_left obs	讓 agent 明確知道哪個 slot 還有棋子
Combo bonus	鼓勵連續消行，而不只是單次消行
Linear LR decay	訓練後期學習率降低，避免 policy 來回震盪
有沒有哪個部分想更深入了解？例如 Critic 怎麼訓練、GAE 的數學、或是 action masking 的實作細節？