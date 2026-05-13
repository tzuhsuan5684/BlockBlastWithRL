# Block Blast RL — 期末報告 Outline

> 三週專題期末報告骨架。各章節主筆人見章節下方括號標註。
> 寫作前先 `git pull origin main` 看其他人有沒有更新；圖表請從 `results/` copy 到 `report/figures/`。

---

## 1. Introduction
**主筆：組員 E** | **對應 proposal §1, §2**

- [ ] 介紹 Block Blast 遊戲規則（8×8 盤面、每回合 3 個方塊、消行 +1、不可逆死局）
- [ ] 點出三個 RL 核心挑戰：稀疏 reward、合法動作空間約束、不可逆死局預防
- [ ] 引出兩個 Research Question：
  - **RQ1**: 同網路架構下 PPO vs DQN 哪個表現好？
  - **RQ2**: 密集 reward shaping 是否提升存活步數與分數？
- [ ] 對應現實應用：排程、資源分配、機器人操作
- [ ] 文章組織

## 2. Related Work
**主筆：組員 E** | **TODO: 文獻彙整**

- [ ] DQN / Double DQN（Mnih et al. 2015 / van Hasselt et al. 2016）
- [ ] PPO（Schulman et al. 2017）
- [ ] Action masking 在離散動作空間的應用（Huang & Ontañón 2020 *invalid action masking*）
- [ ] Reward shaping in sparse-reward envs（Ng et al. 1999 potential-based shaping）
- [ ] Tetris/方塊類遊戲 RL 既有工作（heuristic agents、approximate DP）
- [ ] 強調本工作的 niche：Block Blast 自製環境、無公開 benchmark

## 3. Environment & Methods
**整合：組員 E**；各小節由對應組員撰寫

### 3.1 Environment Design (組員 A) | proposal §3.1
- [ ] 8×8 盤面、35 種方塊形狀（含旋轉）、Gymnasium API
- [ ] Observation: `board (8,8)` + `pieces (3,5,5)`
- [ ] Action: `Discrete(192) = 3 pieces × 64 positions`
- [ ] Reward: sparse(+lines, −10 death) / dense(− 0.1×holes − 0.05×bumpiness)

### 3.2 Network Architecture (組員 B) | proposal §3.2
- [ ] CNN backbone 共用設計（`agents/network.py`）
- [ ] DQN head: `(B, 192)` Q values
- [ ] Actor-Critic head: `(B, 192)` logits + `(B, 1)` value

### 3.3 DQN (組員 B) | proposal §3.3
- [ ] Double DQN + experience replay
- [ ] Action masking: 非法動作 Q ← −∞
- [ ] Hyperparams: epsilon decay, replay buffer size, target update interval

### 3.4 PPO (組員 C) | proposal §3.3
- [ ] MaskablePPO (SB3-contrib) 或自製 PPO
- [ ] Hyperparams: learning rate, entropy coef, clip range
- [ ] 為何 entropy coef 太小會 policy collapse

### 3.5 Reward Shaping (組員 D) | proposal §3.1
- [ ] Sparse vs Dense 公式
- [ ] Holes / Bumpiness 計算定義（`block_blast_env._count_holes`, `_count_bumpiness`）
- [ ] 係數調整邏輯（`reward_functions.py`）

### 3.6 Baselines (組員 E) | proposal §3.4
- [ ] Random agent: uniform sample from legal actions
- [ ] Greedy agent: 一步前瞻、消行最大化、平手隨機
- [ ] 評估流程：100 episodes per agent, seed=42, info["score"] 為主要指標

## 4. Experiments
**整合：組員 E**

- [ ] 訓練設定：500k steps × 2 algorithms × 2 reward modes（B、C 各跑兩次）
- [ ] 評估：每個 final model 跑 100 局, seed=42, 計算 mean ± std
- [ ] 統一 metrics schema (`evaluation/metrics_schema.py`) — 所有 agent 把結果寫成 JSON 到 `results/`
- [ ] 圖表生成：`python -m evaluation.plot_comparison`
- [ ] 表格生成：`python -m evaluation.aggregate` → `results/summary.csv`
- [ ] 硬體規格、隨機種子、訓練時間

## 5. Results
**主筆：組員 E**

- [ ] **Table 1**: 全 agent × reward_mode 的 mean_score / mean_steps（從 `results/summary.csv` 直接貼）
- [ ] **Figure 1**: `figures/comparison_score.png` — Score bar chart
- [ ] **Figure 2**: `figures/comparison_steps.png` — Survival bar chart
- [ ] **Figure 3** (組員 B/C 提供): DQN / PPO 訓練曲線（reward vs steps）
- [ ] **Figure 4** (組員 D 提供): sparse vs dense reward 的 PPO 訓練曲線對比
- [ ] 文字描述各組對比，明確答覆 RQ1（PPO vs DQN）與 RQ2（reward shaping）

### 目前 baseline 結果（Week 1 已完成，2026-05-10）

| Agent  | Reward | mean_score | std_score | mean_steps | std_steps |
|--------|--------|-----------:|----------:|-----------:|----------:|
| Random | sparse |       1.35 |      1.56 |      12.75 |      3.74 |
| Greedy | sparse |       4.32 |      3.84 |      18.27 |      8.09 |

> RL 方法的目標：mean_score > 4.32（超過 Greedy）。

## 6. Discussion
**主筆：組員 E** | **對應 proposal §6**

- [ ] **預期排序**：DQN > Greedy > PPO(未調) > Random，或 PPO(調好) > DQN > Greedy
- [ ] 若 PPO < DQN：可能是 entropy coef 太小造成 policy collapse、超參未調好
- [ ] 若 RL < Greedy：分析 sample efficiency、reward shaping 設計、訓練步數
- [ ] **不可逆死局**：dense reward 是否真的讓 agent 更保守？holes / bumpiness 隨訓練的變化？
- [ ] **動作空間**：192 個離散動作 + mask 帶來的探索負擔
- [ ] Limitations：500k steps 可能不夠、單一 seed 噪音大、無公開 benchmark 比較

## 7. Conclusion
**主筆：組員 E**

- [ ] 重述兩個 RQ 的答案
- [ ] 主要發現（從 Discussion 萃取 1–2 句）
- [ ] 貢獻：(1) 自製 Block Blast Gymnasium 環境，(2) 同網路架構下的 PPO vs DQN 公平對比
- [ ] Future work（連結 proposal §7 次要任務）：Transformer backbone、死局預測 auxiliary task、curriculum 4×4→8×8、消融實驗

## 8. References
**整合：組員 E**

- [ ] Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- [ ] van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
- [ ] Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- [ ] Huang, S. & Ontañón, S. (2020). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.*
- [ ] Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations.* ICML.
- [ ] Stable-Baselines3, Gymnasium 套件引用

---

## 寫作守則

- 圖表 caption 用「Figure N: 描述」格式，文中以 (Fig. N) 引用
- 表格 caption 在表格上方
- 數字統一保留 2 位小數
- mean ± std 格式：`4.32 ± 3.84`
- 章節長度建議：Intro 1 頁、Related Work 1 頁、Methods 2–3 頁、Experiments 1 頁、Results 2 頁、Discussion 1.5 頁、Conclusion 0.5 頁、總計約 8–10 頁

## TODO 進度追蹤

- [x] Baseline 評估（Week 1，組員 E 已完成）
- [x] 評估管線 + JSON schema（Week 1，組員 E 已完成）
- [x] 圖表 / CSV 彙整腳本（Week 2，組員 E 已完成）
- [ ] DQN 500k steps 訓練 + 結果 JSON（Week 3，組員 B）
- [ ] PPO 500k steps 訓練 + 結果 JSON（Week 3，組員 C）
- [ ] Sparse vs Dense reward 對比實驗（Week 2，組員 D）
- [ ] 報告各章節填寫（Week 2–3，全員）
- [ ] 簡報製作 + 預演（Week 3）
