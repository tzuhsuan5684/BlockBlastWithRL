# 組員 C PPO 開發歷程

> 從零實作到兩輪 1M 訓練 + 一次關鍵 reward 修復的完整紀錄,做為報告 PPO 章節的素材。

---

## 0. 任務範圍

| 題目 | 工作 | 對應 proposal |
|---|---|---|
| RQ1 | 在相同網路下,PPO 與 DQN 哪個贏 | §2 |
| RQ2 | 密集 reward shaping 比稀疏 reward 好嗎 | §2 |
| §3.2 | 與 DQN 共用 CNN backbone | 對比公平性 |
| §3.3 | 用 SB3 或自製,要 action masking | 演算法核心 |

最終採取 **方案二:自製 PPO + 共用 `BlockBlastActorCritic`**,而非 SB3 MaskablePPO,理由是 SB3 的 `MultiInputPolicy` 會生自己的 policy network,違反「共用 backbone」的 §3.2 公平性要求。

---

## 1. 實作階段(commit 軌跡)

| commit 順序 | 主題 | 內容 |
|---|---|---|
| `chore: untrack env/__pycache__` | cleanup | 移除誤 commit 的 bytecode |
| `chore: update .gitignore` | cleanup | 開放 `results/*.json` 進 git 給組員 E 收 |
| `chore: add uv project metadata` | setup | `pyproject.toml` + `uv.lock` + `.python-version`,確保隊友 `uv sync` 可重現 |
| `feat: implement custom PPO with shared CNN backbone` | **核心** | `agents/ppo/{rollout_buffer, ppo_agent, train_ppo, evaluate}.py` |
| `docs: document PPO training workflow` | docs | README §8 組員 C 章節重寫 |
| `docs: add CLAUDE.md` | docs | 給未來 Claude session 用的 repo 導覽 |
| `feat: add PPO playback support to demo/play.py` | demo | `--agent ppo --checkpoint <.pt>` 可視化 |
| `fix: ENTER / → fallback in demo manual mode` | bugfix | Windows 中文 IME 攔截 SPACE 的解法 |
| `refactor: reduce hole and bumpiness penalties` | **關鍵** | reward 係數 v1→v2,**這是 PPO 結果的分水嶺** |

---

## 2. 程式架構

```
agents/ppo/
├── rollout_buffer.py   # n_steps × n_envs buffer + GAE-λ
├── ppo_agent.py        # PPOAgent: clipped surrogate + entropy + value loss
├── train_ppo.py        # SubprocVecEnv × 8 平行 rollout + TensorBoard + ckpt
└── evaluate.py         # deterministic argmax,輸出組員 E 的 JSON schema
```

### 三個關鍵實作決定

1. **action_mask 一定要存進 buffer**: PPO update 重新評估動作時要再次套 mask,否則非法動作會被當合法行為學習。`logits.masked_fill(~mask, -inf)` 後接 `Categorical`,讓非法動作機率 = 0、entropy 貢獻 = 0。
2. **VecEnv 取 mask 的 timing**: `info["action_mask"]` 是 step 後的 mask,但 SubprocVecEnv 在 `done=True` 時會自動 reset,info 帶的是 terminal 的全 False mask,要 `env_method("action_masks")` 重抓新 episode 的 mask。
3. **超參數預設值對齊 SB3 MaskablePPO**: 不是因為 SB3 最好,是為了擋掉「PPO 沒調好」的審稿質疑。

---

## 3. 兩輪訓練結果

### v1(reward 係數 `-0.3 / -0.1`,500k steps)

| 指標 | sparse | dense |
|---|---|---|
| `ep_score_mean` | 2.69 | **1.89** |
| `ep_score_max` | 14.5 | 8.4 |
| `ep_length_mean` | 15.78 | 14.27 |
| `value_loss` | 1.30 | **152** 🚨 |
| `approx_kl` | 0.029 | 0.06 |
| `explained_var` | 0.64 | 0.49 |
| `policy_loss` | -0.027 | -0.0001 |

**現象**: dense 比 sparse 還差,而且 random baseline 也只有 ~2 分,dense PPO 等於沒在學。

### 診斷(沒有跑就會錯過的部分)

從 `train/value_loss = 152` 切入,推出整條失敗鏈:

```
shaping 太強(每步 -1 ~ -3)→ return variance 暴衝
→ critic 無法收斂(value_loss 152)
→ advantage 估計不準(explained_var 0.49)
→ policy gradient 方向亂跳(KL 0.06,clip_fraction 0.27)
→ PPO 把多數 update 截斷(clip_fraction 0.27)
→ policy 實質上沒在動(policy_loss ~0)
→ agent 學到「快點死避免累積負 reward」
```

**根本原因**: `reward_functions.py` 裡 `HOLE_PENALTY = -0.3`、`BUMPINESS_PENALTY = -0.1`,實際每步 shaping 達 `-1 ~ -3`,**遠大於 +1 line clear**。

### 修復

降到 `-0.02 / -0.01`(比 proposal 的 `-0.1 / -0.05` 還弱 5 倍,比實際在用的 v1 弱 15 倍),把 shaping 變成「輕推一把」而非「主導 reward」。

### v2(reward 係數 `-0.02 / -0.01`,1M steps)

| 指標 | sparse | dense | v1→v2 變化(dense) |
|---|---|---|---|
| `ep_score_mean` | 2.82 | **4.22** | **+123%** ⬆️ |
| `ep_score_max` | 14.6 | 16.3 | +94% |
| `ep_length_mean` | 16.4 | 19.1 | +34% |
| `value_loss` | 2.17 | **0.74** | **−99.5%** ⬇️ |
| `approx_kl` | 0.048 | 0.026 | 進入健康區 |
| `explained_var` | 0.51 | 0.35 | 較低但 value_loss 也小 |
| `policy_loss` | -0.013 | **-0.026** | 真正在改善 |

**RQ2 結論定案**: 當 shaping 係數合理時,**dense > sparse**(4.22 vs 2.82,+50%)。但 shaping 設計嚴重影響結果(v1 vs v2 dense 差 123%)。

---

## 4. 主要發現(寫進報告的部分)

### Finding 1 — RQ2 確認

Dense reward shaping 顯著優於 sparse,在 **正確係數設定** 下。

### Finding 2 — Shaping 係數對性能的決定性

同樣 dense 公式,係數從 (-0.3, -0.1) 改成 (-0.02, -0.01) 帶來 **+123% 提升**。這證明 shaping 不能憑直覺亂設,而要看「single-step shaping 量」是否被「line clear reward」蓋過。

### Finding 3 — 失敗 / 修復的內部機制完整可追

不是黑箱「重訓就好」,而是有明確的 cause-and-effect:

```
shaping variance ↑ → value_loss ↑ → advantage 不準 → policy_loss ~0 → score 不升
```

每一環都有 TensorBoard 數據佐證。

### Finding 4 — Sparse 的訓練後期不健康

Sparse v2 在 600k-1M 階段 `value_loss` 從 1.3 漂到 2.2、`explained_var` 從 0.65 降到 0.51、`approx_kl` 從 0.02 漂到 0.05,但 `ep_score_mean` 沒進步(2.7 → 2.8)。**sparse 大概在 500k-700k 就到天花板**,1M 是 overkill。

---

## 5. 給組員的 take-away

| 組員 | 跟你相關的事 |
|---|---|
| **A** | env/`_dense_shaping` 已經接上 `reward_functions.py`,以後 D 改係數會直接生效 |
| **B** | DQN 訓練時 reward_mode 也記得用 dense v2 的新係數;不然 RQ1 對比會不公平 |
| **D** | 你的 reward shaping 工作可以開始迭代了。可考慮做 ablation:`(-0.3, -0.1)` vs `(-0.1, -0.05)` vs `(-0.02, -0.01)` vs `(0, 0)` 四點,畫一張「shaping 強度 vs 最終 score」的曲線,**這就是你的核心 deliverable** |
| **E** | 我會交兩個 JSON:`ppo_sparse.json`、`ppo_dense.json`。需要的話我再補 `ppo_dense_harsh.json` 做你的 ablation 圖 |
