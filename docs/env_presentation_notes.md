# Environment 簡報講稿與技術筆記（第 5、6 頁 + Appendix）

> **用途**：DL 期末報告中「Environment 介紹」（第 5、6 頁）的上台講稿、技術問答備忘，以及 Appendix 兩張示意圖的說明。
> **對應程式碼**：[`env/block_blast_env.py`](../env/block_blast_env.py)、[`env/shapes.py`](../env/shapes.py)、[`agents/ppo/afterstate.py`](../agents/ppo/afterstate.py)、[`agents/ppo/network_afterstate.py`](../agents/ppo/network_afterstate.py)
> **Reward 版本**：採 **B 版**（`lines²` + combo 連擊加成），與 `test` 分支現行程式碼一致。
> **相關資產**：[appendix_env_diagrams.pptx](appendix_env_diagrams.pptx)、[appendix_env_diagrams.html](appendix_env_diagrams.html)、[appendix_A_heuristics.png](appendix_A_heuristics.png)、[appendix_B_coordsystem.png](appendix_B_coordsystem.png)、[build_appendix_pptx.py](build_appendix_pptx.py)

---

## 目錄

1. [上台講稿（約 4 分鐘，英中對照）](#1-上台講稿約-4-分鐘英中對照)
2. [技術詳解（防止被問倒）](#2-技術詳解防止被問倒)
3. [Appendix：afterstate 特徵與兩張示意圖](#3-appendixafterstate-特徵與兩張示意圖)
4. [總結與待辦](#4-總結與待辦)

---

## 1. 上台講稿（約 4 分鐘，英中對照）

總字數約 570 字，配合 `⏱` 停頓正好約 4 分鐘。Page 5 ≈ 2:10，Page 6 ≈ 1:50。

### 📄 Page 5 — Environment

**English**

> Now let me walk you through how we built the environment. We implemented Block Blast as a **standard Gymnasium environment**, so any RL algorithm can plug straight in. ⏱
>
> Let's start with the **action space** — it's a `Discrete(192)`. Why 192? Each round deals **three** pieces, and the board is 8×8, that's **64** cells — three times 64 is 192. We flatten that into a single integer with the formula `piece_index × 64 + row × 8 + col`. So every action the agent takes picks **both which piece to place and where to place it** — piece and position, in one shot. ⏱
>
> Next, the **reward**. We designed two variants. The sparse reward rewards clearing lines — and here's the key: clearing **more lines at once is worth disproportionately more**, because the reward scales with the **square** of the number of lines cleared, and **consecutive** clears stack an extra **combo** bonus. Game over gives **minus ten**. The dense reward adds two shaping terms on top: a small penalty for **holes** created, and a small penalty for board **bumpiness**. ⏱
>
> Now, a finding I think is worth sharing: for our best method, the **sparse reward actually beat the dense one**. The reason is that when the shaping penalties get too strong, they drown out the reward for clearing lines — the agent even learns to **lose early** just to stop accumulating negative reward. **Too much hand-holding ends up distracting from the real goal.** ⏱
>
> Finally, **action masking** — and this is critical. At any board state, **most of the 192 actions are actually illegal**: out of bounds, overlapping a filled cell, or using a piece slot that's already been used. We call `env.action_masks()` to get a **192-dimensional boolean array** that masks all of those out, so the agent only ever chooses among legal moves. And this mask **doubles as our termination signal**: when all 192 actions are masked — no legal move left — the episode ends. That's our exact definition of game over.

**中文翻譯**

> 接下來我帶大家看我們怎麼建這個環境。我們把 Block Blast 實作成一個**標準的 Gymnasium 環境**，所以任何 RL 演算法都能直接接上。
>
> 先看 **action space**，它是 `Discrete(192)`。為什麼是 192？每一回合會發**三塊**方塊，棋盤是 8×8，也就是 **64** 格——三乘以 64 就是 192。我們用 `piece_index × 64 + row × 8 + col` 這條公式把它攤平成一個整數。所以 agent 的**每一個 action，會同時決定要放哪一塊、以及放在哪個格子**——選塊和選位置，一步到位。
>
> 再來是 **reward**，我們設計了兩種版本。Sparse reward 獎勵清行，重點在於：**一次清越多行，分數會不成比例地高**，因為它隨清除行數的**平方**成長，而且**連續**清行還會疊加額外的 **combo** 加成。遊戲結束給 **−10**。Dense reward 在這之上再加兩個 shaping 項：一個對製造出的**洞**的小懲罰，一個對盤面**崎嶇度**的小懲罰。
>
> 這裡有個值得分享的發現：對我們的最佳方法來說，**反而是 sparse reward 贏過 dense**。原因是當 shaping 的懲罰太強，它會蓋過清行的獎勵——agent 甚至會學會**提早輸掉**，只為了不再累積負分。**過度的人為引導，最後反而干擾了真正的目標。**
>
> 最後是 **action masking**，這非常關鍵。在任何盤面下，**192 個 action 絕大多數都是非法的**：出界、疊到已填的格、或用到已經用過的方塊格。我們呼叫 `env.action_masks()` 取得一個 **192 維的布林陣列**把這些全部遮掉，agent 永遠只在合法動作中選擇。而這個遮罩還**兼任終局訊號**：當 192 個 action 全被遮蔽、沒有合法步可走時，這一局就結束——這就是我們對 game over 的精確定義。

### 📄 Page 6 — Observation

**English**

> So that's how the environment works — now let's look at **what the agent actually sees** at each step. The observation has **four** components. ⏱
>
> **First, the board** — an 8×8 float32 grid, where 0 is empty and 1 is filled. This is the agent's raw vision.
>
> **Second, the pieces** — the three pieces for the current round, each rendered into its own **5×5** grid. We use 5×5 because the largest piece is five cells wide. If a piece has already been placed, its grid is **all zeros**.
>
> **Third, pieces_left** — a 3-dimensional binary vector that tells the agent directly **which of the three slots are still available**.
>
> And **fourth** — the one I want to highlight — **heuristics**: a **40-dimensional** vector of hand-crafted features. It includes things like each **column's height**, the number of **holes**, board **bumpiness**, and so on, all normalized to between 0 and 1. These are exactly the **tactical signals a human player would look at** — by feeding them in directly, the agent doesn't have to learn concepts like *"holes are bad"* from scratch. ⏱
>
> So to sum up, this observation deliberately gives the agent **two kinds of information at once**: the **raw visual** from the board and pieces, and the **domain-knowledge features** from the heuristics. The agent can both *see* the board and *read off* ready-made tactical cues — and that's the foundation our methods build on. I'll hand over to the next speaker.

**中文翻譯**

> 看完環境怎麼運作，接下來看 agent **每一步實際看到什麼**。Observation 由**四個**部分組成。
>
> **第一，board**——一個 8×8 的 float32 網格，0 代表空、1 代表已填。這是 agent 的原始視覺。
>
> **第二，pieces**——當回合的三塊方塊，每一塊都畫進自己的 **5×5** 網格。用 5×5 是因為最大的方塊有五格寬。如果某塊已經放下去了，它的網格就**全部是 0**。
>
> **第三，pieces_left**——一個 3 維的二元向量，直接告訴 agent**三個方塊格中哪些還能用**。
>
> **第四**，也是我想特別強調的——**heuristics**：一個 **40 維**的手工特徵向量。裡面包含像是每一**欄的高度**、**洞**的數量、盤面**崎嶇度**等等，全部正規化到 0 與 1 之間。這些正是**人類玩家會去看的戰術指標**——直接餵給 agent，它就不必從零開始學「洞是壞事」這種概念。
>
> 所以總結來說，這份 observation 刻意**同時**給 agent 兩種資訊：來自 board 和 pieces 的**原始視覺**，以及來自 heuristics 的**領域知識特徵**。agent 既能「看」盤面，又能「讀」到現成的戰術提示——這也是我們後面方法的基礎。接下來交給下一位同學。

### 上台提醒

- **節奏**：講完 `192` 的公式務必停一拍（⏱）讓聽眾消化；Page 6 用「First / Second / Third / Fourth」串，聽眾很好跟。
- **記憶點**：*"Too much hand-holding ends up distracting from the real goal"* 講慢一點，是 Page 5 的亮點。
- **B 版已採用**：reward 講的是「平方 + combo」精確版。唯一仍要確認的是「sparse 贏 dense」這句，要對得上**最新一次**的實驗數字（文件舊快照是 dense 較高，見 §4）。

---

## 2. 技術詳解（防止被問倒）

以下每一點都對照過現行程式碼，括號標出處。

### 2.1 Action space `Discrete(192)` 與編碼

- **為什麼 192**：每回合發 3 塊方塊，棋盤 8×8 = 64 格 → `3 × 64 = 192` 種「(選哪塊, 放哪格)」組合。
- **編碼公式**：`action = piece_idx × 64 + row × 8 + col`（[`encode_action`](../env/block_blast_env.py)）。這是進位制攤平，方便 policy 輸出單一 192 維分類。
- **解碼**：`piece_idx = action // 64`；`remainder = action % 64`；`row = remainder // 8`；`col = remainder % 8`（[`_decode_action`](../env/block_blast_env.py)）。
- **可能被追問**：「為什麼一個 action 同時含選塊與放位置，而不分兩步？」→ 三塊同時擺在面前、可任意順序放，把「選塊」也編進 action 能讓 agent 一次決策，masking 也更乾淨。

### 2.2 Action masking：為何「必要」+ `logits[~mask] = -inf` 原理

**為何必要**：任何盤面下 192 個動作絕大多數非法（出界 / 疊到已填格 / 用到已用 slot）。不遮罩的話，softmax 會分機率給非法動作，agent 學到垃圾。

**數學原理**：policy 對每個動作輸出 logit $z_i$，softmax 為

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

把非法位置設 $-\infty$，因為 $e^{-\infty}=0$：

$$p_i = \frac{e^{z_i}}{\sum_{j\in\text{legal}} e^{z_j}}$$

- 非法動作機率變 **0**，永不被採樣。
- 合法動作**彼此的相對機率完全不變**（等於只在合法動作上做 softmax）。
- **argmax** 也安全：$-\infty$ 永不是最大值，所以 argmax 一定落在合法動作。

**實務細節**：
- 不用真 `float('-inf')`（易出 `NaN`），用大負數如 `-1e9` 或 `torch.finfo(dtype).min`；PyTorch 慣用 `logits.masked_fill(~mask, -1e9)`。
- **DQN**：選動作 $\arg\max_a Q(s,a)$，非法 Q 設 $-\infty$；計算 target 的 $\max_{a'}Q(s',a')$ **也要套 mask**。
- **PPO**：用 masked logits 建 `Categorical(logits=...)`，採樣 / `log_prob` / entropy 全基於合法分布。
- SB3-contrib `MaskablePPO` 內建自動拉 mask；**自訂 DQN/PPO 必須自己套**。

**程式碼**：`action_masks()` 回傳 `(192,)` bool；`_can_place()` 只檢查 `r>=8 or c>=8`（右/下出界）與 `board[r,c]==1`（重疊），不檢查負座標——因為 offset 全非負、`row/col∈[0,7]`，座標不可能往上/左跑。終局：`terminated = not np.any(mask)`，終局再 `reward -= 10`。

### 2.3 Reward 設計（B 版：平方 + combo）

現行 [`step()`](../env/block_blast_env.py) 的 sparse reward：

```python
if lines > 0:
    self.combo_streak += 1
    reward = float(lines ** 2) * (1.0 + COMBO_STREAK_BONUS * self.combo_streak)
else:
    self.combo_streak = 0
    reward = 0.0
if self.reward_mode == "dense":
    reward += self._dense_shaping()   # HOLE_PENALTY*holes + BUMPINESS_PENALTY*bumpiness
# 終局再 reward -= 10
```

- 常數（[`reward_functions.py`](../reward_functions.py)）：`COMBO_STREAK_BONUS=0.2`、`HOLE_PENALTY=-0.02`、`BUMPINESS_PENALTY=-0.01`、死亡 `-10`（寫死在 env）。
- 數值範例：清 1 行＝`1²×1.2=1.2`；一次清 2 行＝`2²×1.2=4.8`；連續第二步再清 1 行＝`1²×(1+0.2×2)=1.4`。
- **平方** → 鼓勵一次清多行；**combo** → 鼓勵連續清行。
- ⚠️ 陷阱：`SPARSE_LINE_REWARD=1.0` 與 `DEATH_PENALTY=-10.0` 這兩個常數**沒被 env 使用**（env 只 import `HOLE_PENALTY/BUMPINESS_PENALTY/COMBO_STREAK_BONUS`），是殘留設定；docstring 與 `CLAUDE.md` 仍寫舊版「+1 per line」，以 `step()` 實作為準。

**為何 dense 可能更差**（亮點，專案踩過的坑）：shaping 係數太大時，每步累積的負分會蓋過清行正獎勵，PPO critic 的 `value_loss` 爆到 ~150、`approx_kl` 升高、policy gradient 被 clip 掉，agent 學會「早死少扣分」。把係數降 5–15 倍（現 `−0.02/−0.01`）才把 `value_loss` 拉回 <1。詳見 [docs/ppo_journey.md](ppo_journey.md)。

### 2.4 Observation 四元件（[`observation_space`](../env/block_blast_env.py)）

| key | shape | 內容 |
|---|---|---|
| `board` | (8,8) float32 | 0 空 / 1 填，agent 的原始視覺 |
| `pieces` | (3,5,5) float32 | 三塊各畫進 5×5（最大塊 5 寬）；用過的 slot 全 0 |
| `pieces_left` | (3,) float32 | binary，1 = 該 slot 還有方塊 |
| `heuristics` | (40,) float32 | Tier-1 手工特徵，正規化 [0,1]（見 §2.5） |

> 註：`CLAUDE.md` 仍寫 observation 只有 board/pieces 兩元件，那是過時文件；現行程式碼是**四元件**，講稿正確。

### 2.5 40 維 heuristic 向量逐段拆解（[`_compute_heuristics`](../env/block_blast_env.py)）

全部算在**當前**盤面、各自除最大值正規化到 [0,1]：

| 切片 | 維度 | 內容 | ÷ | 來源 |
|---|---|---|---|---|
| `[0:8]` | 8 | 每欄高度（最頂填滿格→底部；**無重力，含洞**） | 8 | `_column_heights()` |
| `[8:16]` | 8 | 每欄洞數（某填滿格下方的空格） | 7 | `_holes_per_col()` |
| `[16:24]` | 8 | 每**橫列**填滿格數 | 8 | `board.sum(axis=1)` |
| `[24:32]` | 8 | 每**直行**填滿格數 | 8 | `board.sum(axis=0)` |
| `[32:39]` | 7 | 相鄰欄高度差（**逐對 7 個**，非總和） | 8 | `abs(diff(heights))` |
| `[39:40]` | 1 | 合法動作數（危險度訊號） | 192 | `count_nonzero(action_masks())` |

合計 8+8+8+8+7+1 = **40**；最後 `np.clip(out,0,1)` 保險。正規化讓各維尺度相近、訓練穩定。
> 與 Appendix 的 `bumpiness_after` 區別：這裡 bumpiness 是 **7 個逐對分量**；Appendix 那個是**加總成 1 純量**。

### 2.6 row/col 對應方塊的哪個錨點？

**答：bounding-box 左上角**，即 offset 座標系原點 `(0,0)` 對齊到棋盤 `(row,col)`。方塊每格 = `(row+dr, col+dc)`（[`step()`](../env/block_blast_env.py)）。`shapes.py` 所有 offset 非負 → 方塊只往**右下**展開。

⚠️ **陷阱：錨點那格不一定被填**。`(row,col)` 是外接框左上角，不是某個實體格、也不是中心。很多 shape 的 `(0,0)` 是空的，例如 `J-0 = [(0,1),(1,1),(2,0),(2,1)]`、以及 `L-270`、`BigL-270`、`T-north`、`T-west`、`S-flat`、`Z-up`、`corner-NE`。
- 正確答法：被問「(row,col) 是方塊哪個點」→ **「外接框左上角」**，不要說「方塊本身的格」或「中心」。

### 2.7 pieces 與放置共用同一套座標系

- **看（observation）**：`shape_to_grid()` 用 `grid[dr][dc]=1` 畫進 5×5（[`shapes.py`](../env/shapes.py)）。
- **放（action）**：`step()` 用 `board[row+dr][col+dc]=1`。
- 兩者用**同一組 offset、同一個原點 (0,0)=bbox 左上角**。

**例**：J-0 在 5×5 看到 `(0,1)(1,1)(2,0)(2,1)`；放到 `(row=2,col=3)` → `(2,4)(3,4)(4,3)(4,4)`。5×5 裡 `(0,1)` 那格 = 棋盤 `(row, col+1)`，**相對位置一一對應**。
- **意義**：agent 看到的相對形狀 = 它放下去的相對 footprint，中間**沒有隱藏的座標轉換**要學；padding 永遠在右下。

### 2.8 講稿 vs 程式碼核對結論

- Page 5/6 的描述**全部與程式碼一致**：action 編碼、masking（出界/重疊/已用 slot、全遮罩終局）、四元件 observation、5×5、用過全 0、heuristics 40 維 [0,1]。
- 唯一曾有的差異是 reward（原稿「+1 per line」），**已改用 B 版**修正。
- 待確認（非 env 程式碼）：「sparse 贏 dense」要對最新數字（見 §4）。

---

## 3. Appendix：afterstate 特徵與兩張示意圖

### 3.1 第 16 頁「9 Hand-crafted Features」核對與解釋

這 9 個與 §2.5 的 40 維**是不同的兩套**（見 §3.3）。逐項對照 [`afterstate.py`](../agents/ppo/afterstate.py)，**全部一致**：

| 特徵 | 白話 | 偏好 | 範圍 |
|---|---|---|---|
| `lines_cleared` | 這步清掉幾行＋列 | ↑ | 0–16 |
| `eroded_piece_cells` | 這塊有幾格參與被清的行列（Dellacherie 經典） | ↑ | 0–5 |
| `holes_after` | 放完後的洞數 | ↓ | 0–64 |
| `max_height_after` | 最高欄高 | ↓ | 0–8 |
| `mean_height_after` | 平均欄高 | ↓ | 0–8 |
| `bumpiness_after` | 相鄰欄高差**總和** | ↓ | 0–56 |
| `row_transitions_after` | 每列「填↔空」交界數 | ↓ | 0–72 |
| `col_transitions_after` | 每行「填↔空」交界數 | ↓ | 0–72 |
| `near_full_count_after` | ≥7/8 的行列數（差一格就清） | ↑ | 0–16 |

兩個投影片為簡潔省略、但該知道的細節：
1. **transitions 把棋盤邊緣當作「已填」**（`prev` 從 1 起算、結尾補一次）——全空列算 2、全填列算 0；這是 Tetris Dellacherie/BCTS 的標準定義。
2. 這 9 個**刻意不正規化**（保留原始整數尺度），與 40 維 heuristics 正規化到 [0,1] **相反**（理由見 §3.2 末）。

**注意**：「偏好方向」只是直覺；實際每個特徵的權重是 `Linear(9,1)` **學出來的**，不是人工指定（與 Dellacherie 手調權重的差別）。

### 3.2 Afterstate evaluation 方法

來自 [`afterstate.py`](../agents/ppo/afterstate.py) 與 [`network_afterstate.py`](../agents/ppo/network_afterstate.py) 的設計：

1. **為何能用 afterstate**：Block Blast 放方塊是**確定性**的——選了動作盤面就固定，隨機性只在三塊用完、重新發牌時才進入。所以可以「往前看一步」。
2. **做法**：對每個合法動作，在複製盤面上模擬放下 → 算 9 個 afterstate 特徵 → 過**一個共享的** `Linear(9→1)` → 192 個動作分數（logits）→ 選最好的。
3. **critic**：另用 §2.5 的 **40 維 heuristics** 過 `MLP(40→64→1)` 估 `V(s)`。
4. **好處**：整個 actor 僅 **~2.7k 參數**，比共用的 `BlockBlastActorCritic` 小好幾個數量級；特徵已 informative，小模型梯度變異低、學得穩。

**為何 9 特徵不正規化（亮點）**：線性 scorer 要自己學權重；若先壓到 [0,1]，初始各動作的 logit 差異會小到被 entropy bonus 蓋過，policy 卡在接近 uniform、無法 bootstrap。保留原始尺度才能讓動作分數一開始就拉得開。**與 40 維 heuristics 正規化的對比，本身就是好講點**（用途不同：一個餵單層線性打分、一個餵有 hidden layer 的 critic）。

### 3.3 「9 afterstate features」 vs 「40 heuristics」對照

| | 40 維 heuristics（第 6 頁） | 9 features（第 16 頁） |
|---|---|---|
| 算在哪個盤面 | **當前** state | **模擬放下後**的 afterstate |
| 每個 action 一份？ | 否，整個 state 共一份 | **是**，`(192, 9)` |
| 正規化 | 是，[0,1] | **否**，原始整數 |
| 在程式碼哪 | env observation | PPO agent 端，**不在 env** |
| 餵給誰 | **critic** 估 V(s) | **actor** 算動作分數 |

一句話記法：**actor 看「每個動作會把盤面變成什麼樣」（9 維 afterstate），critic 看「我現在處境如何」（40 維 heuristics）。**

### 3.4 兩張示意圖（PPTX/HTML/PNG）說明

已產生兩張 16:9、複刻簡報風格（Office accent1 藍 banner、藍表頭交替列、Calibri）的 appendix 投影片：

- **Slide A — 40-D Heuristic Vector**（[appendix_A_heuristics.png](appendix_A_heuristics.png)）：上方分段條顯示 6 個 block 如何串成 `[0:40]`（index 範圍、維度、正規化分母），下方對照表給意義與程式來源。對應 §2.5。
- **Slide B — Shared Coordinate System**（[appendix_B_coordsystem.png](appendix_B_coordsystem.png)）：左 5×5（agent 看到的 J-0）↔ 右 8×8（放到 `(row=2,col=3)`），①②③④ 標出同一格在兩邊的對應，橙色虛線框標 anchor `(0,0)`/`(2,3)`（故意都空，示範「錨點可為空、padding 在右下」）。對應 §2.6、§2.7。

**檔案**：可編輯原生 PPTX [appendix_env_diagrams.pptx](appendix_env_diagrams.pptx)（已用 PowerPoint 渲染驗證）、HTML 原始檔 [appendix_env_diagrams.html](appendix_env_diagrams.html)、生成腳本 [build_appendix_pptx.py](build_appendix_pptx.py)（改完重跑：`uv run --with python-pptx python docs/build_appendix_pptx.py`）。
> 頁碼目前是佔位 `A`/`B`，併入主簡報後請改成實際頁碼。

**若被問到時的口頭講法（各 ~30 秒）**：
- Slide A：「這 40 維是把人類會看的盤面指標——每欄多高、有幾個洞、哪行哪列快滿、平不平、還剩幾步——攤成固定長度、正規化後餵給 critic。」
- Slide B：「方塊在 observation 怎麼畫、在棋盤怎麼放，用的是同一個左上角錨點與同一組 offset，所以 agent 看到的形狀直接對應它放下去的位置，不必額外學座標換算。」

### 3.5 ⚠️ arXiv 引用要驗證

`afterstate.py` / `network_afterstate.py` 的 docstring 引用 **Chen et al. 2026, arXiv:2603.26765, Fig. 11**。此編號超出可驗證範圍，**上台前請親自查證**這篇論文確實存在、編號正確、Fig. 11 確為該架構；若查無此文（可能是開發時的 placeholder），引用要拿掉或換成 afterstate / Dellacherie 特徵的經典文獻。

---

## 4. 總結與待辦

**這份筆記涵蓋**：第 5、6 頁的 4 分鐘英中講稿（B 版 reward）、environment 的完整技術問答備忘（action 編碼、masking 數學、reward、observation、40 維特徵、錨點與座標系），以及 Appendix 的 9 個 afterstate 特徵、afterstate 方法、兩張示意圖說明。

**上台前 checklist**：
- [ ] 確認「sparse 贏 dense」對得上**最新**實驗數字（`CLAUDE.md` 舊快照是 dense 4.22 > sparse 2.82，可能是加入 heuristic/afterstate 前的結果）。
- [ ] 查證 arXiv:2603.26765 是否真實（§3.5）。
- [ ] 把兩張 appendix 圖頁碼從 `A`/`B` 改成實際頁碼。
- [ ] 確認簡報字型（這兩張用 Calibri，主檔若為 Aptos 可全選統一）。

**關鍵記憶點**：
- Page 5：「Too much hand-holding ends up distracting from the real goal.」
- 錨點＝**bounding-box 左上角**（可為空）。
- actor 用 **9 維 afterstate**、critic 用 **40 維 heuristics**——兩套不同特徵。
