# PPO Presentation Outline — Section 05 (Member C)
# PPO 簡報大綱 — 第 05 章節（第 C 組員）

> Audience: RL course final report. RL basics (MDP, policy gradient, A2C/A3C)
> were already covered in class — don't re-teach them; mention A2C only as the
> family PPO improves on. Focus on PPO, which the class did not cover.
> 聽眾：RL 課程期末報告。RL 基礎（MDP、policy gradient、A2C/A3C）課堂已教——
> 不要重教；A2C 只當作 PPO 改進的「同類」一筆帶過。聚焦課堂沒講的 PPO。
>
> Scope: PPO concept → training loop (motivation) → math (DETAILED) →
> implementation (EMPHASIZED) → two variants. No code on slides, no results.
> 範圍：PPO 概念 → 訓練流程（動機）→ 數學（詳細）→ 實作（重點）→ 兩個變體。
> 投影片不放程式碼、不含結果。
>
> Narrative: pose the puzzle first (PPO reuses each batch — how is that safe
> on-policy?), then let the math answer it (importance sampling + clipping).
> 敘事：先拋出疑問（PPO 重複利用每批資料——on-policy 怎麼還安全？），
> 再讓數學來解答（重要性採樣 + 裁剪）。
>
> Target: section page + 14 content slides, ~11–13 min. Trim with the guide at
> the bottom.
> 目標：章節頁 + 14 張內容，約 11–13 分鐘。要砍時間看最後指南。
>
> On-slide text = English. 中文 = your reading aid only.
> 投影片文字＝英文。中文＝只給你閱讀理解用。

---

## Slide 0 — Section Divider / 章節頁

**On slide / 投影片內容**
- **05 · PPO**
- Subtitle: *Proximal Policy Optimization — algorithm, math, and a from-scratch implementation for Block Blast*
- Your name · Member C

中文：
- 大標：05 · PPO
- 副標：Proximal Policy Optimization——演算法、數學、以及為 Block Blast 從零打造的實作
- 你的名字 · 第 C 組員

**Script / 講稿**
> "This section is on PPO — Proximal Policy Optimization, the algorithm I used for our agent. I'll cover what kind of method it is, the math that makes it work, and how I implemented it from scratch."

中文：
> 「這個章節講 PPO——Proximal Policy Optimization，也就是我為我們 agent 用的演算法。我會講它是哪一類方法、讓它運作的數學、以及我怎麼從零實作它。」

---

## Slide 1 — What Kind of Algorithm Is PPO? / PPO 是哪一類演算法？

**On slide / 投影片內容**
- PPO sits on three axes:
  - **Model-Free** — learns from experience; never models the game's transitions
  - **Policy-Based (Actor–Critic)** — directly optimizes the policy π(a|s), *not* Q-values
  - **On-Policy** — trains on data from the current policy, then discards it
- Same actor–critic family as **A2C/A3C**, but **more stable and more sample-efficient**
- Contrast with our DQN: DQN is *value-based* + *off-policy*

中文：
- PPO 在三個軸上的定位：
  - **Model-Free**——從經驗學，不對遊戲的轉移建模
  - **Policy-Based（Actor-Critic）**——直接優化策略 π(a|s)，不是學 Q 值
  - **On-Policy**——用當前策略的資料訓練，用完即丟
- 與 A2C/A3C 同屬 actor-critic 家族，但**更穩定、樣本效率更高**
- 與我們的 DQN 對比：DQN 是 value-based + off-policy → 這正是我們專案要比較的兩端

**Script / 講稿**
> "First, where PPO sits. It's model-free — it learns purely from experience and never tries to predict the game's dynamics. It's policy-based, meaning it directly optimizes the policy, the actor, rather than learning Q-values like DQN does. And it's on-policy — it trains on data the current policy just collected, then throws it away. It's in the same actor–critic family as A2C and A3C, but it's noticeably more stable and squeezes more learning out of each batch. Notice the contrast with our DQN, which is value-based and off-policy — that contrast is exactly what our project compares."

中文：
> 「首先是 PPO 的定位。它是 model-free——純粹從經驗學，從不試圖預測遊戲的動態。它是 policy-based，意思是直接優化策略（actor），而不是像 DQN 那樣學 Q 值。它是 on-policy——用當前策略剛收集的資料訓練，然後丟掉。它和 A2C、A3C 同屬 actor-critic 家族，但明顯更穩定、而且能從每批資料榨出更多學習。注意它和我們 DQN 的對比，DQN 是 value-based、off-policy——這個對比正是我們專案要比較的。」

---

## Slide 2 — The PPO Training Pipeline / PPO 訓練流程（三階段循環）

**On slide / 投影片內容**
- One iteration = three stages, repeated until trained:
```
  ┌→ 1. ROLLOUT ── actor plays N steps; store  s, a, logπ_old, r, V, done
  │       │
  │       ▼
  │  2. GAE ────── backward pass → advantages Â, returns R̂   (network frozen)
  │       │
  │       ▼
  │  3. OPTIMIZE ─ shuffle; K epochs of minibatch updates on the SAME batch
  │       │
  └───────┘  repeat next iteration
```
- vs A2C: stage 3 **reuses the same batch for K epochs**, instead of discarding it after one update → big efficiency win
- But reusing on-policy data is exactly what breaks vanilla policy gradient — **so how does PPO stay safe?**
- **Two tricks, both coming up in the math:**
  - **Importance sampling (rₜ)** → makes the reuse correct
  - **Clipping** → keeps each reuse step small → stable

中文：
- 一次 iteration ＝ 三階段，反覆循環直到訓練完成：
  1. **Rollout**：actor 跑 N 步，存 s, a, logπ_old, r, V, done
  2. **GAE**：反向掃描 → 優勢 Â、目標回報 R̂（凍結網路）
  3. **Optimize**：打散，用**同一批**資料做 K 個 epoch 的 minibatch 更新
- 相對 A2C：第 3 階段把**同一批資料重複利用 K 個 epoch**，而不是更新一次就丟 → 大幅提升效率
- 但重複利用 on-policy 資料，正是搞垮傳統 policy gradient 的元兇——**那 PPO 怎麼保持安全？**
- **兩招，接下來數學都會講：**
  - **重要性採樣（rₜ）** → 讓重複利用在數學上正確
  - **Clip 機制** → 讓每次重複利用的更新都很小 → 穩定

**Script / 講稿**
> "Before the math, let me show you how PPO trains — it's a loop with three stages. Stage one, rollout: the current actor plays N steps and we store states, actions, the old log-probabilities, rewards, values, and done flags. Stage two, GAE: with the network frozen, a backward pass gives advantages and returns. Stage three, optimize: we shuffle that batch and train on it. Now here's the thing you'll notice coming from A2C — stage three reuses the same batch for K epochs, instead of throwing it away after one update. That's a big efficiency win, but it should make you nervous: reusing on-policy data is exactly what breaks vanilla policy gradient. So how does PPO get away with it? Two tricks — importance sampling and clipping — and that's precisely what the next few slides, the math, are about. Keep this loop in mind; we're about to fill in the two tricks that make it safe."

中文：
> 「進數學前，先讓我給你看 PPO 怎麼訓練——它是一個三階段的迴圈。階段一，rollout：當前的 actor 跑 N 步，把狀態、動作、舊的 log 機率、獎勵、價值、done 旗標都存下來。階段二，GAE：把網路凍結，一次反向掃描算出優勢和回報。階段三，optimize：把那批資料打散、拿來訓練。重點來了——從 A2C 過來的你會注意到，第三階段把同一批資料重複利用 K 個 epoch，而不是更新一次就丟。這是很大的效率提升，但它應該讓你緊張：重複利用 on-policy 資料，正是搞垮傳統 policy gradient 的元兇。那 PPO 怎麼做到的？兩招——重要性採樣和 clip——而這正是接下來數學那幾頁要講的。把這個迴圈記在腦裡，我們馬上來補上讓它安全的兩招。」

---

## Slide 3 — The Math at a Glance / 數學總覽（先看全貌）

**On slide / 投影片內容**
- The two tricks plus two more pieces — four formulas total. We'll walk through each next.
  1. **Probability ratio** — how much the policy changed
     $$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$$
  2. **Advantage (GAE)** — was the action better than expected?
     $$\hat{A}_t = \delta_t + \gamma\lambda\,\hat{A}_{t+1},\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
  3. **Clipped surrogate objective** — the safe policy update
     $$L^{CLIP} = \hat{\mathbb{E}}_t\big[\min(r_t\hat{A}_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)\big]$$
  4. **Total objective** (the thing we *maximize*) — actor + critic + exploration
     $$L^{obj} = L^{CLIP} - c_1 L^{VF} + c_2 S[\pi_\theta]$$

中文：
- 兩招再加兩塊——總共四個公式，接下來逐一拆解：
  1. 機率比率——策略改變了多少
  2. 優勢（GAE）——這動作比預期好嗎？
  3. 裁剪代理目標——安全的策略更新
  4. 總目標（要**最大化**的）——actor + critic + 探索

**Script / 講稿**
> "We just saw the loop and the two tricks that keep it safe — here's all the math on one slide. PPO is really just four pieces: the ratio that measures how much the policy changed, an advantage that measures whether an action was good, the clipped objective that updates the policy safely, and a total objective that ties in the critic and exploration — that's what we maximize. Don't worry about the symbols yet — I'll take them one at a time. Keep this slide as the map."

中文：
> 「我們剛看完迴圈、還有讓它安全的兩招——這頁把整個數學放在一起。PPO 其實就四塊：量化策略改變多少的 ratio、量化動作好不好的 advantage、安全更新策略的裁剪目標、把 critic 和探索綁進來、要最大化的總目標。符號先別擔心，我會一個一個講。這頁當作地圖。」

---

## Slide 4 — Piece 1: The Probability Ratio / 第一塊：機率比率

**On slide / 投影片內容**
- $$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$$
- Numerator = prob the **new** (being-trained) policy gives the action
- Denominator = prob the **old** policy (that collected the data) gave it
- $r_t = 1$ unchanged · $r_t > 1$ new policy prefers it more · $r_t < 1$ less
- This ratio is **importance sampling** — it corrects for the fact that we're reusing data collected by the old policy
- Computed in **log-space** (subtract log-probs, then exponentiate) for numerical stability — probabilities can be tiny

中文：
- 公式如上：新策略機率 ÷ 舊策略機率
- 分子 ＝ 正在訓練的**新**策略給這動作的機率
- 分母 ＝ 收集資料的**舊**策略給的機率
- r=1 沒變 · r>1 新策略更偏好 · r<1 更不偏好
- 這個比率就是**重要性採樣**——它修正「我們在重複利用舊策略收集的資料」這件事
- 用 **log 空間**計算（log 機率相減再取 exp）以保數值穩定——機率可能極小

**Script / 講稿**
> "Piece one: the ratio — and this is the first of our two tricks, importance sampling. It's the new policy's probability for an action divided by the old policy's probability for that same action. One means nothing changed; above one means the policy now likes that action more. Because we're reusing data the old policy collected, this ratio is exactly what corrects for the mismatch — it's what makes the reuse mathematically valid. We compute it in log-space for numerical stability. This ratio is also the quantity PPO will clip — the second trick, coming up."

中文：
> 「第一塊：ratio——這也是我們兩招中的第一招，重要性採樣。它是新策略對某動作的機率，除以舊策略對同一動作的機率。等於 1 代表沒變；大於 1 代表策略現在更喜歡那動作。因為我們在重複利用舊策略收集的資料，這個比率正是修正這個落差的東西——它讓重複利用在數學上成立。我們在 log 空間算以保數值穩定。這個 ratio 也是 PPO 接下來要 clip 的量——也就是第二招。」

---

## Slide 5 — Piece 2: Advantage & GAE / 第二塊：優勢與 GAE

**On slide / 投影片內容**
- **Advantage** $\hat{A}_t$ = how much better the action was than the critic's baseline V(s)
  - $\hat{A}_t > 0$ → reinforce · $\hat{A}_t < 0$ → discourage
- **How do we estimate Â? Two extremes — a bias–variance trade-off:**
  - **one-step TD** — one real reward + the critic's value guess V(s′) (*bootstrapping*) → **low variance, high bias**
  - **Monte-Carlo** — real rewards all the way to episode end, no guessing → **high variance, low bias** (unbiased)
  - **GAE-λ** — weighted blend of 1-step, 2-step, … whole-episode estimates; λ→0 ≈ TD, λ→1 ≈ MC (we use λ=0.95):
  $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\quad(\text{TD error})$$
  $$\hat{A}_t = \delta_t + \gamma\lambda\,\hat{A}_{t+1}\quad(\text{recursive, backward in time})$$
- **What λ controls:** unrolling the recursion gives $\hat{A}_t = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}$ — an *exponentially-weighted* sum of future TD errors, so more distant (noisier) steps are down-weighted; λ is literally the variance↔bias dial
- Returns for the critic: $\hat{R}_t = \hat{A}_t + V(s_t)$
- Computed in one backward pass over the rollout; at an episode boundary the recursion resets, so no value bleeds across game-over

中文：
- 優勢 Â ＝ 動作比 Critic 基準 V(s) 好多少（Â>0 強化、Â<0 抑制）
- **怎麼估計 Â？兩個極端——一個偏差–變異權衡：**
  - **單步 TD** —— 一個真實獎勵 + Critic 的價值猜測 V(s′)（*bootstrap 自舉*）→ **低變異、高偏差**
  - **Monte-Carlo** —— 用到 episode 結束的真實獎勵，完全不靠猜測 → **高變異、低偏差**（無偏）
  - **GAE-λ** —— 1 步、2 步、…整局估計的加權混合；λ→0 ≈ TD、λ→1 ≈ MC（我們用 λ=0.95）：
  - δ（TD error）＝ 即時獎勵 + γ×下一狀態價值 − 當前價值
  - Â ＝ δ + γλ×下一步的 Â（沿時間反向遞迴）
- **λ 在控制什麼：**把遞迴展開 → Â_t = Σ_{l≥0} (γλ)^l·δ_{t+l}——對未來 TD error 做**指數加權**，越遠（越吵）的步數權重越小；λ 就是變異↔偏差的旋鈕
- 給 Critic 的 returns：R̂ ＝ Â + V(s)
- 用一次反向掃描算完；在 episode 邊界遞迴會重置，價值不會跨越遊戲結束外溢

**Script / 講稿**
> "Piece two: advantage — was the action better or worse than the critic expected? Positive, we reinforce it; negative, we discourage it. The real question is how to estimate it, and there are two extremes. A one-step TD estimate uses just the next real reward plus the critic's own value guess for everything after — relying on a guess like that is called bootstrapping. It's stable, low variance, but biased by that guess. Monte-Carlo is the opposite extreme: use the real rewards all the way to the end of the episode, no guessing — that's unbiased, but high variance, because it piles up all the randomness of the whole trajectory. GAE generalizes both: it's a weighted blend of the one-step, two-step, all the way up to whole-episode estimates, controlled by lambda — lambda zero is pure TD, lambda one is pure Monte-Carlo, and we use 0.95. If you unroll the recursion, the advantage is just an exponentially-weighted sum of all the future TD errors, with decay gamma-times-lambda — so the further out a step is, the less its noise counts. That one number is the dial that trades variance against bias. Concretely we compute a TD error at each step and accumulate it backward through the rollout. The returns we feed the critic are just advantage plus the old value. And at an episode boundary the recursion resets, so value doesn't bleed across the end of a game."

中文：
> 「第二塊：advantage——這動作比 Critic 預期好還是壞？正的強化、負的抑制。真正的問題是怎麼估計它，這有兩個極端。單步 TD 只用下一個真實獎勵，加上 Critic 自己對後面的價值猜測——這種依賴猜測的做法叫 bootstrap（自舉）。它穩定、變異低，但被那個猜測帶偏。Monte-Carlo 是另一個極端：用到 episode 結束的真實獎勵、完全不猜——無偏，但變異高，因為它累積了整條軌跡的所有隨機性。GAE 把兩者一般化：它是 1 步、2 步、一直到整局估計的加權混合，由 lambda 控制——lambda 為 0 是純 TD、為 1 是純 Monte-Carlo，我們用 0.95。把遞迴展開，advantage 其實就是所有未來 TD error 的指數加權和，衰減率是 gamma 乘 lambda——所以一個步數越遠，它的雜訊影響就越小。這一個數字就是權衡變異與偏差的旋鈕。具體上每步算一個 TD error，再沿 rollout 反向累加。餵給 Critic 的 returns 就是 advantage 加舊的 value。另外在 episode 邊界，遞迴會重置，讓價值不會跨越遊戲結束外溢。」

---

## Slide 6 — Piece 3: The Clipped Surrogate Objective / 第三塊：裁剪代理目標

**On slide / 投影片內容**
- $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(\underbrace{r_t \hat{A}_t}_{\text{unclipped}},\; \underbrace{\text{clip}(r_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t}_{\text{clipped}}\big)\Big]$$
- This is the **second trick** — it answers the puzzle from the pipeline slide: how reuse stays safe
- **Unclipped** = ordinary policy-gradient term
- **Clipped** = ratio forced into $[1-\epsilon, 1+\epsilon]$, with ε = 0.2 → $[0.8, 1.2]$
- The `min` keeps the **pessimistic (smaller)** term → a lower bound that removes any incentive to move the policy too far
- In one line: a **ceiling** on already-likely good moves, a **floor** on already-unlikely bad moves, but a **full corrective gradient** if the policy lurches the wrong way (each case worked through next slide)
- Why this matters: it's the cheap replacement for TRPO's expensive trust-region constraint

中文：
- 公式如上：min 裡有「未裁剪」與「裁剪」兩項
- 這就是**第二招**——它回答了流程那頁的疑問：重複利用為什麼還安全
- 未裁剪 ＝ 一般 policy-gradient 項
- 裁剪 ＝ 比率被壓進 [1−ε, 1+ε]，ε=0.2 → [0.8, 1.2]
- min 取較**悲觀（較小）**的 → 形成下界，移動太遠就沒有任何誘因
- 一句話：對已經好的動作設**天花板**、對已經壞的動作設**地板**，但若策略往**錯方向暴衝**則給**完整的修正梯度**（下一頁逐一說明）
- 重點：這是用便宜方式取代 TRPO 昂貴的 trust-region 約束

**Script / 講稿**
> "Piece three is the heart of PPO, and it's the second trick — the answer to the puzzle from the pipeline slide. We take two versions of the same term: the unclipped one, which is the ordinary policy gradient, and a clipped one where the ratio is forced to stay between 0.8 and 1.2. We then keep the smaller — the pessimistic — of the two. That lower bound is what removes any reward for moving the policy too far in one update, which is exactly what makes reusing the batch for many epochs safe. This single, cheap operation replaces the heavy trust-region machinery that the earlier TRPO algorithm needed. The next slide shows exactly how it brakes the update in each case."

中文：
> 「第三塊是 PPO 的核心，也是第二招——流程那頁疑問的答案。我們對同一項取兩個版本：未裁剪的，也就是一般的 policy gradient；以及裁剪的，比率被強制留在 0.8 到 1.2 之間。然後取兩者中較小、較悲觀的那個。那個下界就是讓『單次更新把策略移動太遠』得不到任何獎勵的關鍵——這正是讓『同一批資料重複利用多個 epoch』安全的原因。這一個便宜的操作，取代了早期 TRPO 需要的笨重 trust-region 機制。下一頁精確說明它在各種情況下怎麼踩煞車。」

---

## Slide 7 — Why Clipping Works — Case Analysis / 裁剪為何有效——分情況分析

**On slide / 投影片內容**
- **Case A — good action ($\hat{A}_t > 0$):**
  - $r_t \le 1+\epsilon$: unclipped = clipped → normal gradient, keep improving
  - $r_t > 1+\epsilon$: the min picks the constant $(1+\epsilon)\hat{A}_t$ → **gradient = 0** → *ceiling*, stop over-committing
- **Case B — bad action ($\hat{A}_t < 0$):**
  - $r_t \ge 1-\epsilon$: normal gradient, keep suppressing
  - $r_t < 1-\epsilon$: the min picks the constant $(1-\epsilon)\hat{A}_t$ → **gradient = 0** → *floor*, stop over-punishing
- **Safety case — wrong direction (bad action, but $r_t$ jumps up > 1+ε):**
  - unclipped $r_t\hat{A}_t$ is *more negative* than clipped → the min keeps the **unclipped** term → large corrective gradient → policy pulled back hard
- **Summary:** ceiling on good moves, floor on bad moves, **no free pass for going the wrong way**

中文：
- 情況 A——好動作（Â>0）：
  - r ≤ 1+ε：未裁剪＝裁剪 → 正常梯度，持續改善
  - r > 1+ε：min 選常數 (1+ε)Â → **梯度=0** → 天花板，停止過度押注
- 情況 B——壞動作（Â<0）：
  - r ≥ 1−ε：正常梯度，持續抑制
  - r < 1−ε：min 選常數 (1−ε)Â → **梯度=0** → 地板，停止過度懲罰
- 安全情況——往錯方向（壞動作但 r 暴增 > 1+ε）：
  - 未裁剪 r·Â 比裁剪項更負 → min 保留**未裁剪**項 → 大修正梯度 → 策略被狠狠拉回
- 總結：好動作設天花板、壞動作設地板、**往錯方向不給通行證**

**Script / 講稿**
> "Let's go case by case. For a good action with positive advantage: while the ratio is within range, it's just the normal gradient and we keep improving. But once the ratio passes 1.2, the min switches to a constant, its gradient is zero, and we stop — a ceiling that prevents over-committing. For a bad action with negative advantage, it's symmetric: below 0.8 the gradient flattens — a floor that prevents over-punishing. Now the clever part, the safety case: if the policy moves the *wrong* way on a bad action and the ratio shoots up, the unclipped term is even more negative, so the min keeps it — and we get a big gradient pulling the policy back. So: a ceiling, a floor, but no free pass for moving in the wrong direction."

中文：
> 「我們一個情況一個情況看。好動作、advantage 為正：比率在範圍內時就是正常梯度，持續改善。但一旦比率超過 1.2，min 切換成常數，梯度為零，我們就停——一個天花板，防止過度押注。壞動作、advantage 為負則對稱：低於 0.8 梯度變平——一個地板，防止過度懲罰。接著是巧妙的安全情況：如果策略對壞動作往『錯』方向移動、比率暴衝，未裁剪項會更負，所以 min 保留它——我們得到一個大梯度把策略拉回。所以：設天花板、設地板，但往錯方向不給通行證。」

---

## Slide 8 — Piece 4: The Complete Loss / 第四塊：完整損失函數

**On slide / 投影片內容**
- The paper states an **objective to maximize**:
  $$L^{obj} = L^{CLIP} - c_1 L^{VF} + c_2 S[\pi_\theta]$$
- PyTorch optimizers **minimize**, so the code negates it — the actual **loss**:
  $$L^{loss} = -\,L^{CLIP} + c_1 L^{VF} - c_2 S[\pi_\theta]$$
- Three terms (identical math, only the sign convention flips):
  - **$L^{CLIP}$** — clipped surrogate → trains the **actor**
  - **$L^{VF}$** — MSE between critic V(s) and GAE returns → trains the **critic** (c₁ = vf_coef = 0.5)
  - **$S$** — policy entropy → keeps exploration alive, prevents premature collapse (c₂ = ent_coef = 0.01)
- This is *why* the value term is **subtracted** in the objective but **added** in the code's loss — same equation, opposite sign

中文：
- 論文寫的是**要最大化的目標**：L^obj = L^CLIP − c₁·L^VF + c₂·S
- PyTorch 優化器是**最小化**，所以程式碼取負號——實際的**損失**：L^loss = −L^CLIP + c₁·L^VF − c₂·S
- 三項（兩式是同一個數學，只有符號慣例相反）：
  - L^CLIP ＝ 裁剪代理目標 → 訓練 **actor**
  - L^VF ＝ Critic V(s) 與 GAE returns 的均方誤差 → 訓練 **Critic**（c₁=vf_coef=0.5）
  - S ＝ 策略熵 → 維持探索、避免太早崩塌（c₂=ent_coef=0.01）
- 這就是為什麼價值項在「目標」裡是**減**、在程式碼「損失」裡是**加**——同一條式子、相反的符號

**Script / 講稿**
> "Piece four ties it together. There are three terms: the clipped objective trains the actor; a mean-squared-error term trains the critic to predict the GAE returns; and an entropy bonus keeps the policy a little random so it keeps exploring instead of collapsing too early. One detail that often draws a question — notice the value term is *subtracted* in the paper's objective but *added* in the code. That's only because the paper writes an objective to *maximize* while PyTorch *minimizes*, so the code negates the whole thing and every sign flips. Same equation, opposite convention. The coefficients, 0.5 and 0.01, are the standard values I kept from SB3."

中文：
> 「第四塊把全部綁起來。三項：裁剪目標訓練 actor；一個均方誤差項訓練 critic 預測 GAE returns；一個熵獎勵讓策略保持一點隨機，持續探索而不會太早崩塌。一個常被問的細節——注意價值項在論文的『目標』裡是減、在程式碼裡卻是加。那只是因為論文寫的是要『最大化』的目標，而 PyTorch 是『最小化』，所以程式碼把整個取負號、每一項符號都翻過來。同一條式子、相反的慣例。係數 0.5 和 0.01 是我沿用 SB3 的標準值。」

---

## Slide 9 — Implementation #1: Action Masking / 實作一：動作遮罩

**On slide / 投影片內容**
- Action space = **Discrete(192)**: action = piece_idx × 64 + row × 8 + col
- At any state most of the 192 are **illegal** (out of bounds, cell filled, slot used)
- Mask before sampling: set illegal actions' logits to **−∞** → they get **0 probability & 0 entropy**
- The env exposes a legality mask over the 192 actions; game-over = no legal action remains

中文：
- 動作空間 ＝ Discrete(192)：action ＝ 棋子索引 × 64 + 列 × 8 + 行
- 任一狀態下 192 個大多**非法**（超界、格子已滿、slot 已用）
- 取樣前先遮罩：把非法動作的 logit 設成 **−∞** → 它們**機率 0、熵 0**
- 環境提供一個涵蓋 192 個動作的合法性遮罩；遊戲結束 ＝ 沒有任何合法動作

**Script / 講稿**
> "Now the implementation, and the single most important detail is action masking. We have 192 possible actions but at any moment most are illegal — out of bounds, or the cell's already filled. Before every sampling step I set the illegal actions' scores to negative infinity, so after softmax they get exactly zero probability and contribute zero entropy. Without this the policy would waste most of its capacity just learning not to make illegal moves. The same mask doubles as game-over detection: if nothing is legal, the episode is done."

中文：
> 「進到實作，最重要的一個細節就是動作遮罩。我們有 192 個可能動作，但任何時刻大多非法——超出邊界、或格子已經填了。每次取樣前我把非法動作的分數設成負無窮，softmax 後它們機率剛好為零、熵貢獻也是零。沒有這步，策略會把大半能力浪費在學『不要下非法步』。同一個遮罩還兼任遊戲結束偵測：沒有任何合法步，episode 就結束。」

---

## Slide 10 — Implementation #2: My PPO Setup — Custom, not SB3 / 實作二：我的 PPO 設定——自寫而非 SB3

**On slide / 投影片內容**
- **Custom PPO, not SB3** — we must share the *exact* CNN backbone with the DQN member for a fair comparison; SB3 won't allow that cleanly
- But all hyperparameters **match SB3 MaskablePPO defaults** → "neither method was hand-tuned to win"
- Concrete settings: 8 parallel envs × 128 steps = **1024 transitions/iter**, 10 epochs × 64 minibatch, advantages normalized per minibatch, linear LR decay (3e-4 → 0)
- Health checks each update: **approx-KL, clip fraction, explained variance**
- Key hyperparams: γ=0.99, λ=0.95, ε=0.2, lr=3e-4, vf_coef=0.5, ent_coef=0.01

中文：
- **自寫 PPO，不用 SB3**——必須和 DQN 組員共用**完全相同**的 CNN 主幹才能公平比較；SB3 沒辦法乾淨替換
- 但所有超參數**對齊 SB3 MaskablePPO 預設值** →「兩種方法都沒為了贏而特調」
- 具體設定：8 平行環境 × 128 步 ＝ **每輪 1024 筆**、10 epoch × 每批 64、advantage 每批標準化、線性 LR 衰減（3e-4→0）
- 每次更新的健康指標：**approx-KL、clip fraction、explained variance**
- 關鍵超參數：γ=0.99, λ=0.95, ε=0.2, lr=3e-4, vf_coef=0.5, ent_coef=0.01

**Script / 講稿**
> "One design decision worth calling out: I wrote PPO from scratch instead of using Stable-Baselines3. The reason is fairness — our project compares PPO to DQN, and that's only meaningful if both share the identical network backbone, which SB3 doesn't allow cleanly. To stay honest I matched every hyperparameter to SB3's defaults, so neither side was secretly tuned to win. Each iteration collects 1024 steps across 8 parallel games and reuses them for ten epochs. I also log a few health metrics every update — approximate KL, clip fraction, explained variance — to confirm training stays stable."

中文：
> 「一個值得特別提的設計決定：我從零寫了 PPO，而不是用 Stable-Baselines3。原因是公平——我們專案要比較 PPO 和 DQN，只有兩者共用完全相同的網路主幹才有意義，而 SB3 沒辦法乾淨做到。為了誠實，我把每個超參數都對齊 SB3 預設值，兩邊都沒有偷偷為了贏而調。每輪在 8 個平行遊戲收集 1024 步、重複利用十個 epoch。我每次更新還會記錄幾個健康指標——近似 KL、clip fraction、explained variance——確認訓練保持穩定。」

---

## Slide 11 — Implementation #3: The Shared Backbone / 實作三：共享主幹

**On slide / 投影片內容**
- The base actor–critic network, **shared with the DQN member**
- Forward path:
  - board (1,8,8) → 2× Conv 3×3 → (64,8,8)
  - pieces (3,5,5) → weight-shared encoder → pad → (48,8,8)
  - fuse via conv → flatten → FC(4096→128) → concat pieces_left(3) → FC(131→128)
  - → **actor head** (192 logits) + **critic head** (1 value)
- **Spatial fusion is performed while features retain their 2-D topology (8×8)** — the network lines up "empty cell here" with "this piece covers here" *before* flattening, instead of recovering board geometry from a flat vector
- Any PPO-vs-DQN gap is attributable to the *algorithm*, not the encoder

中文：
- 基礎 actor-critic 網路，**與 DQN 組員共用**
- 前向路徑：
  - 棋盤 (1,8,8) → 2 層 3×3 卷積 → (64,8,8)
  - 棋子 (3,5,5) → 權重共享 encoder → 補零 → (48,8,8)
  - 卷積融合 → 攤平 → FC(4096→128) → 接 pieces_left(3) → FC(131→128)
  - → actor head（192 logits）＋ critic head（1 value）
- **在特徵仍保留二維拓撲（8×8）時進行空間融合**——網路在攤平**之前**就把「這裡有空格」和「這塊棋子蓋住這裡」對齊，而不必從攤平後的向量重建棋盤幾何
- PPO 與 DQN 的差距可歸因於**演算法**，而非編碼器

**Script / 講稿**
> "Here's the network — the base actor-critic, and the same backbone the DQN member uses. The board passes through two convolutions; the three pieces go through a weight-shared encoder, get padded to the same spatial size, and a fusion convolution combines them while everything is still 2-D — so the network lines up empty cells with piece shapes positionally, before any flattening. We append the 'pieces left' flags, then split into the actor and critic heads. Because this encoder is identical on both sides, any performance gap between PPO and DQN comes from the algorithm, not the network. Everything in the next two slides is built on top of this."

中文：
> 「這是網路——基礎 actor-critic，也是 DQN 組員用的同一個主幹。棋盤經過兩層卷積；三塊棋子經過權重共享的 encoder、補零到相同空間大小，一個融合卷積在還是二維時把它們結合——讓網路在攤平前就能把空格和棋子形狀在位置上對齊。我們接上『剩餘棋子』旗標，再分成 actor 和 critic 兩個 head。因為這個編碼器兩邊完全相同，PPO 和 DQN 的任何表現差距都來自演算法，而非網路。接下來兩頁的東西都建在這之上。」

---

## Slide 12 — Variant 1: Heuristic-Aware Actor–Critic / 變體一：加入啟發式特徵

**On slide / 投影片內容**
- Base backbone **+ 40-d hand-crafted features**
- The env emits 40 normalized features: column heights, holes, row/col fill, bumpiness, #legal moves
- Fed through a small MLP (40→32), concatenated into the shared layer:
  fusion(128) + pieces_left(3) + heuristics(32) → FC(163→128)
- **Why:** the agent no longer has to *learn* "this column is too tall" from raw input — we hand it the answer
- Added without touching the shared backbone → comparison stays fair

中文：
- 基礎主幹 **+ 40 維手工特徵**
- 環境輸出 40 個正規化特徵：各欄高度、洞、列/欄填充、崎嶇度、合法步數
- 經小 MLP (40→32)，串接進共享層：fusion(128)+pieces_left(3)+heuristics(32)→FC(163→128)
- 為什麼：agent 不必再從原始輸入自己**學**「這欄太高」——直接把答案給它
- 在不改共享主幹的前提下加入 → 比較仍公平

**Script / 講稿**
> "The first variant adds hand-crafted features. The environment already computes 40 useful numbers — column heights, holes, how bumpy the surface is, how many legal moves remain. Rather than forcing the CNN to rediscover these from raw input, I feed them through a small MLP and merge them into the shared layer, giving the agent a head start. Crucially I add this alongside the shared backbone without modifying it, so the PPO-versus-DQN comparison stays fair."

中文：
> 「第一個變體加入手工特徵。環境本來就會算出 40 個有用的數字——各欄高度、洞、表面多崎嶇、還剩幾個合法步。與其逼 CNN 從原始輸入重新發現，我透過一個小 MLP 把它們餵進去、合併到共享層，給 agent 起跑優勢。關鍵是我在不修改共享主幹的前提下『額外』加上它，所以 PPO 對 DQN 的比較仍公平。」

---

## Slide 13 — Variant 2: Afterstate Evaluation / 變體二：後繼狀態評估

**On slide / 投影片內容**
- A tiny network — only **~2.7k parameters**
- Idea (Chen et al. 2026, arXiv:2603.26765): don't score the *current* state — score the board **after** each placement
- For each of 192 actions, simulate it and extract **9 features**: lines cleared, eroded cells, holes, max/mean height, bumpiness, row/col transitions, near-full lines
- **Actor**: one shared linear layer (9→1) scores all actions → 192 logits
- **Critic**: 40-d heuristics → small MLP → V(s)
- Features kept at **raw integer scale on purpose** — the actor is one small-init linear layer; squashing features to [0,1] compresses the logit differences below the entropy bonus, freezing the policy into a **uniform random distribution** — it never clears the **entropy floor**

中文：
- 一個極小網路——只有 **約 2.7k 參數**
- 想法（Chen et al. 2026, arXiv:2603.26765）：不評估「現在」狀態——評估放下每塊棋「**之後**」的棋盤
- 對 192 個動作各自模擬，抽取 **9 個特徵**：消的行數、被消掉的棋子格、洞、最大/平均高度、崎嶇度、列/欄轉換、接近全滿的行
- Actor：一個共享線性層（9→1）為所有動作打分 → 192 logits
- Critic：40 維啟發式 → 小 MLP → V(s)
- 特徵**刻意保留原始整數尺度**——actor 只是一個小初始化的線性層；把特徵壓到 [0,1] 會把 logit 差距壓到小於熵獎勵，策略就凍結成**均勻隨機分布（uniform random distribution）**、跨不過**熵地板（entropy floor）**

**Script / 講稿**
> "The second variant is the most interesting, and it's tiny — under three thousand parameters. The idea, from a 2026 paper, is to stop scoring the current state and instead look one move ahead: for every legal action, simulate the placement and describe the resulting board with nine simple features, like lines cleared and holes created. A single shared linear layer scores all 192 actions. One subtle but important point: I deliberately don't normalize these features. The actor is a single linear layer with small initial weights, so if the features were squeezed into zero-to-one, every action's score would start almost identical — those tiny gaps fall below the entropy bonus, which actively rewards a uniform policy, and training never escapes random play. Keeping the raw integer scale — clearing two lines really is a 2, not 0.13 — gives the scorer a wide enough gap to break through that entropy floor and tell good moves from bad from the very first updates."

中文：
> 「第二個變體最有趣，而且非常小——不到三千個參數。想法來自一篇 2026 年的論文：別再評估現在的狀態，改成往前看一步：對每個合法動作，模擬放置，再用九個簡單特徵描述結果棋盤，像消了幾行、製造幾個洞。一個共享的線性層為全部 192 個動作打分。一個微妙但重要的點：我刻意不正規化這些特徵。actor 只是一個小初始化的線性層，如果把特徵壓到 0 到 1，每個動作的分數一開始幾乎一樣——這些微小差距會小於熵獎勵，而熵獎勵會主動獎勵均勻策略，訓練就永遠跳不出隨機。保留原始整數尺度——消兩行真的就是 2，不是 0.13——讓打分層有夠大的差距突破那個熵地板，從第一次更新就分辨好步壞步。」

---

## Slide 14 — Recap / 總結

**On slide / 投影片內容**
- **PPO** = model-free, policy-based, on-policy actor–critic; same family as A2C but safer & more efficient
- **Two ideas make it work**: importance sampling (rₜ) → reuse data; clipping → stable updates
- **Math (4 pieces)**: ratio → advantage (GAE) → clipped objective → 3-term loss
- **Implementation**:
  - Action masking (mandatory — 192 actions, mostly illegal)
  - From-scratch PPO, SB3-matched hyperparameters (fair vs DQN)
  - Shared CNN backbone
- **Two variants**: heuristic features · afterstate evaluation
- *(Results compared in the next section.)*

中文：
- PPO ＝ model-free、policy-based、on-policy 的 actor-critic；與 A2C 同類但更安全、更高效
- 兩個關鍵想法：重要性採樣（rₜ）→ 重複利用資料；clip → 穩定更新
- 數學（四塊）：ratio → advantage（GAE）→ 裁剪目標 → 三項損失
- 實作：
  - 動作遮罩（必要——192 動作大多非法）
  - 從零寫的 PPO，超參數對齊 SB3（與 DQN 公平）
  - 共享 CNN 主幹
- 兩個變體：啟發式特徵 · 後繼狀態評估
- （結果在下一章節比較。）

**Script / 講稿**
> "To wrap up: PPO is a model-free, policy-based, on-policy actor–critic — the same family as A2C, but safer and more sample-efficient thanks to two ideas: importance sampling, which lets us reuse each batch, and clipping, which keeps those updates stable. The math is four pieces: ratio, advantage, the clipped objective, and the three-term loss. I implemented it from scratch to share a backbone with DQN, kept the hyperparameters honest by matching SB3, and built two variants on top — one with hand-crafted features, one with afterstate evaluation. The next section compares how they actually performed."

中文：
> 「總結：PPO 是 model-free、policy-based、on-policy 的 actor-critic——和 A2C 同類，但因為兩個想法而更安全、更省樣本：重要性採樣讓我們重複利用每批資料，clip 讓那些更新保持穩定。數學就四塊：ratio、advantage、裁剪目標、三項損失。我從零實作它以便和 DQN 共用主幹，透過對齊 SB3 讓超參數誠實，並在上面做了兩個變體——一個加手工特徵、一個用後繼狀態評估。下一章節比較它們實際表現。」

---

## Trimming guide / 砍時間指南

To shorten, merge/drop in this order:
要縮短，依序合併或刪除：

1. **Slide 7** (case analysis) → drop it; the one-line case summary now lives on Slide 6, so the punchline survives.
   第 7 頁（分情況分析）→ 直接刪；那句案例總結現在已放在第 6 頁，重點不會丟。
2. **Slides 12 + 13** → one "Two Variants" slide, one bullet each.
   第 12+13 頁 → 合成一頁「兩個變體」，各一個 bullet。
3. **Slide 4** (ratio detail) → already on the Slide 3 map; drop if tight.
   第 4 頁（ratio 細節）→ 已在第 3 頁總覽圖上；時間緊就刪。
4. Spine to keep: Slides 0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 14.
   保留主幹：第 0、1、2、3、5、6、8、9、10、11、14 頁。
